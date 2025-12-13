#!/usr/bin/env python3
"""
Enhanced Dynamic Quiz Solver - Production Ready v6.0
Handles any quiz type with improved reliability and error handling
"""

import os
import json
import time
import requests
import re
import io
import traceback
import base64
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from openai import OpenAI
from dotenv import load_dotenv

import logging
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import pdfplumber

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

EMAIL = os.getenv("STUDENT_EMAIL")
SECRET = os.getenv("STUDENT_SECRET")
AIMLAPI_BASE_URL = os.getenv("AIMLAPI_BASE_URL", "https://aipipe.org/openai/v1")
AIMLAPI_API_KEY = os.getenv("AIMLAPI_API_KEY")
AIMLAPI_MODEL = os.getenv("AIMLAPI_MODEL", "gpt-4o")

CHROME_BINARY = os.getenv("CHROME_BIN", "/usr/bin/chromium")
CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")

# Initialize OpenAI client
client = None
if AIMLAPI_API_KEY:
    try:
        client = OpenAI(api_key=AIMLAPI_API_KEY, base_url=AIMLAPI_BASE_URL)
        logger.info("✅ LLM client initialized")
    except Exception as e:
        logger.error(f"❌ LLM init failed: {e}")
else:
    logger.error("❌ AIMLAPI_API_KEY not set!")

# Cache for downloaded files
file_cache: Dict[str, bytes] = {}

# ============================================================================
# BROWSER SETUP
# ============================================================================

def setup_browser() -> webdriver.Chrome:
    """Initialize headless Chrome with optimal settings"""
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option('useAutomationExtension', False)
    opts.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-plugins")
    
    if CHROME_BINARY and os.path.exists(CHROME_BINARY):
        opts.binary_location = CHROME_BINARY
    
    service = Service(executable_path=CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(45)
    driver.set_script_timeout(30)
    return driver

# ============================================================================
# PAGE CONTENT EXTRACTION
# ============================================================================

def extract_page_content(url: str, max_wait: int = 10) -> Dict[str, Any]:
    """
    Extract content from any webpage with intelligent waiting
    """
    logger.info(f"🌐 Extracting: {url}")
    driver = setup_browser()
    
    try:
        driver.get(url)
        
        # Wait for body to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Progressive waiting - check if content stabilizes
        previous_text = ""
        previous_html_length = 0
        stable_count = 0
        
        # Keywords that indicate quiz content has loaded
        critical_keywords = ["result", "quiz", "question", "data", "download", "submit", "answer", "command", "file"]
        
        for i in range(max_wait):
            time.sleep(1)
            
            # Get content via JavaScript (more reliable)
            try:
                current_text = driver.execute_script("return document.body.innerText;")
                current_html_length = len(driver.page_source)
            except:
                current_text = driver.find_element(By.TAG_NAME, "body").text
                current_html_length = len(driver.page_source)
            
            # Check if critical content exists
            has_critical_content = any(keyword in current_text.lower() for keyword in critical_keywords)
            
            # Check if content has stabilized
            if (current_text == previous_text and 
                current_html_length == previous_html_length and 
                len(current_text) > 50 and
                has_critical_content):
                stable_count += 1
                if stable_count >= 2:
                    logger.info(f"✅ Content stabilized after {i+1}s")
                    break
            else:
                stable_count = 0
            
            previous_text = current_text
            previous_html_length = current_html_length
        
        # Extra wait for any remaining dynamic content
        time.sleep(2)
        
        # Final content extraction
        try:
            page_text = driver.execute_script("return document.body.innerText;")
        except:
            page_text = driver.find_element(By.TAG_NAME, "body").text
        
        page_source = driver.page_source
        
        # Extract all links
        links = []
        try:
            for link_element in driver.find_elements(By.TAG_NAME, "a"):
                try:
                    href = link_element.get_attribute("href")
                    if href and not href.startswith(("javascript:", "mailto:", "#", "data:")):
                        absolute_url = urljoin(url, href)
                        if absolute_url not in links and absolute_url != url:
                            links.append(absolute_url)
                except:
                    continue
        except Exception as e:
            logger.warning(f"⚠️ Error extracting links: {e}")
        
        logger.info(f"✅ Extracted: {len(page_text)} chars, {len(links)} links")
        
        return {
            "html": page_source,
            "text": page_text,
            "links": links,
            "url": url
        }
        
    except Exception as e:
        logger.error(f"❌ Extraction error: {e}")
        logger.error(traceback.format_exc())
        return {
            "html": "",
            "text": f"Error extracting page: {str(e)}",
            "links": [],
            "url": url,
            "error": str(e)
        }
    finally:
        try:
            driver.quit()
        except:
            pass

# ============================================================================
# FILE DOWNLOADING
# ============================================================================

def download_file(url: str, timeout: int = 30, use_cache: bool = True) -> Optional[bytes]:
    """Download any file with caching support"""
    
    # Check cache first
    if use_cache and url in file_cache:
        logger.info(f"📦 Using cached file: {url}")
        return file_cache[url]
    
    try:
        logger.info(f"📥 Downloading: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        
        content = resp.content
        logger.info(f"✅ Downloaded {len(content)} bytes from {url}")
        
        # Cache the file
        if use_cache:
            file_cache[url] = content
        
        return content
        
    except Exception as e:
        logger.error(f"❌ Download error for {url}: {e}")
        return None

# ============================================================================
# FILE PARSING
# ============================================================================

def smart_parse_file(file_content: bytes, url: str) -> Dict[str, Any]:
    """
    Intelligently parse any file type - FIXED ORDER
    """
    result = {"type": "unknown", "content": None, "error": None, "url": url}
    
    try:
        url_lower = url.lower()
        
        # CRITICAL: Check file extensions FIRST before content sniffing
        
        # Audio files - BEFORE any content checks
        if any(url_lower.endswith(ext) for ext in ['.opus', '.mp3', '.wav', '.ogg', '.m4a', '.flac']):
            result["type"] = "audio"
            result["content"] = {
                "base64": base64.b64encode(file_content).decode('utf-8'),
                "size": len(file_content),
                "format": url.split('.')[-1].lower(),
                "note": "Audio file - requires transcription service (not available)"
            }
            logger.info(f"✅ Audio file: {url.split('.')[-1].upper()}")
            return result
        
        # Image files - BEFORE any content checks  
        if any(url_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
            result["type"] = "image"
            # Analyze dominant color using PIL
            try:
                from PIL import Image
                from collections import Counter
                img = Image.open(io.BytesIO(file_content)).convert('RGB')
                img_resized = img.resize((150, 150))  # Resize for faster processing
                pixels = list(img_resized.getdata())
                color_counts = Counter(pixels)
                dominant_rgb = color_counts.most_common(1)[0][0]
                dominant_hex = '#{:02x}{:02x}{:02x}'.format(dominant_rgb[0], dominant_rgb[1], dominant_rgb[2])
                
                result["content"] = {
                    "base64": base64.b64encode(file_content).decode('utf-8'),
                    "size": len(file_content),
                    "format": url.split('.')[-1].lower(),
                    "dominant_color": dominant_hex,
                    "dominant_rgb": dominant_rgb
                }
                logger.info(f"✅ Image: dominant color {dominant_hex}")
            except Exception as e:
                logger.error(f"❌ Image analysis error: {e}")
                result["content"] = {
                    "base64": base64.b64encode(file_content).decode('utf-8'),
                    "size": len(file_content),
                    "format": url.split('.')[-1].lower(),
                    "error": str(e)
                }
            return result
        
        # JSON - Check extension BEFORE content
        if url_lower.endswith('.json'):
            result["type"] = "json"
            try:
                result["content"] = json.loads(file_content.decode('utf-8'))
            except Exception as e:
                result["error"] = f"Invalid JSON: {e}"
            return result
        
        # Excel - Check extension
        if url_lower.endswith(('.xlsx', '.xls')):
            result["type"] = "excel"
            result["content"] = parse_excel(file_content)
            return result
        
        # PDF - Check magic bytes OR extension
        if file_content[:4] == b"%PDF" or url_lower.endswith('.pdf'):
            result["type"] = "pdf"
            result["content"] = parse_pdf(file_content)
            return result
        
        # CSV - Check extension OR content (LAST!)
        if url_lower.endswith('.csv') or (b',' in file_content[:1000] and not url_lower.endswith(('.json', '.txt'))):
            result["type"] = "csv"
            result["content"] = parse_csv(file_content)
            return result
        
        # Text fallback
        if url_lower.endswith(('.txt', '.text', '.md')):
            result["type"] = "text"
            try:
                result["content"] = file_content.decode('utf-8')
            except:
                result["content"] = file_content.decode('utf-8', errors='ignore')
            return result
        
        # Final fallback - try as text
        result["type"] = "text"
        try:
            result["content"] = file_content.decode('utf-8', errors='ignore')
        except Exception as e:
            result["error"] = f"Could not decode: {e}"
            result["content"] = f"Binary content: {len(file_content)} bytes"
        
        logger.info(f"✅ Parsed as: {result['type']}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Parse error: {e}")
        result["error"] = str(e)
        return result

def parse_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    """Parse PDF - extract ALL text and tables"""
    try:
        result = {
            "pages": [],
            "all_text": "",
            "all_tables": [],
            "page_count": 0
        }
        
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            result["page_count"] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_data = {"page_number": page_num}
                
                # Extract text
                text = page.extract_text()
                if text:
                    page_data["text"] = text
                    result["all_text"] += f"\n--- Page {page_num} ---\n{text}"
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    page_data["tables"] = []
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0:
                            try:
                                # Detect if first row looks like headers
                                first_row = table[0]
                                looks_like_header = (
                                    len(table) > 1 and 
                                    all(isinstance(cell, str) and cell and 
                                        not (str(cell).replace('.','').replace('-','').replace(',','').isdigit())
                                        for cell in first_row if cell)
                                )
                                
                                # Create DataFrame
                                if looks_like_header:
                                    df = pd.DataFrame(table[1:], columns=table[0])
                                else:
                                    df = pd.DataFrame(table)
                                    df.columns = [f"Column_{i}" for i in range(len(df.columns))]
                                
                                table_info = {
                                    "index": table_idx,
                                    "data": df.to_dict('records'),
                                    "shape": list(df.shape),
                                    "columns": list(df.columns)
                                }
                                page_data["tables"].append(table_info)
                                result["all_tables"].append({
                                    "page": page_num,
                                    **table_info
                                })
                            except Exception as e:
                                logger.warning(f"⚠️ Could not parse table on page {page_num}: {e}")
                
                result["pages"].append(page_data)
        
        logger.info(f"✅ PDF: {result['page_count']} pages, {len(result['all_tables'])} tables")
        return result
        
    except Exception as e:
        logger.error(f"❌ PDF parsing error: {e}")
        return {"error": str(e)}

def parse_csv(csv_content: bytes) -> Dict[str, Any]:
    """Parse CSV with comprehensive statistical analysis"""
    try:
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode("utf-8", errors='ignore')
        
        # Try multiple parsing strategies
        df = None
        separators = [',', ';', '\t', '|']
        
        for sep in separators:
            try:
                df = pd.read_csv(io.StringIO(csv_content), sep=sep)
                if len(df.columns) > 1 or (len(df.columns) == 1 and len(df) > 0):
                    logger.info(f"✅ CSV parsed with separator: '{sep}'")
                    break
            except Exception as e:
                continue
        
        if df is None or df.empty:
            return {"error": "Could not parse CSV with any known separator"}
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Try to convert string numbers to numeric (more conservative approach)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Remove common formatting
                    cleaned = df[col].astype(str).str.replace(r'[$,€£¥\s%]', '', regex=True)
                    numeric_col = pd.to_numeric(cleaned, errors='coerce')
                    
                    # Check if values have leading zeros (likely IDs/codes, not numbers)
                    has_leading_zeros = df[col].astype(str).str.match(r'^0\d').any()
                    
                    # Only convert if 80%+ are numeric AND no leading zeros
                    if (numeric_col.notna().sum() > len(df) * 0.8 and not has_leading_zeros):
                        df[col] = numeric_col
                        logger.info(f"  🔢 Converted '{col}' to numeric")
                except:
                    pass
        
        # Comprehensive column analysis
        column_analysis = {}
        for col in df.columns:
            col_data = df[col]
            col_data_clean = col_data.dropna()
            
            analysis = {
                "dtype": str(col_data.dtype),
                "non_null_count": int(col_data.notna().sum()),
                "null_count": int(col_data.isna().sum()),
                "total_count": len(col_data)
            }
            
            if pd.api.types.is_numeric_dtype(col_data):
                if len(col_data_clean) > 0:
                    analysis.update({
                        "sum": float(col_data_clean.sum()),
                        "mean": float(col_data_clean.mean()),
                        "median": float(col_data_clean.median()),
                        "mode": float(col_data_clean.mode()[0]) if len(col_data_clean.mode()) > 0 else None,
                        "min": float(col_data_clean.min()),
                        "max": float(col_data_clean.max()),
                        "std": float(col_data_clean.std()) if len(col_data_clean) > 1 else 0,
                        "variance": float(col_data_clean.var()) if len(col_data_clean) > 1 else 0,
                        "q25": float(col_data_clean.quantile(0.25)),
                        "q50": float(col_data_clean.quantile(0.50)),
                        "q75": float(col_data_clean.quantile(0.75)),
                        "iqr": float(col_data_clean.quantile(0.75) - col_data_clean.quantile(0.25)),
                        "range": float(col_data_clean.max() - col_data_clean.min()),
                        "count_above_mean": int((col_data_clean > col_data_clean.mean()).sum()),
                        "count_below_mean": int((col_data_clean < col_data_clean.mean()).sum()),
                        "count_above_median": int((col_data_clean > col_data_clean.median()).sum()),
                        "count_below_median": int((col_data_clean < col_data_clean.median()).sum()),
                        "count_zero": int((col_data_clean == 0).sum()),
                        "count_positive": int((col_data_clean > 0).sum()),
                        "count_negative": int((col_data_clean < 0).sum())
                    })
                    logger.info(f"  📊 '{col}': sum={analysis['sum']:.2f}, mean={analysis['mean']:.2f}, rows={len(col_data_clean)}")
            else:
                # String/categorical column
                unique_vals = col_data_clean.unique()
                value_counts = col_data_clean.value_counts()
                analysis.update({
                    "unique_count": len(unique_vals),
                    "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "sample_values": [str(v) for v in unique_vals[:15]],
                    "value_counts": {str(k): int(v) for k, v in value_counts.head(20).items()}
                })
            
            column_analysis[col] = analysis
        
        # Smarter data sampling - include first, last, and middle rows
        if len(df) <= 300:
            data_sample = df.to_dict('records')
        else:
            sample_data = (
                df.head(100).to_dict('records') +  # First 100
                df.tail(100).to_dict('records') +  # Last 100
                df.sample(n=min(100, len(df)-200)).to_dict('records')  # 100 random from middle
            )
            # Remove duplicates while preserving order
            seen = set()
            data_sample = []
            for item in sample_data:
                item_str = json.dumps(item, sort_keys=True)
                if item_str not in seen:
                    seen.add(item_str)
                    data_sample.append(item)
                if len(data_sample) >= 300:
                    break
        
        result = {
            "shape": list(df.shape),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "column_analysis": column_analysis,
            "first_rows": df.head(20).to_dict('records'),
            "last_rows": df.tail(10).to_dict('records'),
            "data_sample": data_sample,
            "raw_content": csv_content[:500]  # Add raw content for debugging
        }
        
        logger.info(f"✅ CSV: {df.shape[0]} rows × {df.shape[1]} columns")
        return result
        
    except Exception as e:
        logger.error(f"❌ CSV parsing error: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "raw_preview": str(csv_content)[:1000] if csv_content else "No content"
        }

def parse_excel(excel_bytes: bytes) -> Dict[str, Any]:
    """Parse Excel files (multiple sheets)"""
    try:
        dfs = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)
        
        result = {"sheets": {}, "sheet_names": list(dfs.keys())}
        
        for sheet_name, df in dfs.items():
            logger.info(f"📄 Parsing Excel sheet: {sheet_name}")
            # Convert to CSV and use CSV parser
            csv_str = df.to_csv(index=False)
            result["sheets"][sheet_name] = parse_csv(csv_str.encode())
        
        logger.info(f"✅ Excel: {len(result['sheets'])} sheets")
        return result
        
    except Exception as e:
        logger.error(f"❌ Excel parsing error: {e}")
        return {"error": str(e)}

# ============================================================================
# SUBMIT URL EXTRACTION
# ============================================================================

def extract_submit_url(content: str, base_url: str) -> Optional[str]:
    """
    Dynamically extract submit URL with multiple strategies
    """
    if not content:
        return None
    
    parsed_base = urlparse(base_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
    
    # Strategy 1: Look for URL in JSON payload examples
    json_patterns = [
        r'"url":\s*"([^"]*submit[^"]*)"',
        r"'url':\s*'([^']*submit[^']*)'",
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            url = matches[0].strip()
            if url.startswith('/'):
                return f"{base_domain}{url}"
            elif url.startswith('http'):
                return url
    
    # Strategy 2: Look for "POST to" or "submit to" patterns
    post_patterns = [
        r'(?:POST|Post|post|submit|Submit)\s+(?:your answer\s+)?(?:to|at)\s+([^\s<>"\']+)',
        r'(?:POST|Post|post)\s+([^\s<>"\']+)',
    ]
    
    for pattern in post_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            match = match.strip()
            if 'submit' in match.lower():
                if match.startswith('/'):
                    return f"{base_domain}{match}"
                elif match.startswith('http'):
                    return match
    
    # Strategy 3: Look for href attributes
    href_patterns = [
        r'href=["\']([^"\']*submit[^"\']*)["\']',
        r'action=["\']([^"\']*submit[^"\']*)["\']',
    ]
    
    for pattern in href_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            url = matches[0].strip()
            if url.startswith('/'):
                return f"{base_domain}{url}"
            elif url.startswith('http'):
                return url
    
    # Strategy 4: Find any URL with /submit in path
    url_pattern = r'(https?://[^\s<>"\']+/submit[^\s<>"\']*)'
    matches = re.findall(url_pattern, content, re.IGNORECASE)
    if matches:
        return matches[0]
    
    # Strategy 5: Fallback - construct from base URL ONLY if we see JSON payload structure
    if 'submit' in content.lower() and ('"url"' in content or '"answer"' in content or 'POST' in content):
        fallback_url = f"{base_domain}/submit"
        logger.info(f"⚠️ Using fallback submit URL: {fallback_url}")
        return fallback_url
    
    logger.warning("❌ No submit URL found in content")
    return None

# ============================================================================
# LLM SOLVING
# ============================================================================

def solve_with_llm(quiz_page: Dict[str, Any], downloaded_files: Dict[str, Any], quiz_url: str, user_email: str) -> Dict[str, Any]:
    """
    Solve quiz with LLM using advanced prompting
    """
    if not client:
        return {"error": "LLM not initialized"}
    
    page_text = quiz_page.get("text", "")
    
    # Calculate email length precisely
    email_length = len(user_email)
    email_mod_2 = email_length % 2
    
    logger.info(f"📧 User email: {user_email}")
    logger.info(f"📏 Email length: {email_length}, mod 2 = {email_mod_2}")
    logger.info(f"📄 Question length: {len(page_text)} chars")
    logger.info(f"📦 Data files: {len(downloaded_files)}")
    
    # Build comprehensive context with smart truncation per file
    context_parts = {}
    total_size = 0
    max_total_size = 40000
    
    for url, file_data in downloaded_files.items():
        file_str = json.dumps(file_data, indent=2, default=str)
        
        # If file is too large, create summary
        if len(file_str) > 5000:
            summary = {
                "type": file_data.get("type"),
                "url": url,
                "size_chars": len(file_str)
            }
            
            # Include key parts based on file type
            if file_data.get("type") == "csv" and "content" in file_data:
                content = file_data["content"]
                summary["columns"] = content.get("columns", [])
                summary["column_analysis"] = content.get("column_analysis", {})
                summary["row_count"] = content.get("row_count", 0)
                summary["first_rows"] = content.get("first_rows", [])[:10]
                summary["last_rows"] = content.get("last_rows", [])[:5]
                summary["raw_content"] = content.get("raw_content", "")[:300]
            elif file_data.get("type") == "pdf" and "content" in file_data:
                content = file_data["content"]
                summary["page_count"] = content.get("page_count", 0)
                summary["all_text"] = content.get("all_text", "")[:3000]
                summary["all_tables"] = content.get("all_tables", [])
            elif file_data.get("type") == "audio":
                summary["audio_format"] = file_data.get("content", {}).get("format")
                summary["audio_size"] = file_data.get("content", {}).get("size")
                summary["note"] = "Audio file detected - transcription service required"
            elif file_data.get("type") == "image":
                content = file_data.get("content", {})
                summary = {
                    "type": "image",
                    "dominant_color": content.get("dominant_color"),
                    "dominant_rgb": content.get("dominant_rgb"),
                    "format": content.get("format"),
                    "note": "Use dominant_color for color-related questions"
                }
            elif file_data.get("type") == "json":
                summary = {
                    "type": "json",
                    "content": file_data.get("content")
                }
            else:
                summary["content_preview"] = str(file_data.get("content", ""))[:3000]
            
            context_parts[url] = summary
            total_size += len(json.dumps(summary))
        else:
            context_parts[url] = file_data
            total_size += len(file_str)
        
        if total_size > max_total_size:
            logger.warning(f"⚠️ Context size limit reached")
            break
    
    context_str = json.dumps(context_parts, indent=2, default=str)
    logger.info(f"📊 Context size: {len(context_str)} chars")
    
    # Create the prompt
    prompt = f"""You are a precise data analyst solving a quiz. Provide the EXACT answer requested.

USER EMAIL: {user_email}
EMAIL LENGTH: {email_length}
EMAIL MOD 2: {email_mod_2}

QUESTION:
{page_text[:8000]}

AVAILABLE DATA:
{context_str}

INSTRUCTIONS:
1. READ carefully - what EXACTLY is asked?
2. Command strings: Return EXACT format, replace <your email> with {user_email}
3. Image colors: Use "dominant_color" hex value (e.g., "#b45a1e")
4. JSON normalization: Compact format NO SPACES - {{"key":"value"}} not {{"key": "value"}}
5. File counting + math: Count files, then ADD {email_mod_2}
6. Audio: State "Audio transcription not available"

ANSWER FORMATS:
- Commands: plain string (no extra escaping)
- Numbers: integer or float
- JSON arrays: COMPACT no spaces [{{"id":1,"name":"A"}}]
- Colors: "#rrggbb"
- Text: plain string

CRITICAL:
✓ Replace <your email> with {user_email}
✓ JSON: NO SPACES between keys/values
✓ File count: Add {email_mod_2} to count
✓ Use ACTUAL data, not examples

Return JSON:
{{
    "answer": <exact_answer>,
    "reasoning": "brief explanation"
}}"""

    try:
        logger.info("🤖 Querying LLM...")
        
        response = client.chat.completions.create(
            model=AIMLAPI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"Precise analyst. Email: {user_email} (len={email_length}, mod2={email_mod_2}). JSON: NO SPACES. File count: add {email_mod_2}. Return valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            #temperature=0.05,
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"🤖 Response: {response_text[:400]}")
        
        response_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        
        try:
            solution = json.loads(response_text)
        except:
            match = re.search(r'\{.*?"answer".*?\}', response_text, re.DOTALL)
            if match:
                solution = json.loads(match.group())
            else:
                return {"error": "Parse failed", "raw": response_text[:500]}
        
        answer = solution.get("answer")
        
        # Fix email placeholder if present
        if answer and isinstance(answer, str) and "<your email>" in answer:
            answer = answer.replace("<your email>", user_email)
            solution["answer"] = answer
        
        # Validate not placeholder
        invalid = ["", None, "N/A", "null", "placeholder", "<your email>"]
        if answer in invalid or str(answer).lower().strip() in invalid:
            logger.warning("⚠️ Invalid answer, retrying...")
            
            retry_prompt = f"""PREVIOUS FAILED. Return ACTUAL answer.

EMAIL: {user_email} (length {email_length}, mod 2 = {email_mod_2})

QUESTION:
{page_text[:5000]}

DATA:
{context_str[:15000]}

For JSON: NO SPACES - {{"id":1,"name":"A"}}
For file count: count + {email_mod_2}
For commands: replace <your email> with {user_email}

Return {{
    "answer": <actual_value>,
    "reasoning": "explanation"
}}"""

            try:
                retry_resp = client.chat.completions.create(
                    model=AIMLAPI_MODEL,
                    messages=[
                        {"role": "system", "content": f"Email: {user_email}. Length: {email_length}. Mod2: {email_mod_2}. JSON: NO SPACES. No placeholders."},
                        {"role": "user", "content": retry_prompt}
                    ],
                    #temperature=0.1,
                )
                
                retry_text = retry_resp.choices[0].message.content.strip()
                retry_text = re.sub(r'```json\s*|\s*```', '', retry_text).strip()
                
                try:
                    solution = json.loads(retry_text)
                    answer = solution.get("answer")
                    if answer and isinstance(answer, str) and "<your email>" in answer:
                        answer = answer.replace("<your email>", user_email)
                        solution["answer"] = answer
                    logger.info(f"✅ Retry success: {answer}")
                except:
                    logger.error("❌ Retry parse failed")
                    return {"error": "Retry failed", "raw": retry_text[:500]}
            except Exception as e:
                logger.error(f"❌ Retry error: {e}")
                return {"error": str(e)}
        
        logger.info(f"✅ Final answer: {str(answer)[:200]}")
        logger.info(f"💡 Reasoning: {solution.get('reasoning', '')[:150]}")
        
        return solution
        
    except Exception as e:
        logger.error(f"❌ LLM error: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# ============================================================================
# ANSWER SUBMISSION
# ============================================================================

def submit_answer(submit_url: str, email: str, secret: str, quiz_url: str, answer: Any) -> Dict[str, Any]:
    """Submit answer to the quiz endpoint"""
    
    if not submit_url or not submit_url.startswith('http'):
        if submit_url and submit_url.startswith('/'):
            parsed = urlparse(quiz_url)
            submit_url = f"{parsed.scheme}://{parsed.netloc}{submit_url}"
        else:
            return {"error": "Invalid submit URL"}
    
    # CRITICAL FIX: Use secret parameter, not SECRET global
    payload = {
        "email": email,
        "secret": secret,  # ✅ FIXED - was using SECRET global
        "url": quiz_url,
        "answer": answer
    }
    
    try:
        logger.info(f"📤 Submitting to: {submit_url}")
        answer_preview = json.dumps(answer) if isinstance(answer, (dict, list)) else str(answer)
        logger.info(f"📤 Answer: {answer_preview[:200]}")
        
        resp = requests.post(
            submit_url, 
            json=payload, 
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        logger.info(f"📥 Status: {resp.status_code}")
        
        try:
            result = resp.json()
            logger.info(f"📥 Response: {json.dumps(result)[:400]}")
        except:
            result = {"raw": resp.text[:500], "status": resp.status_code}
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Submit error: {e}")
        return {"error": str(e)}

def submit_answer_with_retry(submit_url: str, email: str, secret: str, quiz_url: str, answer: Any, max_retries: int = 2) -> Dict[str, Any]:
    """Submit answer with retry logic"""
    
    for attempt in range(max_retries + 1):
        try:
            result = submit_answer(submit_url, email, secret, quiz_url, answer)
            
            if result and "correct" in result:
                return result
            
            status = result.get("status")
            if status in [400, 403]:
                return result
            
            if attempt == max_retries:
                return result
            
            logger.warning(f"⚠️ Attempt {attempt + 1} failed, retrying...")
            time.sleep(1)
            
        except Exception as e:
            if attempt == max_retries:
                return {"error": str(e)}
            time.sleep(1)
    
    return {"error": "Max retries exceeded"}

# ============================================================================
# QUIZ CHAIN PROCESSOR
# ============================================================================

def process_quiz_chain(initial_url: str, email: str, secret: str, start_time: float, timeout: int = 170) -> List[Dict[str, Any]]:
    """Process quiz chain"""
    current_url = initial_url
    results = []
    iteration = 0
    max_iterations = 30
    
    while current_url and iteration < max_iterations:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        
        if remaining < 20:
            logger.warning(f"⚠️ Timeout approaching")
            results.append({"quiz_number": iteration, "url": current_url, "error": "Timeout", "status": "timeout"})
            break
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 QUIZ #{iteration}")
        logger.info(f"🔗 URL: {current_url}")
        logger.info(f"⏱️ Remaining: {remaining:.1f}s")
        logger.info(f"{'='*70}")
        
        try:
            quiz_page = extract_page_content(current_url)
            
            if quiz_page.get("error"):
                results.append({"quiz_number": iteration, "url": current_url, "error": "Extract failed", "status": "extraction_failed"})
                break
            
            combined = quiz_page.get("html", "") + "\n" + quiz_page.get("text", "")
            submit_url = extract_submit_url(combined, current_url)
            
            if not submit_url:
                results.append({"quiz_number": iteration, "url": current_url, "error": "No submit URL", "status": "no_submit_url"})
                break
            
            logger.info(f"✅ Submit: {submit_url}")
            
            # Download files
            downloaded_files = {}
            links = quiz_page.get("links", [])
            
            download_exts = ['.pdf', '.csv', '.json', '.xlsx', '.xls', '.txt', '.opus', '.mp3', '.wav', '.png', '.jpg', '.jpeg']
            download_links = [l for l in links if 'submit' not in l.lower() and any(l.lower().endswith(e) for e in download_exts)]
            
            if download_links:
                logger.info(f"📥 Downloading {len(download_links)} files...")
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {executor.submit(download_file, link): link for link in download_links}
                    
                    for future in as_completed(futures):
                        link = futures[future]
                        try:
                            content = future.result()
                            if content:
                                parsed = smart_parse_file(content, link)
                                downloaded_files[link] = parsed
                        except Exception as e:
                            logger.error(f"❌ File error: {e}")
            
            logger.info(f"✅ Files processed: {len(downloaded_files)}")
            
            # Solve
            solution = solve_with_llm(quiz_page, downloaded_files, current_url, email)
            
            if "error" in solution:
                results.append({"quiz_number": iteration, "url": current_url, "error": "Solve error", "status": "solving_failed"})
                break
            
            answer = solution.get("answer")
            
            if answer is None or answer == "":
                results.append({"quiz_number": iteration, "url": current_url, "error": "No answer", "status": "no_answer"})
                break
            
            # Submit
            submission = submit_answer_with_retry(submit_url, email, secret, current_url, answer)
            
            is_correct = submission.get("correct", False)
            
            result_entry = {
                "quiz_number": iteration,
                "url": current_url,
                "submit_url": submit_url,
                "answer": answer,
                "solution": solution,
                "submission_result": submission,
                "status": "correct" if is_correct else "incorrect",
                "correct": is_correct,
                "time_elapsed": round(time.time() - start_time, 2)
            }
            
            results.append(result_entry)
            
            if is_correct:
                logger.info(f"✅ Quiz #{iteration} CORRECT!")
            else:
                logger.warning(f"❌ Quiz #{iteration} INCORRECT")
                logger.warning(f"   Reason: {submission.get('reason', 'N/A')}")
            
            next_url = submission.get("url")
            
            if next_url:
                logger.info(f"➡️ Next: {next_url}")
                current_url = next_url
                time.sleep(0.3)
            else:
                logger.info("🏁 Complete!")
                break
        
        except Exception as e:
            logger.error(f"❌ Exception: {e}")
            logger.error(traceback.format_exc())
            results.append({"quiz_number": iteration, "url": current_url, "error": str(e), "status": "exception"})
            break
    
    return results

# ============================================================================
# FLASK ENDPOINTS
# ============================================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Enhanced Dynamic Quiz Solver",
        "version": "6.0 - Fixed",
        "status": "running",
        "model": AIMLAPI_MODEL
    }), 200

@app.route("/health", methods=["GET"])
def health_check():
    health = {
        "status": "healthy",
        "llm_ready": client is not None,
        "email_set": bool(EMAIL),
        "secret_set": bool(SECRET)
    }
    
    try:
        driver = setup_browser()
        driver.quit()
        health["browser_ready"] = True
    except Exception as e:
        health["browser_ready"] = False
    
    return jsonify(health), 200

@app.route("/quiz", methods=["POST"])
def quiz_endpoint():
    start_time = time.time()
    
    if not request.is_json:
        return jsonify({"error": "Invalid JSON"}), 400
    
    try:
        data = request.get_json()
    except:
        return jsonify({"error": "Malformed JSON"}), 400
    
    if data.get("secret") != SECRET:
        return jsonify({"error": "Invalid secret"}), 403
    
    if not data.get("email") or not data.get("url"):
        return jsonify({"error": "Missing fields"}), 400
    
    if not client:
        return jsonify({"error": "LLM not available"}), 500
    
    logger.info(f"\n{'='*70}")
    logger.info(f"📨 NEW QUIZ REQUEST")
    logger.info(f"   Email: {data['email']}")
    logger.info(f"   URL: {data['url']}")
    logger.info(f"{'='*70}")
    
    try:
        results = process_quiz_chain(
            initial_url=data["url"],
            email=data["email"],
            secret=data["secret"],  # ✅ Pass user's secret
            start_time=start_time,
            timeout=170
        )
        
        total_time = time.time() - start_time
        num_correct = sum(1 for r in results if r.get("correct"))
        num_incorrect = sum(1 for r in results if r.get("status") == "incorrect")
        num_errors = sum(1 for r in results if r.get("status") not in ["correct", "incorrect"])
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🏁 COMPLETED")
        logger.info(f"   Total: {len(results)}")
        logger.info(f"   ✅ Correct: {num_correct}")
        logger.info(f"   ❌ Incorrect: {num_incorrect}")
        logger.info(f"   ⚠️ Errors: {num_errors}")
        logger.info(f"   ⏱️ Time: {total_time:.2f}s")
        logger.info(f"{'='*70}")
        
        return jsonify({
            "status": "completed",
            "results": results,
            "summary": {
                "total_quizzes": len(results),
                "correct": num_correct,
                "incorrect": num_incorrect,
                "errors": num_errors,
                "time_taken_seconds": round(total_time, 2),
                "success_rate": round(num_correct / len(results) * 100, 1) if results else 0
            }
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "time_taken_seconds": round(time.time() - start_time, 2)
        }), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 Enhanced Dynamic Quiz Solver")
    logger.info(f"   Version: 6.0 - Production Fixed")
    logger.info(f"   Port: {port}")
    logger.info(f"   Model: {AIMLAPI_MODEL}")
    logger.info(f"   LLM: {'✅' if client else '❌'}")
    logger.info(f"   Email: {EMAIL if EMAIL else '❌'}")
    logger.info(f"   Secret: {'✅' if SECRET else '❌'}")
    logger.info(f"{'='*70}\n")
    
    app.run(host="0.0.0.0", port=port, debug=False)