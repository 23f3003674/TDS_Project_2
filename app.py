#!/usr/bin/env python3
"""
Enhanced Dynamic Quiz Solver - Production Ready
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
            "data_sample": data_sample
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
    
    logger.info(f"📄 Question length: {len(page_text)} chars")
    logger.info(f"📦 Data files: {len(downloaded_files)}")
    
    # Build comprehensive context with smart truncation per file
    context_parts = {}
    total_size = 0
    max_total_size = 40000  # Increased limit
    
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
            elif file_data.get("type") == "pdf" and "content" in file_data:
                content = file_data["content"]
                summary["page_count"] = content.get("page_count", 0)
                summary["all_text"] = content.get("all_text", "")[:3000]
                summary["all_tables"] = content.get("all_tables", [])
            elif file_data.get("type") == "audio":
                # Include audio metadata but note transcription needed
                summary["audio_format"] = file_data.get("content", {}).get("format")
                summary["audio_size"] = file_data.get("content", {}).get("size")
                summary["note"] = "Audio file detected - transcription service required"
            elif file_data.get("type") == "image":
                content = file_data.get("content", {})
                context_parts[url] = {
                    "type": "image",
                    "dominant_color": content.get("dominant_color"),
                    "dominant_rgb": content.get("dominant_rgb"),
                    "format": content.get("format"),
                    "note": "Use dominant_color for color-related questions"
                }
            elif file_data.get("type") == "json":
                context_parts[url] = {
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
        
        # Stop if we're approaching limit
        if total_size > max_total_size:
            logger.warning(f"⚠️ Context size limit reached at {total_size} chars")
            break
    
    context_str = json.dumps(context_parts, indent=2, default=str)
    logger.info(f"📊 Context size: {len(context_str)} chars")
    
    # Create the prompt with enhanced instructions
    prompt = f"""You are a precise data analyst and command-line expert solving a quiz. Your job is to read the question carefully, analyze the provided data, and return the EXACT answer requested.

USER EMAIL: {user_email}

QUESTION:
{page_text[:8000]}

AVAILABLE DATA:
{context_str}

CRITICAL INSTRUCTIONS:
1. READ the question VERY carefully - what EXACTLY is it asking for?
2. If the question asks for a COMMAND or CODE, return the EXACT command/code string
3. If the question references YOUR email, use: {user_email}
4. If the question references "<your email>", replace it with: {user_email}
5. For audio files: acknowledge you cannot transcribe but need external service
6. Return answers in the EXACT format requested (string, number, boolean, object)

SPECIAL CASES:
- Command strings: Return EXACTLY as shown in example, replacing placeholders with actual values
- Email placeholders: Always replace "<your email>" with {user_email}  
- Git commands: Return exact command strings as requested
- File paths: Return exact paths as shown
- Audio transcription: Indicate need for transcription service
- Image color: Use the "dominant_color" hex value from image data (e.g., "#b45a1e")
- JSON normalization: Return as actual JSON array, not string

ANSWER FORMAT GUIDE:
- Command/code string → return as string exactly as shown
- Number (sum, count) → return as integer or float
- Text (name, secret, path) → return as string
- Yes/no → return as boolean (true/false)
- Multiple values → return as array or object

EXAMPLES:
- Question: "Submit the command: uv http get..." → Answer: "uv http get https://example.com?email={user_email}"
- Question: "Submit: git add env.sample" → Answer: "git add env.sample\ngit commit -m \"message\""
- Question: "What is the sum of X?" → Answer: 12345 (calculated from data)
- Question: "Transcribe audio file" → Acknowledge cannot transcribe without service

CRITICAL RULES:
✓ If question shows example command with <your email>, replace with {user_email}
✓ Return command strings EXACTLY as formatted in question
✓ For multi-line commands, preserve line breaks
✗ Do NOT add extra quotes or escaping unless shown in example
✗ Do NOT return placeholder values like "N/A", "null", "your_answer"
✓ For calculations, use ACTUAL data provided
✗ Do NOT guess - if you need transcription service, say so

Return ONLY valid JSON (no markdown, no code blocks):
{{
    "answer": <your_exact_answer>,
    "reasoning": "step-by-step explanation of your answer",
    "confidence": "high/medium/low",
    "data_used": "which data/command was used"
}}

IMPORTANT: 
- If question asks for a command, return it as a string
- If email is mentioned, use {user_email}
- If audio file needs transcription, explain limitation clearly"""

    try:
        logger.info("🤖 Querying LLM...")
        
        response = client.chat.completions.create(
            model=AIMLAPI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise data analyst and command-line expert. Return exact commands/strings as shown in questions. Replace <your email> with actual user email. For audio files, acknowledge transcription limitations. Return valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            #temperature=0.1,
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"🤖 LLM raw response: {response_text[:500]}")
        
        # Clean markdown formatting
        response_text = re.sub(r'```json\s*|\s*```', '', response_text)
        response_text = response_text.strip()
        
        # Parse JSON
        try:
            solution = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON parse error: {e}")
            # Try to extract JSON from text
            json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', response_text, re.DOTALL)
            if not json_match:
                # Try multiline
                json_match = re.search(r'\{.*?"answer".*?\}', response_text, re.DOTALL)
            
            if json_match:
                solution = json.loads(json_match.group())
            else:
                return {
                    "error": "Could not parse LLM response as JSON",
                    "raw": response_text[:1500]
                }
        
        if "answer" not in solution:
            return {
                "error": "No 'answer' field in LLM response",
                "raw": response_text[:1500]
            }
        
        answer = solution["answer"]
        
        # Validate answer is not placeholder/invalid
        invalid_answers = [
            "", None, "N/A", "null", "placeholder", "<answer>", 
            "your_answer", "your answer", "calculate this", "TODO",
            "<your email>"
        ]
        
        answer_str = str(answer).lower().strip() if answer is not None else ""
        
        # Check if answer contains unresolved placeholders
        if answer and isinstance(answer, str) and "<your email>" in answer:
            logger.warning("⚠️ Answer still contains <your email> placeholder, fixing...")
            answer = answer.replace("<your email>", user_email)
            solution["answer"] = answer
        
        if answer in invalid_answers or answer_str in invalid_answers:
            logger.warning("⚠️ LLM returned invalid/placeholder answer, retrying with emphasis...")
            
            # Retry with more explicit instructions
            retry_prompt = f"""PREVIOUS ATTEMPT FAILED - You returned a placeholder/invalid answer.

USER EMAIL TO USE: {user_email}

The question is asking for a SPECIFIC value, command, or acknowledgment.

QUESTION (Read VERY carefully):
{page_text[:6000]}

DATA (Analyze this):
{context_str[:20000]}

STEP-BY-STEP PROCESS:
1. What EXACTLY is the question asking for?
2. Is it asking for:
   - A command string? → Return the exact command with {user_email} replacing any email placeholder
   - A calculation? → Calculate from provided data
   - File path? → Return exact path shown
   - Audio transcription? → Acknowledge limitation and explain need for transcription service
   - Text extraction? → Extract from data

3. For COMMANDS: Copy the exact format from the question, replacing:
   - "<your email>" with {user_email}
   - Any other placeholders with actual values

4. For AUDIO files: Return a clear statement like "Cannot transcribe audio file without transcription service. Audio file detected at [URL]."

Return valid JSON with the ACTUAL answer:
{{
    "answer": <actual_value_or_command_or_acknowledgment>,
    "reasoning": "detailed step-by-step explanation"
}}

DO NOT RETURN: null, N/A, placeholder, empty string, or unresolved placeholders like "<your email>"."""

            try:
                retry_response = client.chat.completions.create(
                    model=AIMLAPI_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You must return the actual answer. User email is {user_email}. Replace ALL email placeholders. For audio, acknowledge transcription limitation. No placeholders allowed."
                        },
                        {"role": "user", "content": retry_prompt}
                    ],
                    #temperature=0.2,
                )
                
                retry_text = retry_response.choices[0].message.content.strip()
                retry_text = re.sub(r'```json\s*|\s*```', '', retry_text)
                
                try:
                    solution = json.loads(retry_text)
                    answer = solution.get("answer")
                    
                    # Fix email placeholder again if present
                    if answer and isinstance(answer, str) and "<your email>" in answer:
                        answer = answer.replace("<your email>", user_email)
                        solution["answer"] = answer
                    
                    logger.info(f"✅ Retry successful, new answer: {answer}")
                except:
                    logger.error("❌ Retry JSON parsing also failed")
                    # Try to at least extract text
                    if "cannot" in retry_text.lower() and "audio" in retry_text.lower():
                        return {
                            "answer": "Audio transcription requires external service - file detected but cannot be processed",
                            "reasoning": "Audio file needs transcription service",
                            "confidence": "medium"
                        }
                    return {
                        "error": "Could not get valid answer after retry",
                        "raw": retry_text[:1500]
                    }
            except Exception as e:
                logger.error(f"❌ Retry request failed: {e}")
                # For audio files, provide fallback
                if any('audio' in str(file_data.get('type', '')) for file_data in downloaded_files.values()):
                    return {
                        "answer": "Audio transcription service required - cannot process audio files directly",
                        "reasoning": "Audio file detected but transcription service unavailable",
                        "confidence": "low"
                    }
        
        # Log the final answer
        answer_type = type(answer).__name__
        logger.info(f"✅ Final answer: {answer} (type: {answer_type})")
        logger.info(f"💡 Reasoning: {solution.get('reasoning', 'N/A')[:300]}")
        logger.info(f"🎯 Confidence: {solution.get('confidence', 'unknown')}")
        
        return solution
        
    except Exception as e:
        logger.error(f"❌ LLM error: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

# ============================================================================
# ANSWER SUBMISSION
# ============================================================================

def submit_answer(submit_url: str, email: str, secret: str, quiz_url: str, answer: Any) -> Dict[str, Any]:
    """Submit answer to the quiz endpoint"""
    
    # Validate submit_url
    if not submit_url or not submit_url.startswith('http'):
        logger.error(f"❌ Invalid submit URL: {submit_url}")
        
        # Try to fix relative URLs
        if submit_url and submit_url.startswith('/'):
            parsed = urlparse(quiz_url)
            submit_url = f"{parsed.scheme}://{parsed.netloc}{submit_url}"
            logger.info(f"🔧 Fixed relative URL to: {submit_url}")
        else:
            return {"error": f"Invalid submit URL: {submit_url}"}
    
    payload = {
        "email": email,
        "secret": secret,
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
            logger.info(f"📥 Response: {json.dumps(result, indent=2)[:500]}")
        except:
            result = {
                "raw": resp.text[:1000],
                "status": resp.status_code
            }
            logger.warning(f"⚠️ Non-JSON response: {resp.text[:200]}")
        
        return result
        
    except requests.exceptions.Timeout:
        logger.error("❌ Submission timeout")
        return {"error": "Submission timeout"}
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Submission request error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"❌ Submission error: {e}")
        return {"error": str(e)}

def submit_answer_with_retry(submit_url: str, email: str, secret: str, quiz_url: str, answer: Any, max_retries: int = 2) -> Dict[str, Any]:
    """Submit answer with retry logic for transient failures"""
    
    for attempt in range(max_retries + 1):
        try:
            result = submit_answer(submit_url, email, secret, quiz_url, answer)
            
            # If we got a response (even if wrong answer), return it
            if result and "correct" in result:
                return result
            
            # Check for auth errors (don't retry these)
            status = result.get("status")
            if status in [400, 403]:
                logger.error(f"❌ Auth error {status}, not retrying")
                return result
            
            # If this is the last attempt, return whatever we have
            if attempt == max_retries:
                return result
            
            # Otherwise retry
            logger.warning(f"⚠️ Attempt {attempt + 1} failed, retrying in 1s...")
            time.sleep(1)
            
        except Exception as e:
            if attempt == max_retries:
                return {"error": str(e)}
            logger.warning(f"⚠️ Attempt {attempt + 1} exception: {e}, retrying...")
            time.sleep(1)
    
    return {"error": "Max retries exceeded"}

# ============================================================================
# QUIZ CHAIN PROCESSOR
# ============================================================================

def process_quiz_chain(initial_url: str, email: str, secret: str, start_time: float, timeout: int = 170) -> List[Dict[str, Any]]:
    """
    Process quiz chain - follows quiz URLs dynamically
    """
    current_url = initial_url
    results = []
    iteration = 0
    max_iterations = 20  # Safety limit
    
    while current_url and iteration < max_iterations:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        
        # Check if we're running out of time
        if remaining < 25:
            logger.warning(f"⚠️ Only {remaining:.1f}s remaining, stopping chain")
            results.append({
                "quiz_number": iteration,
                "url": current_url,
                "error": "Timeout - insufficient time remaining",
                "status": "timeout"
            })
            break
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 QUIZ #{iteration}")
        logger.info(f"🔗 URL: {current_url}")
        logger.info(f"⏱️ Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
        logger.info(f"{'='*70}")
        
        try:
            # Step 1: Extract quiz page content
            quiz_page = extract_page_content(current_url)
            
            if quiz_page.get("error"):
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": f"Page extraction failed: {quiz_page['error']}",
                    "status": "extraction_failed"
                })
                break
            
            # Step 2: Find submit URL
            combined_content = quiz_page.get("html", "") + "\n" + quiz_page.get("text", "")
            submit_url = extract_submit_url(combined_content, current_url)
            
            if not submit_url:
                logger.error("❌ Could not find submit URL")
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "Could not find submit URL in page content",
                    "status": "no_submit_url",
                    "page_preview": quiz_page.get("text", "")[:500]
                })
                break
            
            logger.info(f"✅ Submit URL: {submit_url}")
            
            # Step 3: Download and parse all linked files (with parallel processing)
            downloaded_files = {}
            links = quiz_page.get("links", [])
            
            logger.info(f"🔍 Found {len(links)} links to process")
            
            # Separate download and scrape links
            download_links = []
            scrape_links = []
            
            download_extensions = [
                '.pdf', '.csv', '.json', '.xlsx', '.xls', 
                '.txt', '.xml', '.tsv', '.dat',
                '.opus', '.mp3', '.wav', '.ogg', '.m4a', '.flac',  # Audio
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'  # Images
            ]
            
            for link in links:
                if 'submit' in link.lower():
                    continue
                
                parsed_link = urlparse(link)
                path = parsed_link.path.lower()
                
                if any(path.endswith(ext) for ext in download_extensions):
                    download_links.append(link)
                elif len(scrape_links) < 3:  # Limit scraping to save time
                    scrape_links.append(link)
            
            # Parallel file downloads
            if download_links:
                logger.info(f"📥 Downloading {len(download_links)} files in parallel...")
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_url = {executor.submit(download_file, link): link for link in download_links}
                    
                    for future in as_completed(future_to_url):
                        link = future_to_url[future]
                        try:
                            file_content = future.result()
                            if file_content:
                                parsed = smart_parse_file(file_content, link)
                                downloaded_files[link] = parsed
                        except Exception as e:
                            logger.error(f"❌ Error processing {link}: {e}")
            
            # Sequential page scraping (less critical)
            for link in scrape_links:
                logger.info(f"🌐 Scraping page: {link}")
                try:
                    scraped = extract_page_content(link, max_wait=4)
                    if not scraped.get("error") and scraped.get("text"):
                        downloaded_files[link] = {
                            "type": "scraped_page",
                            "content": scraped.get("text", ""),
                            "url": link
                        }
                except Exception as e:
                    logger.error(f"❌ Error scraping {link}: {e}")
            
            logger.info(f"✅ Processed {len(downloaded_files)} data sources")
            
            # Step 4: Solve with LLM
            solution = solve_with_llm(quiz_page, downloaded_files, current_url, email)
            
            if "error" in solution:
                logger.error(f"❌ LLM solving failed: {solution.get('error')}")
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": f"LLM error: {solution.get('error')}",
                    "status": "solving_failed",
                    "solution_raw": solution
                })
                break
            
            answer = solution.get("answer")
            
            if answer is None or answer == "":
                logger.error("❌ No valid answer from LLM")
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "LLM did not provide a valid answer",
                    "status": "no_answer",
                    "solution": solution
                })
                break
            
            # Step 5: Submit answer
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
                reason = submission.get("reason", "No reason provided")
                logger.warning(f"   Reason: {reason}")
            
            # Step 6: Check for next quiz
            next_url = submission.get("url")
            
            if next_url:
                logger.info(f"➡️ Next quiz: {next_url}")
                current_url = next_url
            else:
                logger.info("🏁 Quiz chain complete!")
                break
        
        except Exception as e:
            logger.error(f"❌ Exception in quiz #{iteration}: {e}")
            logger.error(traceback.format_exc())
            results.append({
                "quiz_number": iteration,
                "url": current_url,
                "error": str(e),
                "status": "exception",
                "traceback": traceback.format_exc()
            })
            break
    
    return results

# ============================================================================
# FLASK ENDPOINTS
# ============================================================================

@app.route("/", methods=["GET"])
def home():
    """Home endpoint - service info"""
    return jsonify({
        "service": "Enhanced Dynamic Quiz Solver",
        "version": "5.2",
        "status": "running",
        "model": AIMLAPI_MODEL,
        "capabilities": [
            "JavaScript-rendered web scraping",
            "Command string extraction and formatting",
            "Email placeholder replacement",
            "Audio/image file detection",
            "Comprehensive statistical analysis",
            "Multi-format file parsing (PDF, CSV, Excel, JSON, Text, Audio, Images)",
            "Intelligent answer type detection",
            "Automatic retry with improved prompts",
            "Smart caching for performance",
            "Parallel file downloads"
        ],
        "max_time_per_quiz_chain": "170 seconds",
        "features": {
            "file_caching": True,
            "retry_on_failure": True,
            "multi_format_parsing": True,
            "dynamic_submit_url": True,
            "comprehensive_stats": True,
            "parallel_downloads": True,
            "smart_id_detection": True,
            "command_string_formatting": True,
            "email_placeholder_replacement": True,
            "audio_detection": True
        }
    }), 200

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "llm_initialized": client is not None,
        "email_configured": bool(EMAIL),
        "secret_configured": bool(SECRET),
        "model": AIMLAPI_MODEL,
        "chrome_binary": CHROME_BINARY,
        "chromedriver_path": CHROMEDRIVER_PATH
    }
    
    # Check if Chrome is available
    try:
        driver = setup_browser()
        driver.quit()
        health_status["chrome_available"] = True
    except Exception as e:
        health_status["chrome_available"] = False
        health_status["chrome_error"] = str(e)
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code

@app.route("/quiz", methods=["POST"])
def quiz_endpoint():
    """
    Main webhook endpoint for quiz solving
    """
    start_time = time.time()
    
    # Validate request
    if not request.is_json:
        logger.error("❌ Invalid request: Not JSON")
        return jsonify({"error": "Invalid JSON"}), 400
    
    try:
        data = request.get_json()
    except Exception as e:
        logger.error(f"❌ JSON parsing error: {e}")
        return jsonify({"error": "Malformed JSON"}), 400
    
    # Validate secret
    if data.get("secret") != SECRET:
        logger.error("❌ Invalid secret")
        return jsonify({"error": "Invalid secret"}), 403
    
    # Validate required fields
    if not data.get("email"):
        return jsonify({"error": "Missing 'email' field"}), 400
    
    if not data.get("url"):
        return jsonify({"error": "Missing 'url' field"}), 400
    
    # Check if LLM is initialized
    if not client:
        logger.error("❌ LLM not initialized")
        return jsonify({"error": "LLM service not available"}), 500
    
    # Log the request
    logger.info(f"\n{'='*70}")
    logger.info(f"📨 NEW QUIZ REQUEST")
    logger.info(f"   Email: {data['email']}")
    logger.info(f"   URL: {data['url']}")
    logger.info(f"{'='*70}")
    
    try:
        # Process the quiz chain
        results = process_quiz_chain(
            initial_url=data["url"],
            email=data["email"],
            secret=SECRET,
            start_time=start_time,
            timeout=170
        )
        
        # Calculate summary statistics
        total_time = time.time() - start_time
        num_correct = sum(1 for r in results if r.get("correct"))
        num_incorrect = sum(1 for r in results if r.get("status") == "incorrect")
        num_errors = sum(1 for r in results if r.get("status") not in ["correct", "incorrect"])
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🏁 QUIZ CHAIN COMPLETED")
        logger.info(f"   ✅ Correct: {num_correct}")
        logger.info(f"   ❌ Incorrect: {num_incorrect}")
        logger.info(f"   ⚠️ Errors: {num_errors}")
        logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
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
            "traceback": traceback.format_exc()[:2000],
            "time_taken_seconds": round(time.time() - start_time, 2)
        }), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 Enhanced Dynamic Quiz Solver")
    logger.info(f"   Version: 5.2 (Fixed - Command Support)")
    logger.info(f"   Port: {port}")
    logger.info(f"   Model: {AIMLAPI_MODEL}")
    logger.info(f"   LLM: {'✅ Ready' if client else '❌ Not initialized'}")
    logger.info(f"   Email: {EMAIL if EMAIL else '❌ Not set'}")
    logger.info(f"   Secret: {'✅ Set' if SECRET else '❌ Not set'}")
    logger.info(f"{'='*70}\n")
    
    app.run(host="0.0.0.0", port=port, debug=False)