#!/usr/bin/env python3
"""
Universal Dynamic Quiz Solver - Production Ready
Handles ANY quiz type: commands, data analysis, transcription, vision, etc.
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
from PIL import Image

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
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
AIMLAPI_MODEL = os.getenv("AIMLAPI_MODEL", "gpt-5-nano")

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
    opts.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
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
    """Extract content from any webpage with intelligent waiting"""
    logger.info(f"🌐 Extracting: {url}")
    driver = setup_browser()
    
    try:
        driver.get(url)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        previous_text = ""
        stable_count = 0
        critical_keywords = ["result", "quiz", "question", "data", "download", "submit", "answer", "command"]
        
        for i in range(max_wait):
            time.sleep(1)
            try:
                current_text = driver.execute_script("return document.body.innerText;")
            except:
                current_text = driver.find_element(By.TAG_NAME, "body").text
            
            has_critical = any(kw in current_text.lower() for kw in critical_keywords)
            
            if current_text == previous_text and len(current_text) > 50 and has_critical:
                stable_count += 1
                if stable_count >= 2:
                    logger.info(f"✅ Content stabilized after {i+1}s")
                    break
            else:
                stable_count = 0
            previous_text = current_text
        
        time.sleep(2)
        
        try:
            page_text = driver.execute_script("return document.body.innerText;")
        except:
            page_text = driver.find_element(By.TAG_NAME, "body").text
        
        page_source = driver.page_source
        
        # Extract all links
        links = []
        try:
            for link_elem in driver.find_elements(By.TAG_NAME, "a"):
                try:
                    href = link_elem.get_attribute("href")
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
        return {
            "html": "",
            "text": f"Error: {str(e)}",
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
    """Download any file with caching"""
    if use_cache and url in file_cache:
        logger.info(f"📦 Cache hit: {url}")
        return file_cache[url]
    
    try:
        logger.info(f"📥 Downloading: {url}")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        
        content = resp.content
        logger.info(f"✅ Downloaded {len(content)} bytes")
        
        if use_cache:
            file_cache[url] = content
        
        return content
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        return None

# ============================================================================
# SPECIALIZED FILE PROCESSORS
# ============================================================================

def analyze_image(image_bytes: bytes, url: str) -> Dict[str, Any]:
    """Analyze image and extract dominant color + metadata"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get all pixels
        pixels = list(img.getdata())
        
        # Count color frequencies
        color_counts = {}
        for pixel in pixels:
            rgb_hex = '#{:02x}{:02x}{:02x}'.format(pixel[0], pixel[1], pixel[2])
            color_counts[rgb_hex] = color_counts.get(rgb_hex, 0) + 1
        
        # Find dominant color (most frequent)
        dominant = max(color_counts.items(), key=lambda x: x[1])
        
        # Get top 10 colors
        top_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        logger.info(f"🎨 Image: {img.width}x{img.height}, Dominant: {dominant[0]} ({dominant[1]} pixels)")
        
        return {
            "dominant_color": dominant[0],
            "dominant_count": dominant[1],
            "unique_colors": len(color_counts),
            "total_pixels": len(pixels),
            "dimensions": {"width": img.width, "height": img.height},
            "top_colors": [{"color": c[0], "count": c[1]} for c in top_colors],
            "format": img.format,
            "mode": img.mode
        }
    except Exception as e:
        logger.error(f"❌ Image analysis error: {e}")
        return {"error": str(e)}

def transcribe_audio(audio_bytes: bytes, url: str) -> Dict[str, Any]:
    """Transcribe audio using OpenAI Whisper API"""
    try:
        # Determine format
        audio_format = url.split('.')[-1].lower()
        if audio_format not in ['opus', 'mp3', 'wav', 'ogg', 'm4a', 'flac']:
            audio_format = 'opus'
        
        # Create file-like object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = f"audio.{audio_format}"
        
        logger.info(f"🎤 Transcribing {audio_format} audio ({len(audio_bytes)} bytes)...")
        
        # Call Whisper API
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        
        transcribed_text = transcription if isinstance(transcription, str) else transcription.text
        logger.info(f"✅ Transcription: '{transcribed_text}'")
        
        return {
            "transcription": transcribed_text,
            "format": audio_format,
            "size": len(audio_bytes)
        }
    except Exception as e:
        logger.error(f"❌ Audio transcription error: {e}")
        return {
            "error": str(e),
            "format": audio_format,
            "size": len(audio_bytes),
            "note": "Whisper API may not be available or configured"
        }

def parse_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    """Parse PDF - extract text and tables from all pages"""
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
                                # Smart header detection
                                first_row = table[0]
                                has_header = (
                                    len(table) > 1 and 
                                    all(isinstance(cell, str) and cell and 
                                        not str(cell).replace('.','').replace('-','').isdigit()
                                        for cell in first_row if cell)
                                )
                                
                                if has_header:
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
                                result["all_tables"].append({"page": page_num, **table_info})
                            except Exception as e:
                                logger.warning(f"⚠️ Table parse error page {page_num}: {e}")
                
                result["pages"].append(page_data)
        
        logger.info(f"✅ PDF: {result['page_count']} pages, {len(result['all_tables'])} tables")
        return result
        
    except Exception as e:
        logger.error(f"❌ PDF error: {e}")
        return {"error": str(e)}

def parse_csv(csv_content: bytes) -> Dict[str, Any]:
    """Parse CSV with comprehensive analysis"""
    try:
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode("utf-8", errors='ignore')
        
        # Try multiple separators
        df = None
        for sep in [',', ';', '\t', '|']:
            try:
                df = pd.read_csv(io.StringIO(csv_content), sep=sep)
                if len(df.columns) > 1 or (len(df.columns) == 1 and len(df) > 0):
                    logger.info(f"✅ CSV parsed with sep: '{sep}'")
                    break
            except:
                continue
        
        if df is None or df.empty:
            return {"error": "Could not parse CSV"}
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Smart numeric conversion
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    cleaned = df[col].astype(str).str.replace(r'[$,€£¥\s%]', '', regex=True)
                    numeric = pd.to_numeric(cleaned, errors='coerce')
                    has_leading_zeros = df[col].astype(str).str.match(r'^0\d').any()
                    
                    if numeric.notna().sum() > len(df) * 0.8 and not has_leading_zeros:
                        df[col] = numeric
                        logger.info(f"  🔢 '{col}' → numeric")
                except:
                    pass
        
        # Column analysis
        column_analysis = {}
        for col in df.columns:
            col_data = df[col].dropna()
            
            analysis = {
                "dtype": str(df[col].dtype),
                "non_null": int(df[col].notna().sum()),
                "null": int(df[col].isna().sum()),
                "total": len(df)
            }
            
            if pd.api.types.is_numeric_dtype(df[col]) and len(col_data) > 0:
                analysis.update({
                    "sum": float(col_data.sum()),
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "std": float(col_data.std()) if len(col_data) > 1 else 0
                })
                logger.info(f"  📊 '{col}': sum={analysis['sum']:.2f}, mean={analysis['mean']:.2f}")
            else:
                value_counts = col_data.value_counts()
                analysis.update({
                    "unique": len(col_data.unique()),
                    "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "sample_values": [str(v) for v in col_data.unique()[:10]]
                })
            
            column_analysis[col] = analysis
        
        result = {
            "shape": list(df.shape),
            "rows": len(df),
            "columns": list(df.columns),
            "column_analysis": column_analysis,
            "data": df.to_dict('records'),
            "first_rows": df.head(20).to_dict('records'),
            "last_rows": df.tail(10).to_dict('records')
        }
        
        logger.info(f"✅ CSV: {df.shape[0]} rows × {df.shape[1]} cols")
        return result
        
    except Exception as e:
        logger.error(f"❌ CSV error: {e}")
        return {"error": str(e)}

def parse_excel(excel_bytes: bytes) -> Dict[str, Any]:
    """Parse Excel with multiple sheets"""
    try:
        dfs = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)
        result = {"sheets": {}, "sheet_names": list(dfs.keys())}
        
        for sheet_name, df in dfs.items():
            csv_str = df.to_csv(index=False)
            result["sheets"][sheet_name] = parse_csv(csv_str.encode())
        
        logger.info(f"✅ Excel: {len(result['sheets'])} sheets")
        return result
    except Exception as e:
        logger.error(f"❌ Excel error: {e}")
        return {"error": str(e)}

# ============================================================================
# SMART FILE PARSER
# ============================================================================

def smart_parse_file(file_content: bytes, url: str) -> Dict[str, Any]:
    """
    Universal file parser - detects type and processes accordingly
    IMPORTANT: Check file extension FIRST before content-based detection
    """
    result = {"type": "unknown", "content": None, "error": None, "url": url}
    
    try:
        # PRIORITY 1: Check by file extension (most reliable)
        url_lower = url.lower()
        
        # Images - check FIRST
        if url_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg')):
            result["type"] = "image"
            result["content"] = analyze_image(file_content, url)
            return result
        
        # Audio files
        elif url_lower.endswith(('.opus', '.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac')):
            result["type"] = "audio"
            result["content"] = transcribe_audio(file_content, url)
            return result
        
        # PDF
        elif url_lower.endswith('.pdf') or file_content[:4] == b"%PDF":
            result["type"] = "pdf"
            result["content"] = parse_pdf(file_content)
            return result
        
        # Excel
        elif url_lower.endswith(('.xlsx', '.xls')):
            result["type"] = "excel"
            result["content"] = parse_excel(file_content)
            return result
        
        # CSV
        elif url_lower.endswith('.csv'):
            result["type"] = "csv"
            result["content"] = parse_csv(file_content)
            return result
        
        # JSON
        elif url_lower.endswith('.json'):
            result["type"] = "json"
            result["content"] = json.loads(file_content.decode('utf-8'))
            return result
        
        # Text files
        elif url_lower.endswith(('.txt', '.text', '.md', '.log')):
            result["type"] = "text"
            result["content"] = file_content.decode('utf-8', errors='ignore')
            return result
        
        # PRIORITY 2: Content-based detection (fallback)
        # Only if extension didn't match
        
        # Check for JSON content
        if file_content[:1] in [b'{', b'[']:
            try:
                result["type"] = "json"
                result["content"] = json.loads(file_content.decode('utf-8'))
                return result
            except:
                pass
        
        # Check for CSV content (last resort)
        if b',' in file_content[:1000]:
            try:
                result["type"] = "csv"
                result["content"] = parse_csv(file_content)
                return result
            except:
                pass
        
        # Default: treat as text
        result["type"] = "text"
        try:
            result["content"] = file_content.decode('utf-8', errors='ignore')
        except:
            result["content"] = f"Binary content: {len(file_content)} bytes"
        
        logger.info(f"✅ Parsed as: {result['type']}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Parse error: {e}")
        result["error"] = str(e)
        return result

# ============================================================================
# SUBMIT URL EXTRACTION
# ============================================================================

def extract_submit_url(content: str, base_url: str) -> Optional[str]:
    """Extract submit URL from page content"""
    if not content:
        return None
    
    parsed = urlparse(base_url)
    base_domain = f"{parsed.scheme}://{parsed.netloc}"
    
    # Strategy 1: JSON payload examples
    patterns = [
        r'"url":\s*"([^"]*submit[^"]*)"',
        r"'url':\s*'([^']*submit[^']*)'",
        r'(?:POST|Post|submit)\s+(?:to|at)?\s*([^\s<>"\']+/submit[^\s<>"\']*)',
        r'(https?://[^\s<>"\']+/submit[^\s<>"\']*)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            url = matches[0].strip()
            if url.startswith('/'):
                return f"{base_domain}{url}"
            elif url.startswith('http'):
                return url
    
    # Fallback
    if 'submit' in content.lower():
        return f"{base_domain}/submit"
    
    return None

# ============================================================================
# LLM SOLVING ENGINE
# ============================================================================

def solve_with_llm(quiz_page: Dict[str, Any], downloaded_files: Dict[str, Any], 
                   quiz_url: str, user_email: str) -> Dict[str, Any]:
    """
    Universal LLM solver - handles any question type dynamically
    """
    if not client:
        return {"error": "LLM not initialized"}
    
    page_text = quiz_page.get("text", "")
    logger.info(f"📄 Question: {len(page_text)} chars")
    logger.info(f"📦 Files: {len(downloaded_files)}")
    
    # Build context with smart truncation
    context_parts = {}
    for url, file_data in downloaded_files.items():
        file_type = file_data.get("type")
        
        # Include full content for small files, summarize large ones
        file_str = json.dumps(file_data, default=str)
        if len(file_str) < 5000:
            context_parts[url] = file_data
        else:
            # Summarize large files
            summary = {"type": file_type, "url": url}
            content = file_data.get("content", {})
            
            if file_type == "csv":
                summary["columns"] = content.get("columns", [])
                summary["column_analysis"] = content.get("column_analysis", {})
                summary["data"] = content.get("data", [])[:50]  # First 50 rows
            elif file_type == "image":
                summary["content"] = content  # Keep full image analysis
            elif file_type == "audio":
                summary["content"] = content  # Keep full transcription
            elif file_type == "pdf":
                summary["all_text"] = content.get("all_text", "")[:5000]
                summary["all_tables"] = content.get("all_tables", [])
            else:
                summary["content"] = str(content)[:2000]
            
            context_parts[url] = summary
    
    context_str = json.dumps(context_parts, indent=2, default=str)[:40000]
    
    # Build comprehensive prompt
    prompt = f"""You are a universal problem solver. Analyze the question and data, then provide the EXACT answer.

USER EMAIL: {user_email}

QUESTION:
{page_text[:8000]}

AVAILABLE DATA:
{context_str}

ANSWER GUIDELINES:

1. COMMAND STRINGS:
   - If you see "<your email>", keep it EXACTLY as "<your email>" (preserve placeholder)
   - If you see the actual email address, use {user_email}
   - Return command strings as-is, preserving all formatting

2. DATA ANALYSIS:
   - Calculate from provided data (sums, counts, averages, etc.)
   - Use column_analysis for quick stats
   - Return exact numeric values

3. IMAGE QUESTIONS:
   - Use dominant_color field (format: "#rrggbb")
   - Return hex color in lowercase

4. AUDIO TRANSCRIPTION:
   - Use the transcription field
   - Return the exact transcribed text

5. JSON ARRAYS:
   - If answer is an array, return actual array: [{{}}, {{}}]
   - NOT a string: "[{{}}, {{}}]"
   - Normalize data as requested (snake_case, ISO dates, etc.)

6. FILE PATHS / URLS:
   - Return exact paths as shown in question
   - Preserve leading slashes

7. TEXT EXTRACTION:
   - Extract exact values from provided data
   - Match requested format exactly

ANSWER TYPES:
- String: "text" or "command string"
- Number: 123 or 45.67
- Boolean: true or false
- Array: [{{"key": "value"}}]
- Object: {{"key": "value"}}
- Hex color: "#rrggbb"

CRITICAL RULES:
✓ Read question carefully - what format does it expect?
✓ Preserve placeholders like "<your email>" in commands
✓ Return JSON arrays as arrays, not strings
✓ Use actual data from files, never guess
✓ Match the exact format shown in examples

Return ONLY valid JSON:
{{
    "answer": <your_answer_here>,
    "reasoning": "brief explanation",
    "confidence": "high/medium/low"
}}"""

    try:
        logger.info("🤖 Querying LLM...")
        
        response = client.chat.completions.create(
            model=AIMLAPI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise analyst. Preserve placeholders. Return JSON arrays as arrays. Use provided data."
                },
                {"role": "user", "content": prompt}
            ],
            #temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"🤖 Raw response: {response_text[:300]}")
        
        # Clean markdown
        response_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        
        # Parse JSON
        try:
            solution = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON
            match = re.search(r'\{.*?"answer".*?\}', response_text, re.DOTALL)
            if match:
                solution = json.loads(match.group())
            else:
                return {"error": "Invalid JSON response", "raw": response_text[:500]}
        
        if "answer" not in solution:
            return {"error": "No answer field", "raw": response_text[:500]}
        
        answer = solution["answer"]
        
        # POST-PROCESSING: Fix common issues
        
        # 1. Convert JSON string arrays to actual arrays
        if isinstance(answer, str) and answer.strip().startswith('[') and answer.strip().endswith(']'):
            try:
                parsed = json.loads(answer)
                if isinstance(parsed, list):
                    logger.info("✅ Converted string array to actual array")
                    answer = parsed
                    solution["answer"] = answer
            except:
                pass
        
        # 2. Validate not a placeholder
        if answer in ["", None, "N/A", "null", "placeholder", "TODO"]:
            logger.warning("⚠️ Invalid placeholder answer")
            return {"error": "Placeholder answer detected", "solution": solution}
        
        # Log final answer
        logger.info(f"✅ Answer: {json.dumps(answer)[:200]} (type: {type(answer).__name__})")
        logger.info(f"💡 Reasoning: {solution.get('reasoning', '')[:200]}")
        logger.info(f"🎯 Confidence: {solution.get('confidence', 'unknown')}")
        
        return solution
        
    except Exception as e:
        logger.error(f"❌ LLM error: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

# ============================================================================
# ANSWER SUBMISSION
# ============================================================================

def submit_answer(submit_url: str, email: str, secret: str, quiz_url: str, answer: Any) -> Dict[str, Any]:
    """Submit answer to quiz endpoint"""
    
    if not submit_url or not submit_url.startswith('http'):
        if submit_url and submit_url.startswith('/'):
            parsed = urlparse(quiz_url)
            submit_url = f"{parsed.scheme}://{parsed.netloc}{submit_url}"
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
        logger.info(f"📤 Answer: {json.dumps(answer, default=str)[:200]}")
        
        resp = requests.post(
            submit_url,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        logger.info(f"📥 Status: {resp.status_code}")
        
        try:
            result = resp.json()
            logger.info(f"📥 Response: {json.dumps(result)[:300]}")
        except:
            result = {"raw": resp.text[:500], "status": resp.status_code}
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Submit error: {e}")
        return {"error": str(e)}

# ============================================================================
# QUIZ CHAIN PROCESSOR
# ============================================================================

def process_quiz_chain(initial_url: str, email: str, secret: str, 
                       start_time: float, timeout: int = 170) -> List[Dict[str, Any]]:
    """
    Process entire quiz chain - follows URLs dynamically
    """
    current_url = initial_url
    results = []
    iteration = 0
    max_iterations = 20
    
    while current_url and iteration < max_iterations:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        
        if remaining < 25:
            logger.warning(f"⚠️ Only {remaining:.1f}s left, stopping")
            results.append({
                "quiz_number": iteration,
                "url": current_url,
                "error": "Timeout - insufficient time",
                "status": "timeout"
            })
            break
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 QUIZ #{iteration}")
        logger.info(f"🔗 {current_url}")
        logger.info(f"⏱️  Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
        logger.info(f"{'='*70}")
        
        try:
            # Extract page content
            quiz_page = extract_page_content(current_url)
            
            if quiz_page.get("error"):
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": f"Extraction failed: {quiz_page['error']}",
                    "status": "extraction_failed"
                })
                break
            
            # Find submit URL
            content = quiz_page.get("html", "") + "\n" + quiz_page.get("text", "")
            submit_url = extract_submit_url(content, current_url)
            
            if not submit_url:
                logger.error("❌ No submit URL found")
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "Could not find submit URL",
                    "status": "no_submit_url"
                })
                break
            
            logger.info(f"✅ Submit URL: {submit_url}")
            
            # Download all linked files in parallel
            downloaded_files = {}
            links = quiz_page.get("links", [])
            
            download_links = []
            for link in links:
                if 'submit' not in link.lower():
                    download_links.append(link)
            
            if download_links:
                logger.info(f"📥 Downloading {len(download_links)} files...")
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_url = {executor.submit(download_file, link): link 
                                    for link in download_links}
                    
                    for future in as_completed(future_to_url):
                        link = future_to_url[future]
                        try:
                            content = future.result()
                            if content:
                                parsed = smart_parse_file(content, link)
                                downloaded_files[link] = parsed
                        except Exception as e:
                            logger.error(f"❌ Error with {link}: {e}")
            
            logger.info(f"✅ Processed {len(downloaded_files)} files")
            
            # Solve with LLM
            solution = solve_with_llm(quiz_page, downloaded_files, current_url, email)
            
            if "error" in solution:
                logger.error(f"❌ Solving failed: {solution['error']}")
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": solution["error"],
                    "status": "solving_failed"
                })
                break
            
            answer = solution.get("answer")
            if answer is None or answer == "":
                logger.error("❌ No valid answer")
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "No valid answer from LLM",
                    "status": "no_answer"
                })
                break
            
            # Submit answer
            submission = submit_answer(submit_url, email, secret, current_url, answer)
            is_correct = submission.get("correct", False)
            
            result_entry = {
                "quiz_number": iteration,
                "url": current_url,
                "answer": answer,
                "correct": is_correct,
                "submission": submission,
                "time_elapsed": round(time.time() - start_time, 2),
                "status": "correct" if is_correct else "incorrect"
            }
            
            results.append(result_entry)
            
            if is_correct:
                logger.info(f"✅ Quiz #{iteration} CORRECT!")
            else:
                logger.warning(f"❌ Quiz #{iteration} INCORRECT")
                logger.warning(f"   Reason: {submission.get('reason', 'Unknown')}")
            
            # Check for next quiz
            next_url = submission.get("url")
            if next_url:
                logger.info(f"➡️  Next: {next_url}")
                current_url = next_url
            else:
                logger.info("🏁 Quiz chain complete!")
                break
                
        except Exception as e:
            logger.error(f"❌ Exception: {e}")
            logger.error(traceback.format_exc())
            results.append({
                "quiz_number": iteration,
                "url": current_url,
                "error": str(e),
                "status": "exception"
            })
            break
    
    return results

# ============================================================================
# FLASK ENDPOINTS
# ============================================================================

@app.route("/", methods=["GET"])
def home():
    """Service info"""
    return jsonify({
        "service": "Universal Dynamic Quiz Solver",
        "version": "6.0",
        "status": "running",
        "model": AIMLAPI_MODEL,
        "capabilities": [
            "Command string handling (preserves placeholders)",
            "Audio transcription (Whisper API)",
            "Image analysis (dominant colors)",
            "PDF parsing (text + tables)",
            "CSV/Excel analysis (comprehensive stats)",
            "JSON array handling (actual arrays, not strings)",
            "Dynamic file type detection",
            "Parallel file downloads",
            "Smart caching",
            "Multi-step quiz chains"
        ]
    }), 200

@app.route("/health", methods=["GET"])
def health_check():
    """Health check"""
    health = {
        "status": "healthy",
        "llm": client is not None,
        "email": bool(EMAIL),
        "secret": bool(SECRET),
        "model": AIMLAPI_MODEL
    }
    
    try:
        driver = setup_browser()
        driver.quit()
        health["chrome"] = True
    except:
        health["chrome"] = False
    
    return jsonify(health), 200

@app.route("/quiz", methods=["POST"])
def quiz_endpoint():
    """Main quiz webhook"""
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
        return jsonify({"error": "Missing email or url"}), 400
    
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
            secret=data["secret"],
            start_time=start_time,
            timeout=170
        )
        
        total_time = time.time() - start_time
        correct = sum(1 for r in results if r.get("correct"))
        incorrect = sum(1 for r in results if r.get("status") == "incorrect")
        errors = sum(1 for r in results if r.get("status") not in ["correct", "incorrect"])
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🏁 COMPLETED")
        logger.info(f"   ✅ Correct: {correct}")
        logger.info(f"   ❌ Incorrect: {incorrect}")
        logger.info(f"   ⚠️  Errors: {errors}")
        logger.info(f"   ⏱️  Time: {total_time:.2f}s")
        logger.info(f"{'='*70}")
        
        return jsonify({
            "status": "completed",
            "results": results,
            "summary": {
                "total": len(results),
                "correct": correct,
                "incorrect": incorrect,
                "errors": errors,
                "time_seconds": round(total_time, 2),
                "success_rate": round(correct / len(results) * 100, 1) if results else 0
            }
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "time_seconds": round(time.time() - start_time, 2)
        }), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 Universal Quiz Solver v6.0")
    logger.info(f"   Port: {port}")
    logger.info(f"   Model: {AIMLAPI_MODEL}")
    logger.info(f"   LLM: {'✅' if client else '❌'}")
    logger.info(f"   Email: {EMAIL if EMAIL else '❌'}")
    logger.info(f"   Secret: {'✅' if SECRET else '❌'}")
    logger.info(f"{'='*70}\n")
    
    app.run(host="0.0.0.0", port=port, debug=False)