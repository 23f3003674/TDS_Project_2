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
from urllib.parse import urljoin, urlparse
from typing import Dict, List, Any, Optional

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
        
        for i in range(max_wait):
            time.sleep(1)
            
            # Get content via JavaScript (more reliable)
            try:
                current_text = driver.execute_script("return document.body.innerText;")
                current_html_length = len(driver.page_source)
            except:
                current_text = driver.find_element(By.TAG_NAME, "body").text
                current_html_length = len(driver.page_source)
            
            # Check if content has stabilized
            if (current_text == previous_text and 
                current_html_length == previous_html_length and 
                len(current_text) > 50):
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
    Intelligently parse any file type
    """
    result = {"type": "unknown", "content": None, "error": None, "url": url}
    
    try:
        # Detect file type
        if file_content[:4] == b"%PDF":
            result["type"] = "pdf"
            result["content"] = parse_pdf(file_content)
        
        elif url.lower().endswith('.csv') or b',' in file_content[:1000]:
            result["type"] = "csv"
            result["content"] = parse_csv(file_content)
        
        elif url.lower().endswith(('.xlsx', '.xls')):
            result["type"] = "excel"
            result["content"] = parse_excel(file_content)
        
        elif url.lower().endswith('.json') or file_content[:1] in [b'{', b'[']:
            result["type"] = "json"
            try:
                result["content"] = json.loads(file_content.decode('utf-8'))
            except Exception as e:
                result["error"] = f"Invalid JSON: {e}"
        
        elif url.lower().endswith(('.txt', '.text')):
            result["type"] = "text"
            try:
                result["content"] = file_content.decode('utf-8')
            except:
                result["content"] = file_content.decode('utf-8', errors='ignore')
        
        else:
            # Try as text
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
                                # Create DataFrame
                                if len(table) > 1:
                                    df = pd.DataFrame(table[1:], columns=table[0])
                                else:
                                    df = pd.DataFrame(table)
                                
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
        
        # Try to convert string numbers to numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Remove common formatting
                    cleaned = df[col].astype(str).str.replace(r'[$,€£¥\s%]', '', regex=True)
                    numeric_col = pd.to_numeric(cleaned, errors='coerce')
                    # Only convert if we get reasonable results
                    if numeric_col.notna().sum() > len(df) * 0.5:
                        df[col] = numeric_col
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
        
        result = {
            "shape": list(df.shape),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "column_analysis": column_analysis,
            "first_rows": df.head(20).to_dict('records'),
            "last_rows": df.tail(10).to_dict('records'),
            "data_sample": df.to_dict('records')[:300] if len(df) <= 300 else df.sample(n=300).to_dict('records')
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
    
    # Strategy 5: Fallback - construct from base URL
    if 'submit' in content.lower():
        fallback_url = f"{base_domain}/submit"
        logger.info(f"⚠️ Using fallback submit URL: {fallback_url}")
        return fallback_url
    
    logger.warning("❌ No submit URL found in content")
    return None

# ============================================================================
# LLM SOLVING
# ============================================================================

def solve_with_llm(quiz_page: Dict[str, Any], downloaded_files: Dict[str, Any], quiz_url: str) -> Dict[str, Any]:
    """
    Solve quiz with LLM using advanced prompting
    """
    if not client:
        return {"error": "LLM not initialized"}
    
    page_text = quiz_page.get("text", "")
    
    logger.info(f"📄 Question length: {len(page_text)} chars")
    logger.info(f"📦 Data files: {len(downloaded_files)}")
    
    # Build comprehensive context
    context_str = json.dumps(downloaded_files, indent=2, default=str)[:20000]
    
    # Create the prompt
    prompt = f"""You are a precise data analyst solving a quiz. Your job is to read the question carefully, analyze the provided data, and return the EXACT answer requested.

QUESTION:
{page_text[:6000]}

AVAILABLE DATA:
{context_str}

CRITICAL INSTRUCTIONS:
1. READ the question carefully - what EXACTLY is it asking for?
2. IDENTIFY which data source contains the answer
3. PERFORM the required calculation/extraction on the ACTUAL data (not example values)
4. RETURN the answer in the exact format requested

ANSWER FORMAT GUIDE:
- If asking for a number (sum, count, average): return as integer or float
- If asking for text (code, name, secret): return as string
- If asking for yes/no: return as boolean (true/false)
- If asking for multiple values: return as array or object

CALCULATION EXAMPLES:
- "sum of column X" → Calculate: sum of all values in column X from the data
- "count rows where Y > 10" → Count: number of rows meeting condition
- "average of Z" → Calculate: mean of column Z
- "value on page 2" → Extract: from page 2 of the PDF data
- "secret code" → Extract: the actual code from scraped content

CRITICAL RULES:
✓ Use ACTUAL data provided above
✗ Do NOT use example values from the question
✓ Perform EXACT calculations (don't round unless asked)
✗ Do NOT return placeholder values like "N/A", "null", "your_answer"
✓ Match the data type requested (number vs string vs boolean)
✗ Do NOT guess - analyze the data carefully

Return ONLY valid JSON (no markdown, no code blocks):
{{
    "answer": <your_calculated_answer>,
    "reasoning": "step-by-step: 1) identified question asks for X, 2) found data in Y, 3) calculated/extracted as Z",
    "confidence": "high/medium/low",
    "data_used": "which data source/column was used"
}}

IMPORTANT: Your answer MUST come from analyzing the actual data above."""

    try:
        logger.info("🤖 Querying LLM...")
        
        response = client.chat.completions.create(
            model=AIMLAPI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise data analyst. Analyze data carefully, perform exact calculations, and return valid JSON with the correct answer type. Never use placeholder values."
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
            "your_answer", "your answer", "calculate this", "TODO"
        ]
        
        answer_str = str(answer).lower().strip() if answer is not None else ""
        
        if answer in invalid_answers or answer_str in invalid_answers:
            logger.warning("⚠️ LLM returned invalid/placeholder answer, retrying with emphasis...")
            
            # Retry with more explicit instructions
            retry_prompt = f"""PREVIOUS ATTEMPT FAILED - You returned a placeholder/invalid answer.

The question is asking for a SPECIFIC value that can be calculated or extracted from the data.

QUESTION (Read carefully):
{page_text[:4000]}

DATA (Analyze this):
{context_str[:15000]}

STEP-BY-STEP PROCESS:
1. What is the question asking for? (e.g., sum of column X, secret code, count of rows)
2. Where is this information in the data? (which file, which column, which page)
3. What is the calculation? (sum, count, extract, filter, etc.)
4. What is the ACTUAL value? (calculate it precisely from the data)

Example thinking:
- If question asks "sum of 'value' column" → Find 'value' column in data → Calculate sum of all numbers
- If question asks "secret on page 2" → Find page 2 in PDF data → Extract the secret text

Return valid JSON with the ACTUAL answer (must be a real value, not placeholder):
{{
    "answer": <actual_calculated_value>,
    "reasoning": "detailed explanation of how you got this answer"
}}

DO NOT RETURN: null, N/A, placeholder, empty string, or example values from the question itself."""

            try:
                retry_response = client.chat.completions.create(
                    model=AIMLAPI_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You must return the actual answer calculated from the data. No placeholders allowed."
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
                    logger.info(f"✅ Retry successful, new answer: {answer}")
                except:
                    logger.error("❌ Retry JSON parsing also failed")
                    return {
                        "error": "Could not get valid answer after retry",
                        "raw": retry_text[:1500]
                    }
            except Exception as e:
                logger.error(f"❌ Retry request failed: {e}")
                # Continue with original answer despite being invalid
        
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
        if remaining < 30:
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
        logger.info(f"⏱️  Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
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
            
            # Step 3: Download and parse all linked files
            downloaded_files = {}
            links = quiz_page.get("links", [])
            
            logger.info(f"🔍 Found {len(links)} links to process")
            
            for link in links:
                # Skip submit URLs
                if 'submit' in link.lower():
                    continue
                
                # Determine if we should download or scrape
                parsed_link = urlparse(link)
                path = parsed_link.path.lower()
                
                # Known file extensions to download
                download_extensions = [
                    '.pdf', '.csv', '.json', '.xlsx', '.xls', 
                    '.txt', '.xml', '.tsv', '.dat'
                ]
                
                should_download = any(path.endswith(ext) for ext in download_extensions)
                
                if should_download:
                    logger.info(f"📥 Downloading file: {link}")
                    file_content = download_file(link)
                    if file_content:
                        parsed = smart_parse_file(file_content, link)
                        downloaded_files[link] = parsed
                else:
                    # Scrape as a page (might contain additional data)
                    if len(downloaded_files) < 5:  # Limit scraping to prevent timeout
                        logger.info(f"🌐 Scraping page: {link}")
                        scraped = extract_page_content(link, max_wait=5)
                        if not scraped.get("error") and scraped.get("text"):
                            downloaded_files[link] = {
                                "type": "scraped_page",
                                "content": scraped.get("text", ""),
                                "url": link
                            }
            
            logger.info(f"✅ Processed {len(downloaded_files)} data sources")
            
            # Step 4: Solve with LLM
            solution = solve_with_llm(quiz_page, downloaded_files, current_url)
            
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
                logger.info(f"➡️  Next quiz: {next_url}")
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
        "version": "5.0",
        "status": "running",
        "model": AIMLAPI_MODEL,
        "capabilities": [
            "JavaScript-rendered web scraping",
            "Comprehensive statistical analysis (mean, median, std, quartiles, IQR)",
            "ML predictions & classifications",
            "Data filtering, aggregation & transformation",
            "Complex calculations (z-scores, probabilities, distributions)",
            "Visualization & narrative generation",
            "Multi-format file parsing (PDF, CSV, Excel, JSON, Text)",
            "API integration & data sourcing",
            "Intelligent answer type detection",
            "Automatic retry on failures",
            "Smart caching for performance"
        ],
        "max_time_per_quiz_chain": "170 seconds",
        "features": {
            "file_caching": True,
            "retry_on_failure": True,
            "multi_format_parsing": True,
            "dynamic_submit_url": True,
            "comprehensive_stats": True
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
            secret=data["secret"],
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
        logger.info(f"   ⚠️  Errors: {num_errors}")
        logger.info(f"   ⏱️  Total time: {total_time:.2f}s")
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
    logger.info(f"   Version: 5.0")
    logger.info(f"   Port: {port}")
    logger.info(f"   Model: {AIMLAPI_MODEL}")
    logger.info(f"   LLM: {'✅ Ready' if client else '❌ Not initialized'}")
    logger.info(f"   Email: {EMAIL if EMAIL else '❌ Not set'}")
    logger.info(f"   Secret: {'✅ Set' if SECRET else '❌ Not set'}")
    logger.info(f"{'='*70}\n")
    
    app.run(host="0.0.0.0", port=port, debug=False)