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
from urllib.parse import urljoin, urlparse, unquote
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
    driver.set_page_load_timeout(30)
    driver.set_script_timeout(20)
    return driver

# ============================================================================
# PAGE CONTENT EXTRACTION
# ============================================================================

def extract_page_content(url: str, max_wait: int = 8) -> Dict[str, Any]:
    """
    Extract content from any webpage with intelligent waiting
    """
    logger.info(f"🌐 Extracting: {url}")
    driver = setup_browser()
    
    try:
        driver.get(url)
        
        # Wait for body to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Progressive waiting - check if content stabilizes
        previous_text = ""
        stable_count = 0
        
        for i in range(max_wait):
            time.sleep(0.8)
            
            try:
                current_text = driver.execute_script("return document.body.innerText;")
            except:
                current_text = driver.find_element(By.TAG_NAME, "body").text
            
            if current_text == previous_text and len(current_text) > 30:
                stable_count += 1
                if stable_count >= 2:
                    logger.info(f"✅ Content stabilized after {i+1}s")
                    break
            else:
                stable_count = 0
            
            previous_text = current_text
        
        # Final extraction
        time.sleep(1)
        
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
        except:
            pass
        
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

def download_file(url: str, timeout: int = 20) -> Optional[bytes]:
    """Download any file"""
    
    if url in file_cache:
        logger.info(f"📦 Cached: {url}")
        return file_cache[url]
    
    try:
        logger.info(f"📥 Downloading: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        
        content = resp.content
        file_cache[url] = content
        logger.info(f"✅ Downloaded {len(content)} bytes")
        return content
        
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        return None

# ============================================================================
# FILE PARSING
# ============================================================================

def smart_parse_file(file_content: bytes, url: str) -> Dict[str, Any]:
    """Parse any file type"""
    result = {"type": "unknown", "content": None, "url": url}
    
    try:
        # PDF
        if file_content[:4] == b"%PDF":
            result["type"] = "pdf"
            result["content"] = parse_pdf(file_content)
        # CSV
        elif url.lower().endswith('.csv') or b',' in file_content[:1000]:
            result["type"] = "csv"
            result["content"] = parse_csv(file_content)
        # Excel
        elif url.lower().endswith(('.xlsx', '.xls')):
            result["type"] = "excel"
            result["content"] = parse_excel(file_content)
        # JSON
        elif url.lower().endswith('.json') or file_content[:1] in [b'{', b'[']:
            result["type"] = "json"
            result["content"] = json.loads(file_content.decode('utf-8'))
        # Audio
        elif url.lower().endswith(('.opus', '.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac')):
            result["type"] = "audio"
            result["content"] = {
                "format": url.split('.')[-1],
                "size_bytes": len(file_content),
                "url": url,
                "note": "AUDIO FILE - Cannot transcribe without external service. Inform user that audio transcription is not available."
            }
        # Image
        elif url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg')):
            result["type"] = "image"
            result["content"] = {
                "format": url.split('.')[-1],
                "size_bytes": len(file_content),
                "url": url
            }
        # Text
        else:
            result["type"] = "text"
            result["content"] = file_content.decode('utf-8', errors='ignore')
        
        logger.info(f"✅ Parsed as: {result['type']}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Parse error: {e}")
        result["error"] = str(e)
        return result

def parse_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    """Parse PDF"""
    try:
        result = {"pages": [], "all_text": "", "all_tables": [], "page_count": 0}
        
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            result["page_count"] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_data = {"page_number": page_num}
                
                text = page.extract_text()
                if text:
                    page_data["text"] = text
                    result["all_text"] += f"\n--- Page {page_num} ---\n{text}"
                
                tables = page.extract_tables()
                if tables:
                    page_data["tables"] = []
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0:
                            try:
                                first_row = table[0]
                                looks_like_header = (
                                    len(table) > 1 and 
                                    all(isinstance(cell, str) and cell and 
                                        not str(cell).replace('.','').replace('-','').replace(',','').isdigit()
                                        for cell in first_row if cell)
                                )
                                
                                if looks_like_header:
                                    df = pd.DataFrame(table[1:], columns=table[0])
                                else:
                                    df = pd.DataFrame(table)
                                    df.columns = [f"Col_{i}" for i in range(len(df.columns))]
                                
                                table_info = {
                                    "index": table_idx,
                                    "data": df.to_dict('records')[:50],
                                    "shape": list(df.shape),
                                    "columns": list(df.columns)
                                }
                                page_data["tables"].append(table_info)
                                result["all_tables"].append({"page": page_num, **table_info})
                            except:
                                pass
                
                result["pages"].append(page_data)
        
        logger.info(f"✅ PDF: {result['page_count']} pages, {len(result['all_tables'])} tables")
        return result
        
    except Exception as e:
        logger.error(f"❌ PDF error: {e}")
        return {"error": str(e)}

def parse_csv(csv_content: bytes) -> Dict[str, Any]:
    """Parse CSV"""
    try:
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode("utf-8", errors='ignore')
        
        df = None
        for sep in [',', ';', '\t', '|']:
            try:
                df = pd.read_csv(io.StringIO(csv_content), sep=sep)
                if len(df.columns) > 1 or len(df) > 0:
                    break
            except:
                continue
        
        if df is None or df.empty:
            return {"error": "Could not parse CSV"}
        
        df.columns = [str(col).strip() for col in df.columns]
        
        # Conservative numeric conversion
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    cleaned = df[col].astype(str).str.replace(r'[$,€£¥\s%]', '', regex=True)
                    numeric_col = pd.to_numeric(cleaned, errors='coerce')
                    has_leading_zeros = df[col].astype(str).str.match(r'^0\d').any()
                    
                    if numeric_col.notna().sum() > len(df) * 0.8 and not has_leading_zeros:
                        df[col] = numeric_col
                except:
                    pass
        
        column_analysis = {}
        for col in df.columns:
            col_data = df[col].dropna()
            
            analysis = {
                "dtype": str(df[col].dtype),
                "count": int(len(col_data)),
                "null_count": int(df[col].isna().sum())
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
            else:
                unique_vals = col_data.unique()
                analysis["unique_count"] = len(unique_vals)
                analysis["sample_values"] = [str(v) for v in unique_vals[:10]]
            
            column_analysis[col] = analysis
        
        result = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "column_analysis": column_analysis,
            "data_sample": df.head(100).to_dict('records')
        }
        
        logger.info(f"✅ CSV: {df.shape[0]}×{df.shape[1]}")
        return result
        
    except Exception as e:
        logger.error(f"❌ CSV error: {e}")
        return {"error": str(e)}

def parse_excel(excel_bytes: bytes) -> Dict[str, Any]:
    """Parse Excel"""
    try:
        dfs = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)
        result = {"sheets": {}}
        
        for sheet_name, df in dfs.items():
            csv_str = df.to_csv(index=False)
            result["sheets"][sheet_name] = parse_csv(csv_str.encode())
        
        return result
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# SUBMIT URL EXTRACTION
# ============================================================================

def extract_submit_url(content: str, base_url: str) -> Optional[str]:
    """Extract submit URL"""
    if not content:
        return None
    
    parsed_base = urlparse(base_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
    
    # Try multiple patterns
    patterns = [
        r'"url":\s*"([^"]*submit[^"]*)"',
        r'POST[^h]+(https?://[^\s<>"\']+/submit[^\s<>"\']*)',
        r'href=["\']([^"\']*submit[^"\']*)["\']',
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
# LLM SOLVING - COMPLETELY REWRITTEN
# ============================================================================

def solve_with_llm(quiz_page: Dict[str, Any], downloaded_files: Dict[str, Any], quiz_url: str, user_email: str) -> Dict[str, Any]:
    """Solve quiz with LLM - handles all question types"""
    
    if not client:
        return {"error": "LLM not initialized"}
    
    page_text = quiz_page.get("text", "")
    logger.info(f"📄 Question: {len(page_text)} chars")
    logger.info(f"📦 Files: {len(downloaded_files)}")
    
    # Detect question type from content
    question_lower = page_text.lower()
    
    # Check for audio file
    has_audio = any(f.get('type') == 'audio' for f in downloaded_files.values())
    
    # Check for command patterns
    is_command_question = any(keyword in question_lower for keyword in [
        'uv http', 'git add', 'git commit', 'command', 'shell',
        'curl', 'wget', 'run', 'execute'
    ])
    
    # Check for file path question
    is_path_question = any(keyword in question_lower for keyword in [
        'relative link', 'file path', 'link target', '/project', 'href'
    ])
    
    # Build context
    context_parts = {}
    for url, file_data in downloaded_files.items():
        file_str = json.dumps(file_data, default=str)
        if len(file_str) > 3000:
            summary = {"type": file_data.get("type"), "url": url}
            if file_data.get("type") == "csv":
                summary["columns"] = file_data.get("content", {}).get("columns", [])
                summary["column_analysis"] = file_data.get("content", {}).get("column_analysis", {})
            elif file_data.get("type") == "pdf":
                summary["all_text"] = str(file_data.get("content", {}).get("all_text", ""))[:2000]
                summary["tables"] = file_data.get("content", {}).get("all_tables", [])
            elif file_data.get("type") == "audio":
                summary = file_data.get("content", {})
            else:
                summary["preview"] = str(file_data.get("content", ""))[:2000]
            context_parts[url] = summary
        else:
            context_parts[url] = file_data
    
    context_str = json.dumps(context_parts, indent=2, default=str)[:35000]
    
    # Build dynamic prompt based on question type
    if has_audio:
        special_instructions = f"""
AUDIO FILE DETECTED:
The question references an audio file that requires transcription. 
You CANNOT transcribe audio files directly.

Your answer should be a clear statement that audio transcription is not available, such as:
"Audio file transcription service not available"
OR provide the specific information if it can be found elsewhere in the data.
"""
    elif is_command_question:
        special_instructions = f"""
COMMAND STRING QUESTION:
This question asks you to provide a command or code string.
- Copy the EXACT format from the question
- Replace <your email> with: {user_email}
- Replace any placeholders with actual values
- Return as a plain string (no extra quotes)
- Preserve exact spacing and line breaks

Example: If question shows "uv http get https://example.com?email=<your email>"
Return: uv http get https://example.com?email={user_email} -H "Accept: application/json"
"""
    elif is_path_question:
        special_instructions = f"""
FILE PATH QUESTION:
This question asks for a file path or link target.
- Look for the exact path shown in the question
- Return EXACTLY as written (e.g., /project2/file.md)
- Include leading slash if present
- Do not add or remove any characters
"""
    else:
        special_instructions = """
DATA ANALYSIS QUESTION:
- Calculate from the actual data provided
- Return the exact numeric value or string
- Match the data type requested (number, string, boolean)
"""
    
    prompt = f"""You are solving a quiz question. Read carefully and provide the exact answer.

USER EMAIL: {user_email}
QUIZ URL: {quiz_url}

QUESTION TEXT:
{page_text[:7000]}

{special_instructions}

AVAILABLE DATA:
{context_str}

INSTRUCTIONS:
1. Identify what the question is asking for
2. Find the answer from the question text itself OR the data
3. Return in the exact format requested
4. If you see <your email>, replace with: {user_email}
5. If you see YOUR email, use: {user_email}
6. For commands: copy exactly from question, replacing placeholders
7. For paths: copy exactly as shown
8. For calculations: use the actual data
9. For audio: state that transcription is unavailable

Return ONLY this JSON (no markdown):
{{
    "answer": <exact_answer_here>,
    "reasoning": "brief explanation"
}}

CRITICAL:
- Answer must be the actual value, not a description
- For commands: plain string with {user_email} substituted
- For numbers: just the number
- For text: just the text
- NO placeholders like <your email> in the answer"""

    try:
        logger.info("🤖 Querying LLM...")
        
        response = client.chat.completions.create(
            model=AIMLAPI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"You solve quiz questions precisely. User email: {user_email}. Replace all <your email> placeholders with {user_email}. Return valid JSON only."
                },
                {"role": "user", "content": prompt}
            ],
            #temperature=0.05,
            max_tokens=1000
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"🤖 Response: {response_text[:300]}")
        
        # Clean and parse
        response_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        
        try:
            solution = json.loads(response_text)
        except:
            # Extract JSON
            match = re.search(r'\{.*?"answer".*?\}', response_text, re.DOTALL)
            if match:
                solution = json.loads(match.group())
            else:
                return {"error": "Could not parse response", "raw": response_text[:500]}
        
        answer = solution.get("answer")
        
        # Post-process answer
        if answer and isinstance(answer, str):
            # Replace any remaining email placeholders
            answer = answer.replace("<your email>", user_email)
            answer = answer.replace("your-email", user_email)
            answer = answer.replace("your email", user_email)
            
            # Remove extra quotes if present
            if answer.startswith('"') and answer.endswith('"'):
                answer = answer[1:-1]
            
            solution["answer"] = answer
        
        # Validate answer is not empty/placeholder
        if answer in ["", None, "N/A", "null", "<your email>", "your_answer"]:
            logger.warning("⚠️ Invalid answer, retrying...")
            
            retry_prompt = f"""PREVIOUS FAILED. Question asks for specific answer.

USER EMAIL: {user_email}

QUESTION:
{page_text[:5000]}

Return the ACTUAL answer value (not description, not placeholder).
If it's a command shown in the question, return that exact command with {user_email} replacing email placeholder.
If it's audio file, say "Audio transcription not available"

JSON only:
{{"answer": <actual_value>, "reasoning": "why"}}"""

            retry_response = client.chat.completions.create(
                model=AIMLAPI_MODEL,
                messages=[{"role": "user", "content": retry_prompt}],
                #temperature=0.1
            )
            
            retry_text = retry_response.choices[0].message.content.strip()
            retry_text = re.sub(r'```json\s*|\s*```', '', retry_text).strip()
            
            try:
                solution = json.loads(retry_text)
                answer = solution.get("answer")
                if isinstance(answer, str):
                    answer = answer.replace("<your email>", user_email)
                    solution["answer"] = answer
            except:
                pass
        
        logger.info(f"✅ Answer: {answer}")
        logger.info(f"💡 Reasoning: {solution.get('reasoning', '')[:200]}")
        
        return solution
        
    except Exception as e:
        logger.error(f"❌ LLM error: {e}")
        return {"error": str(e)}

# ============================================================================
# ANSWER SUBMISSION
# ============================================================================

def submit_answer(submit_url: str, email: str, secret: str, quiz_url: str, answer: Any) -> Dict[str, Any]:
    """Submit answer"""
    
    if not submit_url or not submit_url.startswith('http'):
        if submit_url and submit_url.startswith('/'):
            parsed = urlparse(quiz_url)
            submit_url = f"{parsed.scheme}://{parsed.netloc}{submit_url}"
        else:
            return {"error": "Invalid submit URL"}
    
    payload = {"email": email, "secret": secret, "url": quiz_url, "answer": answer}
    
    try:
        logger.info(f"📤 Submitting to: {submit_url}")
        logger.info(f"📤 Answer: {json.dumps(answer) if isinstance(answer, (dict, list)) else str(answer)[:150]}")
        
        resp = requests.post(submit_url, json=payload, timeout=20, headers={"Content-Type": "application/json"})
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
# QUIZ CHAIN PROCESSOR - OPTIMIZED
# ============================================================================

def process_quiz_chain(initial_url: str, email: str, secret: str, start_time: float, timeout: int = 170) -> List[Dict[str, Any]]:
    """Process quiz chain with optimizations"""
    
    current_url = initial_url
    results = []
    iteration = 0
    max_iterations = 30  # Increased to handle 20-25 questions
    
    while current_url and iteration < max_iterations:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        
        if remaining < 20:
            logger.warning(f"⚠️ Timeout approaching ({remaining:.1f}s)")
            results.append({
                "quiz_number": iteration,
                "url": current_url,
                "error": "Timeout",
                "status": "timeout"
            })
            break
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 QUIZ #{iteration}")
        logger.info(f"🔗 URL: {current_url}")
        logger.info(f"⏱️ Remaining: {remaining:.1f}s")
        logger.info(f"{'='*70}")
        
        try:
            # Extract page
            quiz_page = extract_page_content(current_url, max_wait=6)
            
            if quiz_page.get("error"):
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": f"Extraction failed: {quiz_page['error']}",
                    "status": "extraction_failed"
                })
                break
            
            # Find submit URL
            combined = quiz_page.get("html", "") + "\n" + quiz_page.get("text", "")
            submit_url = extract_submit_url(combined, current_url)
            
            if not submit_url:
                logger.error("❌ No submit URL")
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "No submit URL found",
                    "status": "no_submit_url"
                })
                break
            
            logger.info(f"✅ Submit: {submit_url}")
            
            # Download files (parallel)
            downloaded_files = {}
            links = quiz_page.get("links", [])
            
            download_extensions = [
                '.pdf', '.csv', '.json', '.xlsx', '.xls', '.txt', 
                '.opus', '.mp3', '.wav', '.ogg', '.m4a', '.flac'
            ]
            
            download_links = [
                link for link in links 
                if 'submit' not in link.lower() and 
                any(link.lower().endswith(ext) for ext in download_extensions)
            ]
            
            if download_links:
                logger.info(f"📥 Downloading {len(download_links)} files...")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_to_url = {executor.submit(download_file, link): link for link in download_links[:5]}
                    
                    for future in as_completed(future_to_url, timeout=15):
                        link = future_to_url[future]
                        try:
                            content = future.result(timeout=10)
                            if content:
                                parsed = smart_parse_file(content, link)
                                downloaded_files[link] = parsed
                        except Exception as e:
                            logger.error(f"❌ File error {link}: {e}")
            
            logger.info(f"✅ Files processed: {len(downloaded_files)}")
            
            # Solve
            solution = solve_with_llm(quiz_page, downloaded_files, current_url, email)
            
            if "error" in solution:
                logger.error(f"❌ Solve failed: {solution.get('error')}")
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": f"Solve error: {solution.get('error')}",
                    "status": "solving_failed",
                    "solution_raw": solution
                })
                break
            
            answer = solution.get("answer")
            
            if answer is None or answer == "":
                logger.error("❌ No answer")
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "No valid answer",
                    "status": "no_answer",
                    "solution": solution
                })
                break
            
            # Submit (with single retry)
            submission = submit_answer(submit_url, email, secret, current_url, answer)
            
            # Retry once if failed
            if not submission.get("correct") and submission.get("status") != 403:
                logger.warning("⚠️ First attempt failed, retrying...")
                time.sleep(1)
                submission = submit_answer(submit_url, email, secret, current_url, answer)
            
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
            
            # Check for next quiz
            next_url = submission.get("url")
            
            if next_url:
                logger.info(f"➡️ Next: {next_url}")
                current_url = next_url
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
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
                "traceback": traceback.format_exc()[:1000]
            })
            break
    
    return results

# ============================================================================
# FLASK ENDPOINTS
# ============================================================================

@app.route("/", methods=["GET"])
def home():
    """Home endpoint"""
    return jsonify({
        "service": "Dynamic Quiz Solver",
        "version": "6.0",
        "status": "running",
        "model": AIMLAPI_MODEL,
        "capabilities": [
            "Command string extraction with email substitution",
            "Audio file detection and handling",
            "File path extraction",
            "Data analysis and calculations",
            "Multi-format parsing (PDF, CSV, Excel, JSON, Audio)",
            "Parallel file downloads",
            "Handles 20-25+ questions in chain",
            "Optimized for speed and accuracy"
        ],
        "max_quiz_chain": "170 seconds",
        "max_questions": "30",
        "features": {
            "command_strings": True,
            "email_substitution": True,
            "audio_detection": True,
            "path_extraction": True,
            "parallel_downloads": True,
            "optimized_timing": True,
            "dynamic_question_detection": True
        }
    }), 200

@app.route("/health", methods=["GET"])
def health_check():
    """Health check"""
    health = {
        "status": "healthy",
        "llm_ready": client is not None,
        "email_set": bool(EMAIL),
        "secret_set": bool(SECRET),
        "model": AIMLAPI_MODEL
    }
    
    try:
        driver = setup_browser()
        driver.quit()
        health["browser_ready"] = True
    except Exception as e:
        health["browser_ready"] = False
        health["browser_error"] = str(e)
    
    return jsonify(health), 200 if health["status"] == "healthy" else 503

@app.route("/quiz", methods=["POST"])
def quiz_endpoint():
    """Main quiz endpoint"""
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
        num_correct = sum(1 for r in results if r.get("correct"))
        num_incorrect = sum(1 for r in results if r.get("status") == "incorrect")
        num_errors = sum(1 for r in results if r.get("status") not in ["correct", "incorrect"])
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🏁 QUIZ CHAIN COMPLETED")
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
            "traceback": traceback.format_exc()[:1000],
            "time_taken_seconds": round(time.time() - start_time, 2)
        }), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 Dynamic Quiz Solver")
    logger.info(f"   Version: 6.0 - Optimized & Dynamic")
    logger.info(f"   Port: {port}")
    logger.info(f"   Model: {AIMLAPI_MODEL}")
    logger.info(f"   LLM: {'✅' if client else '❌'}")
    logger.info(f"   Email: {EMAIL if EMAIL else '❌'}")
    logger.info(f"   Secret: {'✅' if SECRET else '❌'}")
    logger.info(f"{'='*70}\n")
    
    app.run(host="0.0.0.0", port=port, debug=False)