#!/usr/bin/env python3
"""
Universal Dynamic Quiz Solver - Complete Final Version
Handles ANY quiz type with precise command handling
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

file_cache: Dict[str, bytes] = {}

# ============================================================================
# BROWSER SETUP
# ============================================================================

def setup_browser() -> webdriver.Chrome:
    """Initialize headless Chrome"""
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    if CHROME_BINARY and os.path.exists(CHROME_BINARY):
        opts.binary_location = CHROME_BINARY
    
    service = Service(executable_path=CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(45)
    return driver

# ============================================================================
# PAGE EXTRACTION
# ============================================================================

def extract_page_content(url: str, max_wait: int = 10) -> Dict[str, Any]:
    """Extract content from webpage"""
    logger.info(f"🌐 Extracting: {url}")
    driver = setup_browser()
    
    try:
        driver.get(url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        previous_text = ""
        stable_count = 0
        
        for i in range(max_wait):
            time.sleep(1)
            try:
                current_text = driver.execute_script("return document.body.innerText;")
            except:
                current_text = driver.find_element(By.TAG_NAME, "body").text
            
            if current_text == previous_text and len(current_text) > 50:
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
        return {"html": "", "text": f"Error: {str(e)}", "links": [], "url": url}
    finally:
        try:
            driver.quit()
        except:
            pass

# ============================================================================
# FILE DOWNLOADING
# ============================================================================

def download_file(url: str, timeout: int = 30) -> Optional[bytes]:
    """Download file with caching"""
    if url in file_cache:
        logger.info(f"📦 Cache hit: {url}")
        return file_cache[url]
    
    try:
        logger.info(f"📥 Downloading: {url}")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        
        content = resp.content
        logger.info(f"✅ Downloaded {len(content)} bytes")
        file_cache[url] = content
        return content
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        return None

# ============================================================================
# FILE PROCESSORS
# ============================================================================

def analyze_image(image_bytes: bytes, url: str) -> Dict[str, Any]:
    """Analyze image for dominant color"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        pixels = list(img.getdata())
        color_counts = {}
        
        for pixel in pixels:
            rgb_hex = '#{:02x}{:02x}{:02x}'.format(pixel[0], pixel[1], pixel[2])
            color_counts[rgb_hex] = color_counts.get(rgb_hex, 0) + 1
        
        dominant = max(color_counts.items(), key=lambda x: x[1])
        top_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        logger.info(f"🎨 Image: {img.width}x{img.height}, Dominant: {dominant[0]}")
        
        return {
            "dominant_color": dominant[0],
            "dominant_count": dominant[1],
            "total_pixels": len(pixels),
            "unique_colors": len(color_counts),
            "dimensions": {"width": img.width, "height": img.height},
            "top_colors": [{"color": c[0], "count": c[1]} for c in top_colors]
        }
    except Exception as e:
        logger.error(f"❌ Image error: {e}")
        return {"error": str(e)}

def transcribe_audio(audio_bytes: bytes, url: str) -> Dict[str, Any]:
    """Transcribe audio using Whisper with form-data"""
    try:
        audio_format = url.split('.')[-1].lower()
        if audio_format not in ['opus', 'mp3', 'wav', 'ogg', 'm4a', 'flac']:
            audio_format = 'opus'
        
        logger.info(f"🎤 Transcribing {audio_format} ({len(audio_bytes)} bytes)...")
        
        # Use form-data for audio transcription
        files = {
            'file': (f'audio.{audio_format}', io.BytesIO(audio_bytes), f'audio/{audio_format}')
        }
        data = {
            'model': 'whisper-1',
            'response_format': 'text'
        }
        
        headers = {
            'Authorization': f'Bearer {AIMLAPI_API_KEY}'
        }
        
        response = requests.post(
            f'{AIMLAPI_BASE_URL}/audio/transcriptions',
            headers=headers,
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            transcribed_text = response.text.strip()
            logger.info(f"✅ Transcription: '{transcribed_text}'")
            return {
                "transcription": transcribed_text,
                "format": audio_format,
                "size": len(audio_bytes)
            }
        else:
            logger.error(f"❌ Transcription failed: {response.status_code} - {response.text}")
            return {
                "error": f"HTTP {response.status_code}: {response.text}",
                "format": audio_format,
                "size": len(audio_bytes)
            }
            
    except Exception as e:
        logger.error(f"❌ Audio error: {e}")
        return {"error": str(e)}

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
                                has_header = (len(table) > 1 and 
                                            all(isinstance(c, str) and c for c in first_row if c))
                                
                                if has_header:
                                    df = pd.DataFrame(table[1:], columns=table[0])
                                else:
                                    df = pd.DataFrame(table)
                                
                                table_info = {
                                    "data": df.to_dict('records'),
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
                if len(df.columns) > 1:
                    logger.info(f"✅ CSV parsed with sep: '{sep}'")
                    break
            except:
                continue
        
        if df is None or df.empty:
            return {"error": "Could not parse CSV"}
        
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
        
        column_analysis = {}
        for col in df.columns:
            col_data = df[col].dropna()
            analysis = {"dtype": str(df[col].dtype), "non_null": int(df[col].notna().sum())}
            
            if pd.api.types.is_numeric_dtype(df[col]) and len(col_data) > 0:
                analysis.update({
                    "sum": float(col_data.sum()),
                    "mean": float(col_data.mean()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max())
                })
                logger.info(f"  📊 '{col}': sum={analysis['sum']:.2f}")
            
            column_analysis[col] = analysis
        
        result = {
            "rows": len(df),
            "columns": list(df.columns),
            "column_analysis": column_analysis,
            "data": df.to_dict('records')
        }
        
        logger.info(f"✅ CSV: {df.shape[0]} rows × {df.shape[1]} cols")
        return result
    except Exception as e:
        logger.error(f"❌ CSV error: {e}")
        return {"error": str(e)}

def smart_parse_file(file_content: bytes, url: str) -> Dict[str, Any]:
    """Universal file parser - checks extension first"""
    result = {"type": "unknown", "content": None, "url": url}
    
    try:
        url_lower = url.lower()
        
        # Check by extension first (most reliable)
        if url_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
            result["type"] = "image"
            result["content"] = analyze_image(file_content, url)
        elif url_lower.endswith(('.opus', '.mp3', '.wav', '.ogg', '.m4a', '.flac')):
            result["type"] = "audio"
            result["content"] = transcribe_audio(file_content, url)
        elif url_lower.endswith('.pdf') or file_content[:4] == b"%PDF":
            result["type"] = "pdf"
            result["content"] = parse_pdf(file_content)
        elif url_lower.endswith('.csv'):
            result["type"] = "csv"
            result["content"] = parse_csv(file_content)
        elif url_lower.endswith('.json'):
            result["type"] = "json"
            result["content"] = json.loads(file_content.decode('utf-8'))
        elif url_lower.endswith(('.txt', '.md')):
            result["type"] = "text"
            result["content"] = file_content.decode('utf-8', errors='ignore')
        else:
            # Fallback: try as text
            result["type"] = "text"
            result["content"] = file_content.decode('utf-8', errors='ignore')
        
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
    """Extract submit URL"""
    if not content:
        return None
    
    parsed = urlparse(base_url)
    base_domain = f"{parsed.scheme}://{parsed.netloc}"
    
    patterns = [
        r'"url":\s*"([^"]*submit[^"]*)"',
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
    
    if 'submit' in content.lower():
        return f"{base_domain}/submit"
    
    return None

# ============================================================================
# LLM SOLVING ENGINE
# ============================================================================

def solve_with_llm(quiz_page: Dict[str, Any], downloaded_files: Dict[str, Any], 
                   quiz_url: str, user_email: str) -> Dict[str, Any]:
    """Universal LLM solver with improved command handling"""
    if not client:
        return {"error": "LLM not initialized"}
    
    page_text = quiz_page.get("text", "")
    logger.info(f"📄 Question: {len(page_text)} chars")
    logger.info(f"📦 Files: {len(downloaded_files)}")
    
    # Build context
    context_str = json.dumps(downloaded_files, indent=2, default=str)[:50000]
    
    prompt = f"""You are a universal problem solver. Read the question carefully and provide the EXACT answer.

USER EMAIL: {user_email}

QUESTION:
{page_text[:10000]}

AVAILABLE DATA:
{context_str}

CRITICAL RULES:

1. COMMAND STRINGS WITH PLACEHOLDERS:
   - If you see "<your email>" in a command example, KEEP IT EXACTLY as "<your email>"
   - DO NOT replace "<your email>" with the actual email {user_email}
   - Return the command string with the placeholder preserved EXACTLY as shown
   - Example: Command shows "url?email=<your email>" → Return: "url?email=<your email>"

2. GIT COMMANDS:
   - Use SIMPLE form: "git add filename" (NOT "git add -- filename")
   - No extra flags like "--" unless specifically shown in question
   - Follow exact format: "git add env.sample\\ngit commit -m \\"message\\""

3. DATA NORMALIZATION (for JSON/CSV questions):
   - Convert keys to snake_case (lowercase_with_underscores)
   - Convert dates to ISO-8601 format (YYYY-MM-DD)
   - Sort by ID field in ascending order
   - Return as actual JSON array: [{{}}, {{}}] NOT string

4. FILE COUNTING (JSON tree structures):
   - Parse the tree/list structure in the JSON data
   - Filter items by "path" field matching the pathPrefix
   - Count items where path ends with specified extension (e.g., ".md")
   - Apply any modulo operations as instructed in question
   - Example: "Count .md files under 'project-1/' then add (email_length % 2)"

5. IMAGE/AUDIO:
   - Image: Return the dominant_color field value (hex format)
   - Audio: Return the transcription field value (exact text)

6. PATH/URL QUESTIONS:
   - Return exact paths as shown in question
   - Preserve leading slashes and formatting

ANSWER FORMATS EXAMPLES:
- Command with placeholder: "uv http get url?email=<your email> -H \\"Accept: application/json\\""
- Git commands: "git add env.sample\\ngit commit -m \\"message\\""
- JSON array: [{{"id": 1, "name": "Alpha"}}, {{"id": 2, "name": "Beta"}}]
- Number: 42
- String: "text value"
- Hex color: "#rrggbb"
- Path: "/project2/file.md"

IMPORTANT:
- Preserve ALL placeholders like "<your email>" - do NOT substitute actual values
- Use simple git syntax without extra flags
- Return JSON arrays as actual arrays, not escaped strings
- For file counting, carefully parse the JSON structure and count matching items

Return ONLY valid JSON:
{{
    "answer": <exact_answer>,
    "reasoning": "brief step-by-step explanation",
    "confidence": "high/medium/low"
}}"""

    try:
        logger.info("🤖 Querying LLM...")
        
        response = client.chat.completions.create(
            model=AIMLAPI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are precise. ALWAYS preserve '<your email>' placeholders. Use simple git commands. Return arrays as arrays not strings. Count files by parsing JSON structures."
                },
                {"role": "user", "content": prompt}
            ],
            #temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"🤖 Raw response: {response_text[:300]}")
        
        # Clean markdown
        response_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        
        try:
            solution = json.loads(response_text)
        except:
            match = re.search(r'\{.*?"answer".*?\}', response_text, re.DOTALL)
            if match:
                solution = json.loads(match.group())
            else:
                return {"error": "Invalid JSON", "raw": response_text[:500]}
        
        if "answer" not in solution:
            return {"error": "No answer field"}
        
        answer = solution["answer"]
        
        # Post-processing: Convert JSON string arrays to actual arrays
        if isinstance(answer, str) and answer.strip().startswith('['):
            try:
                parsed = json.loads(answer)
                if isinstance(parsed, list):
                    logger.info("✅ Converted string to array")
                    answer = parsed
                    solution["answer"] = answer
            except:
                pass
        
        logger.info(f"✅ Answer: {json.dumps(answer, default=str)[:200]} (type: {type(answer).__name__})")
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
    """Submit answer"""
    if not submit_url or not submit_url.startswith('http'):
        if submit_url and submit_url.startswith('/'):
            parsed = urlparse(quiz_url)
            submit_url = f"{parsed.scheme}://{parsed.netloc}{submit_url}"
        else:
            return {"error": f"Invalid submit URL"}
    
    payload = {"email": email, "secret": secret, "url": quiz_url, "answer": answer}
    
    try:
        logger.info(f"📤 Submitting to: {submit_url}")
        logger.info(f"📤 Answer: {json.dumps(answer, default=str)[:200]}")
        
        resp = requests.post(submit_url, json=payload, timeout=30,
                           headers={"Content-Type": "application/json"})
        
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
    """Process quiz chain"""
    current_url = initial_url
    results = []
    iteration = 0
    max_iterations = 20
    
    while current_url and iteration < max_iterations:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        
        if remaining < 25:
            logger.warning(f"⚠️ Only {remaining:.1f}s left")
            results.append({
                "quiz_number": iteration,
                "url": current_url,
                "error": "Timeout",
                "status": "timeout"
            })
            break
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 QUIZ #{iteration}")
        logger.info(f"🔗 {current_url}")
        logger.info(f"⏱️  Elapsed: {elapsed:.1f}s | Remaining: {remaining:.1f}s")
        logger.info(f"{'='*70}")
        
        try:
            quiz_page = extract_page_content(current_url)
            
            if quiz_page.get("error"):
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "Extraction failed",
                    "status": "failed"
                })
                break
            
            content = quiz_page.get("html", "") + "\n" + quiz_page.get("text", "")
            submit_url = extract_submit_url(content, current_url)
            
            if not submit_url:
                logger.error("❌ No submit URL")
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "No submit URL",
                    "status": "failed"
                })
                break
            
            logger.info(f"✅ Submit URL: {submit_url}")
            
            # Download files
            downloaded_files = {}
            links = quiz_page.get("links", [])
            download_links = [l for l in links if 'submit' not in l.lower()]
            
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
                            logger.error(f"❌ Error: {e}")
            
            logger.info(f"✅ Processed {len(downloaded_files)} files")
            
            solution = solve_with_llm(quiz_page, downloaded_files, current_url, email)
            
            if "error" in solution:
                logger.error(f"❌ Solving failed")
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": solution["error"],
                    "status": "failed"
                })
                break
            
            answer = solution.get("answer")
            if answer is None or answer == "":
                logger.error("❌ No answer")
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "No answer",
                    "status": "failed"
                })
                break
            
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
    return jsonify({
        "service": "Universal Quiz Solver",
        "version": "7.0",
        "status": "running",
        "model": AIMLAPI_MODEL,
        "capabilities": [
            "Command string handling (preserves <your email> placeholders)",
            "Audio transcription (Whisper API)",
            "Image analysis (dominant colors)",
            "PDF parsing (text + tables)",
            "CSV/Excel analysis",
            "JSON tree traversal and file counting",
            "Dynamic file type detection",
            "Parallel file downloads"
        ]
    }), 200

@app.route("/health", methods=["GET"])
def health_check():
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
            "traceback": traceback.format_exc()[:2000],
            "time_seconds": round(time.time() - start_time, 2)
        }), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 Universal Quiz Solver v7.0")
    logger.info(f"   Port: {port}")
    logger.info(f"   Model: {AIMLAPI_MODEL}")
    logger.info(f"   LLM: {'✅' if client else '❌'}")
    logger.info(f"   Email: {EMAIL if EMAIL else '❌'}")
    logger.info(f"   Secret: {'✅' if SECRET else '❌'}")
    logger.info(f"{'='*70}\n")
    
    app.run(host="0.0.0.0", port=port, debug=False)