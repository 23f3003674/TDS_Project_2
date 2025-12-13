#!/usr/bin/env python3
"""
Dynamic Quiz Solver - Production Ready v7.0
Handles ALL question types with intelligent detection
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
from collections import Counter

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
EMAIL = os.getenv("STUDENT_EMAIL")
SECRET = os.getenv("STUDENT_SECRET")
AIMLAPI_BASE_URL = os.getenv("AIMLAPI_BASE_URL", "https://aipipe.org/openai/v1")
AIMLAPI_API_KEY = os.getenv("AIMLAPI_API_KEY")
AIMLAPI_MODEL = os.getenv("AIMLAPI_MODEL", "gpt-5-nano")
CHROME_BINARY = os.getenv("CHROME_BIN", "/usr/bin/chromium")
CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")

client = None
if AIMLAPI_API_KEY:
    try:
        client = OpenAI(api_key=AIMLAPI_API_KEY, base_url=AIMLAPI_BASE_URL)
        logger.info("✅ LLM initialized")
    except Exception as e:
        logger.error(f"❌ LLM init failed: {e}")

file_cache: Dict[str, bytes] = {}

def setup_browser() -> webdriver.Chrome:
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
    driver.set_page_load_timeout(25)
    return driver

def extract_page_content(url: str, max_wait: int = 6) -> Dict[str, Any]:
    logger.info(f"🌐 Extracting: {url}")
    driver = setup_browser()
    
    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        previous_text = ""
        stable = 0
        
        for i in range(max_wait):
            time.sleep(0.7)
            try:
                current = driver.execute_script("return document.body.innerText;")
            except:
                current = driver.find_element(By.TAG_NAME, "body").text
            
            if current == previous_text and len(current) > 30:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            previous_text = current
        
        time.sleep(1)
        page_text = driver.execute_script("return document.body.innerText;")
        page_source = driver.page_source
        
        links = []
        for link_el in driver.find_elements(By.TAG_NAME, "a"):
            try:
                href = link_el.get_attribute("href")
                if href and not href.startswith(("javascript:", "mailto:", "#", "data:")):
                    links.append(urljoin(url, href))
            except:
                pass
        
        logger.info(f"✅ Extracted: {len(page_text)} chars, {len(links)} links")
        return {"html": page_source, "text": page_text, "links": links, "url": url}
    except Exception as e:
        logger.error(f"❌ Extract error: {e}")
        return {"html": "", "text": "", "links": [], "url": url, "error": str(e)}
    finally:
        try:
            driver.quit()
        except:
            pass

def download_file(url: str) -> Optional[bytes]:
    if url in file_cache:
        return file_cache[url]
    
    try:
        logger.info(f"📥 Downloading: {url}")
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        file_cache[url] = resp.content
        logger.info(f"✅ Downloaded {len(resp.content)} bytes")
        return resp.content
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        return None

def get_dominant_color_from_image(image_bytes: bytes) -> str:
    """Extract dominant color from image"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize((150, 150))  # Resize for faster processing
        
        pixels = list(img.getdata())
        color_counts = Counter(pixels)
        dominant = color_counts.most_common(1)[0][0]
        
        hex_color = '#{:02x}{:02x}{:02x}'.format(dominant[0], dominant[1], dominant[2])
        logger.info(f"🎨 Dominant color: {hex_color}")
        return hex_color
    except Exception as e:
        logger.error(f"❌ Image color extraction error: {e}")
        return "#000000"

def smart_parse_file(file_content: bytes, url: str) -> Dict[str, Any]:
    result = {"type": "unknown", "content": None, "url": url}
    
    try:
        # Check file extension first
        url_lower = url.lower()
        
        # Audio files
        if any(url_lower.endswith(ext) for ext in ['.opus', '.mp3', '.wav', '.ogg', '.m4a', '.flac']):
            result["type"] = "audio"
            result["content"] = {
                "format": url.split('.')[-1],
                "size": len(file_content),
                "note": "Audio file - transcription not available in this environment"
            }
            return result
        
        # Image files
        if any(url_lower.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']):
            result["type"] = "image"
            dominant_color = get_dominant_color_from_image(file_content)
            result["content"] = {
                "format": url.split('.')[-1],
                "size": len(file_content),
                "dominant_color": dominant_color,
                "url": url
            }
            return result
        
        # PDF
        if file_content[:4] == b"%PDF" or url_lower.endswith('.pdf'):
            result["type"] = "pdf"
            result["content"] = parse_pdf(file_content)
            return result
        
        # JSON
        if url_lower.endswith('.json'):
            result["type"] = "json"
            try:
                result["content"] = json.loads(file_content.decode('utf-8'))
            except:
                result["content"] = file_content.decode('utf-8', errors='ignore')
            return result
        
        # CSV
        if url_lower.endswith('.csv') or b',' in file_content[:500]:
            result["type"] = "csv"
            result["content"] = parse_csv(file_content)
            return result
        
        # Excel
        if any(url_lower.endswith(ext) for ext in ['.xlsx', '.xls']):
            result["type"] = "excel"
            result["content"] = parse_excel(file_content)
            return result
        
        # Default to text
        result["type"] = "text"
        result["content"] = file_content.decode('utf-8', errors='ignore')
        return result
        
    except Exception as e:
        logger.error(f"❌ Parse error: {e}")
        result["error"] = str(e)
        return result

def parse_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    try:
        result = {"pages": [], "all_text": "", "all_tables": []}
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    result["all_text"] += f"\n--- Page {page_num} ---\n{text}"
                
                for table in page.extract_tables() or []:
                    if table:
                        try:
                            df = pd.DataFrame(table[1:], columns=table[0]) if len(table) > 1 else pd.DataFrame(table)
                            result["all_tables"].append({"page": page_num, "data": df.to_dict('records')[:30]})
                        except:
                            pass
        return result
    except Exception as e:
        return {"error": str(e)}

def parse_csv(csv_content: bytes) -> Dict[str, Any]:
    try:
        text = csv_content.decode('utf-8', errors='ignore')
        df = None
        
        for sep in [',', ';', '\t', '|']:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep)
                if len(df.columns) > 1 or len(df) > 0:
                    break
            except:
                continue
        
        if df is None:
            return {"error": "Could not parse CSV"}
        
        df.columns = [str(c).strip() for c in df.columns]
        
        # Conservative numeric conversion
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    cleaned = df[col].astype(str).str.replace(r'[$,€£¥\s%]', '', regex=True)
                    numeric = pd.to_numeric(cleaned, errors='coerce')
                    if numeric.notna().sum() > len(df) * 0.8 and not df[col].astype(str).str.match(r'^0\d').any():
                        df[col] = numeric
                except:
                    pass
        
        col_analysis = {}
        for col in df.columns:
            data = df[col].dropna()
            analysis = {"dtype": str(df[col].dtype), "count": len(data)}
            
            if pd.api.types.is_numeric_dtype(df[col]) and len(data) > 0:
                analysis.update({
                    "sum": float(data.sum()),
                    "mean": float(data.mean()),
                    "median": float(data.median()),
                    "min": float(data.min()),
                    "max": float(data.max())
                })
            
            col_analysis[col] = analysis
        
        return {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "column_analysis": col_analysis,
            "data_sample": df.to_dict('records'),
            "full_data": df.to_dict('records')
        }
    except Exception as e:
        return {"error": str(e)}

def parse_excel(excel_bytes: bytes) -> Dict[str, Any]:
    try:
        dfs = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)
        result = {"sheets": {}}
        for name, df in dfs.items():
            result["sheets"][name] = parse_csv(df.to_csv(index=False).encode())
        return result
    except Exception as e:
        return {"error": str(e)}

def extract_submit_url(content: str, base_url: str) -> Optional[str]:
    parsed = urlparse(base_url)
    base_domain = f"{parsed.scheme}://{parsed.netloc}"
    
    patterns = [
        r'"url":\s*"([^"]*submit[^"]*)"',
        r'POST[^h]+(https?://[^\s<>"\']+/submit[^\s<>"\']*)',
        r'(https?://[^\s<>"\']+/submit[^\s<>"\']*)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            url = matches[0].strip()
            return f"{base_domain}{url}" if url.startswith('/') else url
    
    if 'submit' in content.lower():
        return f"{base_domain}/submit"
    
    return None

def solve_with_llm(quiz_page: Dict[str, Any], downloaded_files: Dict[str, Any], quiz_url: str, user_email: str) -> Dict[str, Any]:
    """Universal quiz solver"""
    
    if not client:
        return {"error": "LLM not initialized"}
    
    page_text = quiz_page.get("text", "")
    
    # Build context
    context_parts = {}
    for url, file_data in downloaded_files.items():
        ftype = file_data.get("type")
        
        if ftype == "image":
            context_parts[url] = {
                "type": "image",
                "dominant_color": file_data.get("content", {}).get("dominant_color"),
                "url": url
            }
        elif ftype == "audio":
            context_parts[url] = {
                "type": "audio",
                "note": "Audio transcription not available"
            }
        elif ftype == "json":
            context_parts[url] = {"type": "json", "content": file_data.get("content")}
        elif ftype == "csv":
            content = file_data.get("content", {})
            context_parts[url] = {
                "type": "csv",
                "columns": content.get("columns"),
                "data": content.get("full_data", content.get("data_sample"))
            }
        else:
            context_parts[url] = file_data
    
    context_str = json.dumps(context_parts, indent=2, default=str)[:40000]
    
    prompt = f"""You are solving a quiz question. Read the question VERY carefully and provide the EXACT answer requested.

USER EMAIL: {user_email}
EMAIL LENGTH: {len(user_email)}

QUESTION TEXT:
{page_text}

AVAILABLE DATA:
{context_str}

CRITICAL INSTRUCTIONS:

1. READ THE QUESTION WORD-BY-WORD. What EXACTLY does it ask for?

2. COMMON QUESTION TYPES:
   - "Submit the command string:" → Return the EXACT command AS SHOWN in the question
     → If it shows <your email>, keep it as "<your email>" (don't replace!)
     → Example: "uv http get URL?email=<your email>" → Return EXACTLY that
   
   - "What is the [path/link]?" → Return the exact path from the question
   
   - "Dominant color" → Return the hex color from image data (e.g., "#b45a1e")
   
   - "Normalize/transform JSON" → Apply the transformations described, return JSON
   
   - "Count .md files" → Count from JSON tree data, add modulo calculation if asked
   
   - "Transcribe audio" → State "Audio transcription not available"
   
   - Data calculations → Use the actual data provided

3. ANSWER FORMAT:
   - Command strings: Return as plain string EXACTLY as shown
   - Paths: Return as string (e.g., "/project2/file.md")
   - Colors: Return as hex string (e.g., "#b45a1e")
   - JSON: Return as JSON object/array
   - Numbers: Return as integer or float
   - Audio: Return "Audio transcription not available"

4. KEY RULES:
   ✓ If question shows "<your email>" in a command, KEEP IT as "<your email>"
   ✓ For image dominant color, use the dominant_color field from image data
   ✓ For JSON transformations, follow the exact rules given (snake_case, ISO dates, sorting, etc.)
   ✓ For counting files, parse the JSON tree and count matching files
   ✓ Always match the data type requested

Return ONLY valid JSON:
{{
    "answer": <exact_answer>,
    "reasoning": "brief explanation"
}}

EXAMPLES:
Question: "Submit the command string: uv http get URL?email=<your email>"
Answer: {{"answer": "uv http get URL?email=<your email> -H \\"Accept: application/json\\"", "reasoning": "Copied exact command from question"}}

Question: "What is the dominant color?"
Answer: {{"answer": "#b45a1e", "reasoning": "From image dominant_color field"}}

Question: "Count .md files under docs/, add (email length mod 2)"
Answer: {{"answer": 5, "reasoning": "Found 4 .md files, email length 35, 35%2=1, total 4+1=5"}}"""

    try:
        logger.info("🤖 Querying LLM...")
        
        response = client.chat.completions.create(
            model=AIMLAPI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"You solve quiz questions precisely. User email: {user_email} (length: {len(user_email)}). For commands, keep <your email> placeholder if shown. Return valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            #temperature=0.05,
            #max_tokens=2000
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
        
        # Don't post-process - keep answer as-is
        
        logger.info(f"✅ Answer: {str(answer)[:200]}")
        logger.info(f"💡 Reasoning: {solution.get('reasoning', '')[:150]}")
        
        return solution
        
    except Exception as e:
        logger.error(f"❌ LLM error: {e}")
        return {"error": str(e)}

def submit_answer(submit_url: str, email: str, secret: str, quiz_url: str, answer: Any) -> Dict[str, Any]:
    if not submit_url or not submit_url.startswith('http'):
        if submit_url and submit_url.startswith('/'):
            parsed = urlparse(quiz_url)
            submit_url = f"{parsed.scheme}://{parsed.netloc}{submit_url}"
        else:
            return {"error": "Invalid submit URL"}
    
    payload = {"email": email, "secret": secret, "url": quiz_url, "answer": answer}
    
    try:
        logger.info(f"📤 Submitting to: {submit_url}")
        logger.info(f"📤 Answer: {json.dumps(answer) if isinstance(answer, (dict, list)) else str(answer)[:200]}")
        
        resp = requests.post(submit_url, json=payload, timeout=20, headers={"Content-Type": "application/json"})
        logger.info(f"📥 Status: {resp.status_code}")
        
        try:
            result = resp.json()
            logger.info(f"📥 Response: {json.dumps(result)[:350]}")
        except:
            result = {"raw": resp.text[:500], "status": resp.status_code}
        
        return result
    except Exception as e:
        logger.error(f"❌ Submit error: {e}")
        return {"error": str(e)}

def process_quiz_chain(initial_url: str, email: str, secret: str, start_time: float, timeout: int = 170) -> List[Dict[str, Any]]:
    current_url = initial_url
    results = []
    iteration = 0
    max_iterations = 30
    
    while current_url and iteration < max_iterations:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        
        if remaining < 15:
            logger.warning(f"⚠️ Timeout approaching")
            results.append({"quiz_number": iteration, "url": current_url, "error": "Timeout", "status": "timeout"})
            break
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 QUIZ #{iteration}")
        logger.info(f"🔗 URL: {current_url}")
        logger.info(f"⏱️ Remaining: {remaining:.1f}s")
        logger.info(f"{'='*70}")
        
        try:
            quiz_page = extract_page_content(current_url, max_wait=5)
            
            if quiz_page.get("error"):
                results.append({"quiz_number": iteration, "url": current_url, "error": f"Extract failed", "status": "extraction_failed"})
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
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {executor.submit(download_file, link): link for link in download_links[:5]}
                    
                    for future in as_completed(futures, timeout=12):
                        link = futures[future]
                        try:
                            content = future.result(timeout=8)
                            if content:
                                parsed = smart_parse_file(content, link)
                                downloaded_files[link] = parsed
                        except Exception as e:
                            logger.error(f"❌ File error {link}: {e}")
            
            logger.info(f"✅ Files processed: {len(downloaded_files)}")
            
            # Solve
            solution = solve_with_llm(quiz_page, downloaded_files, current_url, email)
            
            if "error" in solution:
                results.append({"quiz_number": iteration, "url": current_url, "error": f"Solve error", "status": "solving_failed"})
                break
            
            answer = solution.get("answer")
            
            if answer is None or answer == "":
                results.append({"quiz_number": iteration, "url": current_url, "error": "No answer", "status": "no_answer"})
                break
            
            # Submit
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
            
            next_url = submission.get("url")
            
            if next_url:
                logger.info(f"➡️ Next: {next_url}")
                current_url = next_url
                time.sleep(0.3)
            else:
                logger.info("🏁 Quiz chain complete!")
                break
        
        except Exception as e:
            logger.error(f"❌ Exception in quiz #{iteration}: {e}")
            logger.error(traceback.format_exc())
            results.append({"quiz_number": iteration, "url": current_url, "error": str(e), "status": "exception"})
            break
    
    return results

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Dynamic Quiz Solver",
        "version": "7.0",
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
            secret=data["secret"],
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
    logger.info(f"🚀 Dynamic Quiz Solver v7.0")
    logger.info(f"   Port: {port}")
    logger.info(f"   Model: {AIMLAPI_MODEL}")
    logger.info(f"   LLM: {'✅' if client else '❌'}")
    logger.info(f"   Email: {EMAIL if EMAIL else '❌'}")
    logger.info(f"{'='*70}\n")
    
    app.run(host="0.0.0.0", port=port, debug=False)