#!/usr/bin/env python3
"""
app.py - LLM Quiz Solver (Production-Ready)
"""

import os
import json
import time
import requests
import re
import io
import traceback
from urllib.parse import urljoin, urlparse, unquote

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
from bs4 import BeautifulSoup
import pdfplumber

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
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

logger.info("=" * 60)
logger.info("🔧 Configuration Status:")
logger.info(f"   EMAIL: {'✅' if EMAIL else '❌'}")
logger.info(f"   SECRET: {'✅' if SECRET else '❌'}")
logger.info(f"   API_KEY: {'✅' if AIMLAPI_API_KEY else '❌'}")
logger.info(f"   MODEL: {AIMLAPI_MODEL}")
logger.info("=" * 60)

# Initialize OpenAI client
client = None
if AIMLAPI_API_KEY:
    try:
        client = OpenAI(api_key=AIMLAPI_API_KEY, base_url=AIMLAPI_BASE_URL)
        logger.info("✅ OpenAI client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
else:
    logger.error("❌ AIMLAPI_API_KEY not set!")


def setup_browser():
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
    driver.set_page_load_timeout(30)
    return driver


def extract_quiz_page(url):
    """Extract quiz with Selenium"""
    logger.info(f"🌐 Extracting: {url}")
    driver = setup_browser()
    
    try:
        driver.get(url)
        time.sleep(5)  # Wait for JS
        
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        page_source = driver.page_source
        page_text = driver.find_element(By.TAG_NAME, "body").text
        
        # Extract all links
        links = []
        try:
            for link in driver.find_elements(By.TAG_NAME, "a"):
                href = link.get_attribute("href")
                if href:
                    links.append(urljoin(url, href))
        except Exception as e:
            logger.warning(f"Link error: {e}")
        
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
            "text": "",
            "links": [],
            "url": url,
            "error": str(e)
        }
    finally:
        try:
            driver.quit()
        except:
            pass


def download_file(url):
    """Download file"""
    try:
        logger.info(f"📥 Downloading: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        logger.info(f"✅ Downloaded {len(resp.content)} bytes")
        return resp.content
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        return None


def parse_pdf(pdf_bytes):
    """Parse PDF"""
    try:
        logger.info("📄 Parsing PDF...")
        result = {"text_by_page": [], "tables_by_page": []}
        
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    result["text_by_page"].append({"page": page_num, "text": text})
                
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0:
                            try:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                result["tables_by_page"].append({
                                    "page": page_num,
                                    "table_index": table_idx,
                                    "dataframe": df.to_dict(),
                                    "raw": table
                                })
                            except:
                                pass
        
        logger.info(f"✅ Parsed: {len(result['text_by_page'])} pages, {len(result['tables_by_page'])} tables")
        return result
    except Exception as e:
        logger.error(f"❌ PDF error: {e}")
        return None


def parse_csv(csv_content):
    """Parse CSV"""
    try:
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_content))
        logger.info(f"✅ CSV: {len(df)} rows")
        return df.to_dict("records")
    except Exception as e:
        logger.error(f"❌ CSV error: {e}")
        return None


def extract_submit_url(page_content, quiz_url):
    """Extract submit URL - handle both absolute and relative"""
    if not page_content:
        return None
    
    # Pattern 1: Direct TDS submit URLs
    patterns = [
        r'(https://tds-llm-analysis\.s-anand\.net/submit[^\s"<>]*)',
        r'Post your answer to\s+(https?://[^\s<>"]+)',
        r'submit.*?to\s+(https?://[^\s<>"]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, page_content, flags=re.IGNORECASE)
        if matches:
            url = max(matches, key=len)  # Pick longest
            logger.info(f"✅ Found submit URL: {url}")
            return url
    
    # Pattern 2: Relative /submit
    if '/submit' in page_content.lower():
        base_url = 'https://tds-llm-analysis.s-anand.net/submit'
        logger.info(f"✅ Using base submit URL: {base_url}")
        return base_url
    
    logger.warning("⚠️ No submit URL found")
    return None


def solve_quiz_with_llm(quiz_data, email, secret, quiz_url):
    """Solve quiz with LLM"""
    if not client:
        return {"error": "LLM not initialized", "answer": None}
    
    # Download data files mentioned in links
    downloaded_data = {}
    for link in quiz_data.get("links", [])[:5]:
        try:
            # Skip submit URLs
            if 'submit' in link.lower() and 'data' not in link.lower():
                continue
            
            file_content = download_file(link)
            if file_content:
                # Try to parse as JSON first
                try:
                    parsed = json.loads(file_content.decode('utf-8'))
                    downloaded_data[link] = parsed
                    logger.info(f"✅ Parsed JSON from {link}")
                    continue
                except:
                    pass
                
                # Try CSV
                try:
                    parsed = parse_csv(file_content)
                    if parsed:
                        downloaded_data[link] = parsed
                        logger.info(f"✅ Parsed CSV from {link}")
                        continue
                except:
                    pass
                
                # Try PDF
                if file_content[:4] == b"%PDF":
                    parsed = parse_pdf(file_content)
                    if parsed:
                        downloaded_data[link] = parsed
                        logger.info(f"✅ Parsed PDF from {link}")
                        continue
                
                # Store as text
                try:
                    downloaded_data[link] = file_content.decode('utf-8')
                    logger.info(f"✅ Stored text from {link}")
                except:
                    pass
        except Exception as e:
            logger.error(f"File processing error {link}: {e}")
    
    # Create comprehensive prompt
    prompt = f"""You are solving a data analysis quiz. Read the question carefully and solve it.

QUESTION TEXT:
{quiz_data.get('text', '')[:3000]}

DOWNLOADED DATA:
{json.dumps(downloaded_data, indent=2)[:8000]}

INSTRUCTIONS:
1. Read what the question is asking for
2. Use the downloaded data to calculate the answer
3. Provide the exact answer requested

Respond ONLY with JSON (no markdown, no explanation):
{{
    "answer": <the_actual_answer>,
    "reasoning": "brief explanation"
}}

IMPORTANT:
- If question asks for a number, answer must be a number
- If question asks for text/string, answer must be text
- Be precise and exact"""

    try:
        logger.info("🤖 Sending to LLM...")
        response = client.chat.completions.create(
            model=AIMLAPI_MODEL,
            messages=[
                {"role": "system", "content": "You are a data analyst. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_completion_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        logger.info(f"🤖 Response: {response_text[:500]}")
        
        # Extract JSON
        try:
            # Try direct parse
            solution = json.loads(response_text)
        except:
            # Try to find JSON in text
            match = re.search(r'\{[^{}]*"answer"[^{}]*\}', response_text, re.DOTALL)
            if match:
                solution = json.loads(match.group())
            else:
                return {"error": "Could not parse JSON", "raw": response_text}
        
        logger.info(f"✅ Answer: {solution.get('answer')}")
        return solution
        
    except Exception as e:
        logger.error(f"❌ LLM error: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def submit_answer(submit_url, email, secret, quiz_url, answer):
    """Submit answer"""
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer
    }
    
    try:
        logger.info(f"📤 Submitting to: {submit_url}")
        logger.info(f"📤 Answer: {json.dumps(answer) if isinstance(answer, (dict, list)) else answer}")
        
        resp = requests.post(submit_url, json=payload, timeout=30)
        
        try:
            result = resp.json()
            logger.info(f"📥 Result: {result}")
        except:
            result = {"raw": resp.text, "status": resp.status_code}
        
        return result
    except Exception as e:
        logger.error(f"❌ Submit error: {e}")
        return {"error": str(e)}


def process_quiz_chain(initial_url, email, secret, start_time, timeout=170):
    """Process quiz chain"""
    current_url = initial_url
    results = []
    iteration = 0
    
    while current_url and (time.time() - start_time) < timeout:
        iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 QUIZ #{iteration}: {current_url}")
        logger.info(f"⏱️  Elapsed: {time.time() - start_time:.1f}s")
        logger.info(f"{'='*60}")
        
        try:
            # Extract page
            quiz_data = extract_quiz_page(current_url)
            if quiz_data.get("error"):
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": quiz_data["error"],
                    "status": "extraction_failed"
                })
                break
            
            # Find submit URL
            submit_url = extract_submit_url(
                quiz_data.get("html", "") + quiz_data.get("text", ""),
                current_url
            )
            
            # Solve
            solution = solve_quiz_with_llm(quiz_data, email, secret, current_url)
            
            if "error" in solution:
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": solution["error"],
                    "status": "solving_failed"
                })
                break
            
            answer = solution.get("answer")
            
            if not submit_url or answer is None:
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "Missing submit_url or answer",
                    "status": "incomplete"
                })
                break
            
            # Submit
            submission = submit_answer(submit_url, email, secret, current_url, answer)
            
            results.append({
                "quiz_number": iteration,
                "url": current_url,
                "solution": solution,
                "submission_result": submission,
                "status": "submitted",
                "correct": submission.get("correct", False)
            })
            
            # Next quiz?
            if submission.get("url"):
                current_url = submission["url"]
                logger.info(f"➡️  Next: {current_url}")
            else:
                logger.info("🏁 Complete!")
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


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "LLM Quiz Solver",
        "status": "running",
        "model": AIMLAPI_MODEL,
        "llm_ready": client is not None
    }), 200


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "llm_initialized": client is not None,
        "email_configured": bool(EMAIL),
        "secret_configured": bool(SECRET)
    }), 200


@app.route("/quiz", methods=["POST"])
def quiz_endpoint():
    start_time = time.time()
    
    if not request.is_json:
        return jsonify({"error": "Invalid JSON"}), 400
    
    data = request.get_json()
    
    if data.get("secret") != SECRET:
        return jsonify({"error": "Invalid secret"}), 403
    
    if not data.get("email") or not data.get("url"):
        return jsonify({"error": "Missing email or url"}), 400
    
    if not client:
        return jsonify({"error": "LLM not initialized"}), 500
    
    try:
        results = process_quiz_chain(
            data["url"],
            data["email"],
            data["secret"],
            start_time
        )
        
        return jsonify({
            "status": "completed",
            "results": results,
            "quizzes_attempted": len(results),
            "quizzes_correct": sum(1 for r in results if r.get("correct")),
            "time_taken": round(time.time() - start_time, 2)
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Fatal: {e}")
        return jsonify({
            "error": str(e),
            "time_taken": round(time.time() - start_time, 2)
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    logger.info(f"🚀 Starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)