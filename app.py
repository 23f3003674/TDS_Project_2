import os
import json
import time
import requests
import re
import base64
import io
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
from urllib.parse import urljoin, urlparse
import traceback

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration with validation
EMAIL = os.getenv("STUDENT_EMAIL")
SECRET = os.getenv("STUDENT_SECRET")
AIMLAPI_BASE_URL = os.getenv("AIMLAPI_BASE_URL", "https://aipipe.org/openai/v1")
AIMLAPI_API_KEY = os.getenv("AIMLAPI_API_KEY")
AIMLAPI_MODEL = os.getenv("AIMLAPI_MODEL", "gpt-5-nano")

# Allow overriding chromium/chromedriver path via env for portability
CHROME_BINARY = os.getenv("CHROME_BINARY", "/usr/bin/chromium")
CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")

# Log configuration status
logger.info("="*60)
logger.info("🔧 Configuration Status:")
logger.info(f"   EMAIL: {'✅ Set' if EMAIL else '❌ Not set'}")
logger.info(f"   SECRET: {'✅ Set' if SECRET else '❌ Not set'}")
logger.info(f"   API_KEY: {'✅ Set' if AIMLAPI_API_KEY else '❌ Not set'}")
logger.info(f"   BASE_URL: {AIMLAPI_BASE_URL}")
logger.info(f"   MODEL: {AIMLAPI_MODEL}")
logger.info(f"   CHROME_BINARY: {CHROME_BINARY}")
logger.info(f"   CHROMEDRIVER_PATH: {CHROMEDRIVER_PATH}")
logger.info("="*60)

# Initialize OpenAI client with error handling
client = None
if AIMLAPI_API_KEY:
    try:
        client = OpenAI(
            api_key=AIMLAPI_API_KEY,
            base_url=AIMLAPI_BASE_URL
        )
        logger.info("✅ OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize OpenAI client: {e}")
        client = None
else:
    logger.error("❌ AIMLAPI_API_KEY is not set! LLM features will NOT work!")

def setup_browser():
    """Initialize headless Chrome browser with robust options and path fallbacks"""
    chrome_options = Options()
    # Headless + robustness flags
    chrome_options.add_argument("--headless=new")  # newer headless mode where available
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--single-process")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--remote-debugging-port=9222")

    # Use environment-provided binary path if available
    if CHROME_BINARY and os.path.exists(CHROME_BINARY):
        chrome_options.binary_location = CHROME_BINARY

    # Setup service with provided chromedriver path
    service_path = CHROMEDRIVER_PATH
    if not os.path.exists(service_path):
        # try common alternatives
        alternatives = ["/usr/bin/chromedriver", "/usr/local/bin/chromedriver"]
        for alt in alternatives:
            if os.path.exists(alt):
                service_path = alt
                break

    service = Service(executable_path=service_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(30)
    return driver

def extract_quiz_page(url):
    """Extract quiz content using Selenium with JavaScript rendering"""
    logger.info(f"🌐 Extracting quiz from: {url}")
    driver = setup_browser()

    try:
        driver.get(url)
        # small explicit wait to allow JS to run and content to stabilise
        time.sleep(3)

        # Wait for body to be present
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Get page content after JavaScript execution
        page_source = driver.page_source or ""
        try:
            page_text = driver.find_element(By.TAG_NAME, "body").text or ""
        except Exception:
            page_text = ""

        # Extract all links (for file downloads)
        links = []
        try:
            link_elements = driver.find_elements(By.TAG_NAME, "a")
            for link in link_elements:
                href = link.get_attribute('href')
                if href:
                    # Make absolute URL
                    absolute_url = urljoin(url, href)
                    links.append(absolute_url)
        except Exception as e:
            logger.warning(f"Link extraction error: {e}")

        # Extract tables
        tables_html = []
        try:
            table_elements = driver.find_elements(By.TAG_NAME, "table")
            for table in table_elements:
                tables_html.append(table.get_attribute('outerHTML') or "")
        except Exception as e:
            logger.warning(f"Table extraction error: {e}")

        logger.info(f"✅ Extracted: {len(page_text)} chars, {len(links)} links, {len(tables_html)} tables")

        return {
            "html": page_source,
            "text": page_text,
            "tables_html": tables_html,
            "links": links,
            "url": url
        }

    except Exception as e:
        logger.error(f"❌ Extraction error: {str(e)}")
        return {
            "html": "",
            "text": f"Error: {str(e)}",
            "tables_html": [],
            "links": [],
            "url": url,
            "error": str(e)
        }
    finally:
        try:
            driver.quit()
        except Exception:
            pass

def download_file(url):
    """Download file from URL"""
    try:
        logger.info(f"📥 Downloading: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        logger.info(f"✅ Downloaded {len(response.content)} bytes")
        return response.content
    except Exception as e:
        logger.error(f"❌ Download error: {str(e)}")
        return None

def parse_pdf(pdf_bytes):
    """Parse PDF and extract text and tables"""
    try:
        logger.info("📄 Parsing PDF...")
        result = {
            "text_by_page": [],
            "tables_by_page": []
        }

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                try:
                    text = page.extract_text()
                except Exception:
                    text = None
                if text:
                    result["text_by_page"].append({
                        "page": page_num,
                        "text": text
                    })

                # Extract tables
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = None
                if tables:
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0:
                            try:
                                # Convert to DataFrame
                                df = pd.DataFrame(table[1:], columns=table[0])
                                result["tables_by_page"].append({
                                    "page": page_num,
                                    "table_index": table_idx,
                                    "dataframe": df.to_dict(),
                                    "raw": table
                                })
                            except Exception as e:
                                logger.warning(f"Table conversion error: {e}")

        logger.info(f"✅ Parsed PDF: {len(result['text_by_page'])} pages, {len(result['tables_by_page'])} tables")
        return result

    except Exception as e:
        logger.error(f"❌ PDF parsing error: {str(e)}")
        return None

def parse_csv(csv_content):
    """Parse CSV content"""
    try:
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))
        logger.info(f"✅ Parsed CSV: {len(df)} rows, {len(df.columns)} columns")
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"❌ CSV parsing error: {str(e)}")
        return None

def parse_excel(excel_bytes):
    """Parse Excel file"""
    try:
        excel_file = pd.ExcelFile(io.BytesIO(excel_bytes))
        result = {}
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            result[sheet_name] = df.to_dict('records')
        logger.info(f"✅ Parsed Excel: {len(result)} sheets")
        return result
    except Exception as e:
        logger.error(f"❌ Excel parsing error: {str(e)}")
        return None

def extract_submit_url(page_content):
    """Extract submit URL from page content"""
    if not page_content:
        return None

    # Look for submit URLs in various formats
    patterns = [
        r'Post your answer to\s+(https?://[^\s<>"]+)',
        r'submit.*?to\s+(https?://[^\s<>"]+)',
        r'"submit_url":\s*"(https?://[^\s<>"]+)"',
        r'https?://[^\s<>"]+/submit[^\s<>"]*'
    ]

    for pattern in patterns:
        match = re.search(pattern, page_content, re.IGNORECASE)
        if match:
            url = match.group(1) if match.lastindex else match.group(0)
            logger.info(f"✅ Found submit URL: {url}")
            return url

    logger.warning("⚠️ No submit URL found in page content")
    return None

def _extract_json_from_text(text):
    """
    Try to extract the most-likely JSON object from a text blob.
    Strategy:
      1. Try direct json.loads(text)
      2. If fails, find first '{' and last '}' and attempt loads of that slice
      3. As a fallback, use regex to find the first {...} block
    Returns parsed object or raises ValueError.
    """
    if not text:
        raise ValueError("No text to parse")

    # 1. direct
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. slice from first { to last }
    first = text.find('{')
    last = text.rfind('}')
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3. regex to match a simple {...} block (non-greedy)
    m = re.search(r'\{[\s\S]*?\}', text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

    raise ValueError("Could not extract JSON from text")

def solve_quiz_with_llm(quiz_data, email, secret, quiz_url):
    """Use LLM to understand and solve the quiz"""

    # Check if client is initialized
    if not client:
        logger.error("❌ LLM client not initialized")
        return {
            "error": "LLM client not initialized. Check AIMLAPI_API_KEY in secrets.",
            "answer": None
        }

    # Prepare context for LLM
    context = {
        "question": (quiz_data.get("text") or "")[:5000],
        "available_links": quiz_data.get("links", [])[:10],
        "tables_count": len(quiz_data.get("tables_html", [])),
        "downloaded_files": {}
    }

    # Auto-download files mentioned in links
    for link in quiz_data.get("links", [])[:5]:  # Limit to 5 files
        try:
            parsed_url = urlparse(link)
            path = parsed_url.path or ""
            file_ext = path.split('.')[-1].lower() if '.' in path else ''
            # Fall back to content-type detection after download if needed
            if file_ext in ['pdf', 'csv', 'xlsx', 'xls', 'json', 'txt']:
                logger.info(f"📥 Auto-downloading: {link}")
                file_content = download_file(link)

                if file_content:
                    if file_ext == 'pdf':
                        parsed = parse_pdf(file_content)
                        if parsed:
                            context["downloaded_files"][f"pdf_{link}"] = parsed
                    elif file_ext == 'csv':
                        parsed = parse_csv(file_content)
                        if parsed:
                            context["downloaded_files"][f"csv_{link}"] = parsed
                    elif file_ext in ['xlsx', 'xls']:
                        parsed = parse_excel(file_content)
                        if parsed:
                            context["downloaded_files"][f"excel_{link}"] = parsed
                    elif file_ext == 'json':
                        try:
                            parsed = json.loads(file_content.decode('utf-8'))
                        except Exception:
                            parsed = None
                        if parsed:
                            context["downloaded_files"][f"json_{link}"] = parsed
            else:
                # If no obvious extension, attempt to download and sniff content-type
                logger.info(f"📥 Auto-downloading (unknown ext): {link}")
                file_content = download_file(link)
                if file_content:
                    # simple heuristic: check for PDF header
                    if file_content[:4] == b'%PDF':
                        parsed = parse_pdf(file_content)
                        if parsed:
                            context["downloaded_files"][f"pdf_{link}"] = parsed
        except Exception as e:
            logger.error(f"File processing error for {link}: {e}")
            continue

    # Create prompt for LLM (trim downloaded_files details to avoid huge prompt)
    downloaded_summary = list(context["downloaded_files"].keys())
    prompt = f"""You are a data analyst solving a quiz. Analyze the question and data, then provide the answer.

QUESTION:
{context['question']}

AVAILABLE DATA:
- Links found: {len(context['available_links'])}
- Downloaded files: {downloaded_summary}

TASK:
1. Read the question carefully
2. Identify what calculation/analysis is needed
3. Use the downloaded data to solve it (if any)
4. Provide the exact answer in the required format

Respond in JSON only (no surrounding explanation). EXACT format:
{{
    "answer": <the_actual_answer>,
    "reasoning": "<step-by-step explanation>",
    "calculation": "<show your work>",
    "answer_type": "number/string/boolean/json"
}}

IMPORTANT:
- Be precise with numbers (no rounding unless specified)
- If the answer is a sum, provide just the number in the 'answer' field
- If it's text, provide the exact text in the 'answer' field
"""

    try:
        logger.info("🤖 Sending to LLM...")

        # use max_tokens (supported) instead of removed max_completion_tokens
        # keep temperature low to encourage deterministic answers
        response = client.chat.completions.create(
            model=AIMLAPI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise data analyst. Always respond with valid JSON containing the answer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )

        # Newer SDK returns choices with .message.content in many wrappers; be defensive
        response_text = ""
        try:
            # common shapes: response.choices[0].message.content
            response_text = response.choices[0].message.content
        except Exception:
            try:
                # alternate: response.choices[0].text
                response_text = response.choices[0].text
            except Exception:
                # fallback: str(response)
                response_text = str(response)

        logger.info(f"🤖 LLM Response (truncated): {response_text[:500]}...")

        # Extract JSON robustly
        try:
            solution = _extract_json_from_text(response_text)
            logger.info(f"✅ Answer parsed: {solution.get('answer') if isinstance(solution, dict) else 'N/A'}")
            return solution
        except Exception as e:
            logger.error(f"❌ Could not parse LLM response as JSON: {e}")
            return {
                "error": "Could not parse LLM response as JSON",
                "raw_response": response_text
            }

    except Exception as e:
        logger.error(f"❌ LLM error: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def submit_answer(submit_url, email, secret, quiz_url, answer):
    """Submit answer to the endpoint"""
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer
    }

    try:
        logger.info(f"📤 Submitting to: {submit_url}")
        logger.info(f"📤 Answer: {json.dumps(answer) if isinstance(answer, (dict, list)) else answer}")

        response = requests.post(submit_url, json=payload, timeout=30)

        try:
            result = response.json()
            logger.info(f"📥 Response: {result}")
        except Exception:
            result = {
                "raw_response": response.text,
                "status_code": response.status_code
            }
            logger.info(f"📥 Raw response: {response.text}")

        return result

    except Exception as e:
        logger.error(f"❌ Submission error: {str(e)}")
        return {"error": str(e)}

def process_quiz_chain(initial_url, email, secret, start_time, timeout=170):
    """Process a chain of quiz questions"""
    current_url = initial_url
    results = []
    iteration = 0

    while current_url and (time.time() - start_time) < timeout:
        iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 QUIZ #{iteration}: {current_url}")
        logger.info(f"⏱️  Time elapsed: {time.time() - start_time:.1f}s / {timeout}s")
        logger.info(f"{'='*60}\n")

        try:
            # Step 1: Extract quiz page
            quiz_data = extract_quiz_page(current_url)

            if quiz_data.get('error'):
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": quiz_data['error'],
                    "status": "extraction_failed"
                })
                break

            # Step 2: Extract submit URL from page
            submit_url = extract_submit_url((quiz_data.get('html') or "") + (quiz_data.get('text') or ""))

            # Step 3: Solve with LLM
            solution = solve_quiz_with_llm(quiz_data, email, secret, current_url)

            if isinstance(solution, dict) and solution.get("error"):
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": solution["error"],
                    "status": "solving_failed",
                    "solution": solution
                })
                break

            answer = None
            if isinstance(solution, dict):
                answer = solution.get("answer")
            else:
                # if solution is non-dict, keep it raw
                answer = solution

            if not submit_url or answer is None:
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "Missing submit_url or answer",
                    "solution": solution,
                    "status": "incomplete"
                })
                break

            # Step 4: Submit answer
            submission_result = submit_answer(submit_url, email, secret, current_url, answer)

            results.append({
                "quiz_number": iteration,
                "url": current_url,
                "solution": solution,
                "submission_result": submission_result,
                "status": "submitted",
                "correct": submission_result.get("correct", False)
            })

            # Step 5: Check for next URL
            if submission_result.get("url"):
                current_url = submission_result["url"]
                logger.info(f"➡️  Next quiz: {current_url}")
            else:
                logger.info("🏁 Quiz chain complete!")
                break

        except Exception as e:
            logger.error(f"❌ Exception in quiz #{iteration}: {str(e)}")
            logger.error(traceback.format_exc())
            results.append({
                "quiz_number": iteration,
                "url": current_url,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "exception"
            })
            break

    return results

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "service": "LLM Quiz Solver",
        "version": "2.0-production",
        "status": "running",
        "model": AIMLAPI_MODEL,
        "llm_ready": client is not None,
        "features": [
            "✅ JavaScript rendering (Selenium)",
            "✅ Automatic file downloads (PDF/CSV/Excel)",
            "✅ PDF table extraction (pdfplumber)",
            "✅ Data analysis (pandas/numpy)",
            "✅ Multi-quiz chain handling",
            "✅ 3-minute timeout management",
            "✅ Auto submit URL extraction",
            "✅ Comprehensive error handling"
        ],
        "endpoints": {
            "GET /": "This page",
            "GET /health": "Health check",
            "POST /quiz": "Submit quiz task"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": AIMLAPI_MODEL,
        "base_url": AIMLAPI_BASE_URL,
        "llm_initialized": client is not None,
        "email_configured": bool(EMAIL),
        "secret_configured": bool(SECRET),
        "api_key_configured": bool(AIMLAPI_API_KEY)
    }), 200

@app.route('/quiz', methods=['POST'])
def quiz_endpoint():
    """Main quiz endpoint - receives quiz tasks"""
    start_time = time.time()

    # Validate JSON
    if not request.is_json:
        logger.warning("❌ Received non-JSON request")
        return jsonify({"error": "Invalid JSON"}), 400

    data = request.get_json()
    logger.info(f"\n{'='*60}")
    logger.info(f"📨 NEW QUIZ REQUEST")
    logger.info(f"📧 Email: {data.get('email')}")
    logger.info(f"🔗 URL: {data.get('url')}")
    logger.info(f"{'='*60}\n")

    # Validate secret
    if data.get("secret") != SECRET:
        logger.warning(f"❌ Invalid secret")
        return jsonify({"error": "Invalid secret"}), 403

    # Validate required fields
    if not data.get("email") or not data.get("url"):
        logger.warning("❌ Missing required fields")
        return jsonify({"error": "Missing required fields (email or url)"}), 400

    # Check if LLM client is ready
    if not client:
        logger.error("❌ LLM client not initialized")
        return jsonify({
            "error": "LLM client not initialized. Check AIMLAPI_API_KEY in Space secrets.",
            "time_taken": round(time.time() - start_time, 2)
        }), 500

    # Process quiz
    try:
        results = process_quiz_chain(
            data["url"],
            data["email"],
            data["secret"],
            start_time
        )

        response = {
            "status": "completed",
            "results": results,
            "quizzes_attempted": len(results),
            "quizzes_correct": sum(1 for r in results if r.get("correct")),
            "time_taken": round(time.time() - start_time, 2)
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"✅ QUIZ CHAIN COMPLETED")
        logger.info(f"📊 Attempted: {response['quizzes_attempted']}")
        logger.info(f"✅ Correct: {response['quizzes_correct']}")
        logger.info(f"⏱️  Time: {response['time_taken']}s")
        logger.info(f"{'='*60}\n")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "time_taken": round(time.time() - start_time, 2)
        }), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 7860))
    logger.info(f"🚀 Starting LLM Quiz Solver on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
