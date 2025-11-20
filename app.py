#!/usr/bin/env python3
"""
app.py - LLM Quiz Solver (fixed & AIPipe-compatible)

Notes:
- Uses OpenAI Python client (OpenAI class) and chat.completions.create(...) so it's compatible with AIPipe-style endpoints.
- Ensure your environment/requirements contain: openai==1.12.0, selenium, flask, pdfplumber, pandas, etc.
- Configure environment variables:
    STUDENT_EMAIL, STUDENT_SECRET, AIMLAPI_API_KEY, AIMLAPI_BASE_URL (optional), AIMLAPI_MODEL (optional)
- You can override CHROME_BINARY and CHROMEDRIVER_PATH via env when running in containers.
"""

import os
import json
import time
import requests
import re
import io
import traceback
from urllib.parse import urljoin, urlparse

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

# Selenium chrome/chromedriver paths (allow override)
CHROME_BINARY = os.getenv("CHROME_BINARY", "/usr/bin/chromium")
CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")

logger.info("=" * 60)
logger.info("🔧 Configuration Status:")
logger.info(f"   EMAIL: {'✅ Set' if EMAIL else '❌ Not set'}")
logger.info(f"   SECRET: {'✅ Set' if SECRET else '❌ Not set'}")
logger.info(f"   API_KEY: {'✅ Set' if AIMLAPI_API_KEY else '❌ Not set'}")
logger.info(f"   BASE_URL: {AIMLAPI_BASE_URL}")
logger.info(f"   MODEL: {AIMLAPI_MODEL}")
logger.info(f"   CHROME_BINARY: {CHROME_BINARY}")
logger.info(f"   CHROMEDRIVER_PATH: {CHROMEDRIVER_PATH}")
logger.info("=" * 60)

# Initialize OpenAI client (OpenAI class; compatible with openai>=1.0 style wrappers/proxies)
client = None
if AIMLAPI_API_KEY:
    try:
        client = OpenAI(api_key=AIMLAPI_API_KEY, base_url=AIMLAPI_BASE_URL)
        logger.info("✅ OpenAI client initialized successfully")
    except Exception as e:
        client = None
        logger.error(f"❌ Failed to initialize OpenAI client: {e}")
else:
    logger.error("❌ AIMLAPI_API_KEY is not set! LLM features will NOT work!")

# -------------------------
# Browser / scraping utils
# -------------------------
def setup_browser():
    """Initialize headless Chrome browser with robust options and path fallbacks"""
    chrome_options = Options()
    # Use newer headless mode when available
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--single-process")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--remote-debugging-port=9222")

    # Apply binary if exists
    if CHROME_BINARY and os.path.exists(CHROME_BINARY):
        chrome_options.binary_location = CHROME_BINARY

    # Find service path
    service_path = CHROMEDRIVER_PATH
    if not os.path.exists(service_path):
        # try common alternatives
        for alt in ("/usr/bin/chromedriver", "/usr/local/bin/chromedriver"):
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
        time.sleep(2)  # let JS run a bit

        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        page_source = driver.page_source or ""
        try:
            page_text = driver.find_element(By.TAG_NAME, "body").text or ""
        except Exception:
            page_text = ""

        # Links
        links = []
        try:
            link_elements = driver.find_elements(By.TAG_NAME, "a")
            for link in link_elements:
                href = link.get_attribute("href")
                if href:
                    links.append(urljoin(url, href))
        except Exception as e:
            logger.warning(f"Link extraction error: {e}")

        # Tables HTML
        tables_html = []
        try:
            table_elements = driver.find_elements(By.TAG_NAME, "table")
            for table in table_elements:
                tables_html.append(table.get_attribute("outerHTML") or "")
        except Exception as e:
            logger.warning(f"Table extraction error: {e}")

        logger.info(f"✅ Extracted: {len(page_text)} chars, {len(links)} links, {len(tables_html)} tables")
        return {"html": page_source, "text": page_text, "tables_html": tables_html, "links": links, "url": url}

    except Exception as e:
        logger.error(f"❌ Extraction error: {e}")
        return {"html": "", "text": f"Error: {e}", "tables_html": [], "links": [], "url": url, "error": str(e)}
    finally:
        try:
            driver.quit()
        except Exception:
            pass


# -------------------------
# File parsing utils
# -------------------------
def download_file(url):
    try:
        logger.info(f"📥 Downloading: {url}")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        logger.info(f"✅ Downloaded {len(resp.content)} bytes")
        return resp.content
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        return None


def parse_pdf(pdf_bytes):
    try:
        logger.info("📄 Parsing PDF...")
        result = {"text_by_page": [], "tables_by_page": []}
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = None
                try:
                    text = page.extract_text()
                except Exception:
                    text = None
                if text:
                    result["text_by_page"].append({"page": page_num, "text": text})

                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = None
                if tables:
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0:
                            try:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                result["tables_by_page"].append({"page": page_num, "table_index": table_idx, "dataframe": df.to_dict(), "raw": table})
                            except Exception as e:
                                logger.warning(f"Table conversion error: {e}")
        logger.info(f"✅ Parsed PDF: {len(result['text_by_page'])} pages, {len(result['tables_by_page'])} tables")
        return result
    except Exception as e:
        logger.error(f"❌ PDF parsing error: {e}")
        return None


def parse_csv(csv_content):
    try:
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_content))
        logger.info(f"✅ Parsed CSV: {len(df)} rows, {len(df.columns)} columns")
        return df.to_dict("records")
    except Exception as e:
        logger.error(f"❌ CSV parsing error: {e}")
        return None


def parse_excel(excel_bytes):
    try:
        excel_file = pd.ExcelFile(io.BytesIO(excel_bytes))
        result = {}
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            result[sheet_name] = df.to_dict("records")
        logger.info(f"✅ Parsed Excel: {len(result)} sheets")
        return result
    except Exception as e:
        logger.error(f"❌ Excel parsing error: {e}")
        return None


# -------------------------
# Helpers: URL extraction & JSON extraction
# -------------------------
def extract_submit_url(page_content):
    """Find probable submit URL from page content"""
    if not page_content:
        return None

    patterns = [
        r'Post your answer to\s+(https?://[^\s<>"]+)',
        r'submit.*?to\s+(https?://[^\s<>"]+)',
        r'"submit_url":\s*"(https?://[^\s<>"]+)"',
        r'https?://[^\s<>"]+/submit[^\s<>"]*',
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
    Robust attempt to extract JSON object from text.
    """
    if not text:
        raise ValueError("No text to parse")

    # direct
    try:
        return json.loads(text)
    except Exception:
        pass

    # from first { to last }
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # regex non-greedy
    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

    raise ValueError("Could not extract JSON from text")


# -------------------------
# LLM integration
# -------------------------
def solve_quiz_with_llm(quiz_data, email, secret, quiz_url):
    """Send prompt to LLM and parse JSON answer"""
    if not client:
        logger.error("❌ LLM client not initialized")
        return {"error": "LLM client not initialized. Check AIMLAPI_API_KEY in secrets.", "answer": None}

    context = {
        "question": (quiz_data.get("text") or "")[:5000],
        "available_links": quiz_data.get("links", [])[:10],
        "tables_count": len(quiz_data.get("tables_html", [])),
        "downloaded_files": {}
    }

    # Auto-download first few files (if any)
    for link in quiz_data.get("links", [])[:5]:
        try:
            parsed = urlparse(link)
            path = parsed.path or ""
            ext = path.split(".")[-1].lower() if "." in path else ""
            if ext in ["pdf", "csv", "xlsx", "xls", "json", "txt"]:
                logger.info(f"📥 Auto-downloading: {link}")
                content = download_file(link)
                if content:
                    if ext == "pdf":
                        parsed_pdf = parse_pdf(content)
                        if parsed_pdf:
                            context["downloaded_files"][f"pdf_{link}"] = parsed_pdf
                    elif ext == "csv":
                        parsed_csv = parse_csv(content)
                        if parsed_csv:
                            context["downloaded_files"][f"csv_{link}"] = parsed_csv
                    elif ext in ["xlsx", "xls"]:
                        parsed_excel = parse_excel(content)
                        if parsed_excel:
                            context["downloaded_files"][f"excel_{link}"] = parsed_excel
                    elif ext == "json":
                        try:
                            parsed_json = json.loads(content.decode("utf-8"))
                            context["downloaded_files"][f"json_{link}"] = parsed_json
                        except Exception:
                            pass
            else:
                # unknown ext: try to download and sniff
                content = download_file(link)
                if content and content[:4] == b"%PDF":
                    parsed_pdf = parse_pdf(content)
                    if parsed_pdf:
                        context["downloaded_files"][f"pdf_{link}"] = parsed_pdf
        except Exception as e:
            logger.error(f"File processing error for {link}: {e}")
            continue

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
        # Use chat.completions.create (AIPipe-compatible)
        response = client.chat.completions.create(
            model=AIMLAPI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise data analyst. Always respond with valid JSON containing the answer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        # Extract content robustly
        response_text = ""
        try:
            # expected: response.choices[0].message.content
            response_text = response.choices[0].message.content
        except Exception:
            try:
                response_text = response.choices[0].text
            except Exception:
                response_text = str(response)

        logger.info(f"🤖 LLM Response (truncated): {response_text[:500]}...")
        try:
            solution = _extract_json_from_text(response_text)
            logger.info(f"✅ Answer parsed: {solution.get('answer') if isinstance(solution, dict) else 'N/A'}")
            return solution
        except Exception as e:
            logger.error(f"❌ Could not parse LLM response as JSON: {e}")
            return {"error": "Could not parse LLM response as JSON", "raw_response": response_text}

    except Exception as e:
        logger.error(f"❌ LLM error: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}


# -------------------------
# Submission & flow
# -------------------------
def submit_answer(submit_url, email, secret, quiz_url, answer):
    payload = {"email": email, "secret": secret, "url": quiz_url, "answer": answer}
    try:
        logger.info(f"📤 Submitting to: {submit_url}")
        logger.info(f"📤 Answer: {json.dumps(answer) if isinstance(answer, (dict, list)) else answer}")
        resp = requests.post(submit_url, json=payload, timeout=30)
        try:
            result = resp.json()
            logger.info(f"📥 Response: {result}")
        except Exception:
            result = {"raw_response": resp.text, "status_code": resp.status_code}
            logger.info(f"📥 Raw response: {resp.text}")
        return result
    except Exception as e:
        logger.error(f"❌ Submission error: {e}")
        return {"error": str(e)}


def process_quiz_chain(initial_url, email, secret, start_time, timeout=170):
    current_url = initial_url
    results = []
    iteration = 0

    while current_url and (time.time() - start_time) < timeout:
        iteration += 1
        logger.info("\n" + "=" * 60)
        logger.info(f"🎯 QUIZ #{iteration}: {current_url}")
        logger.info(f"⏱️  Time elapsed: {time.time() - start_time:.1f}s / {timeout}s")
        logger.info("=" * 60 + "\n")

        try:
            quiz_data = extract_quiz_page(current_url)

            if quiz_data.get("error"):
                results.append({"quiz_number": iteration, "url": current_url, "error": quiz_data["error"], "status": "extraction_failed"})
                break

            submit_url = extract_submit_url((quiz_data.get("html") or "") + (quiz_data.get("text") or ""))

            solution = solve_quiz_with_llm(quiz_data, email, secret, current_url)

            if isinstance(solution, dict) and solution.get("error"):
                results.append({"quiz_number": iteration, "url": current_url, "error": solution["error"], "status": "solving_failed", "solution": solution})
                break

            answer = None
            if isinstance(solution, dict):
                answer = solution.get("answer")
            else:
                answer = solution

            if not submit_url or answer is None:
                results.append({"quiz_number": iteration, "url": current_url, "error": "Missing submit_url or answer", "solution": solution, "status": "incomplete"})
                break

            submission_result = submit_answer(submit_url, email, secret, current_url, answer)

            results.append({"quiz_number": iteration, "url": current_url, "solution": solution, "submission_result": submission_result, "status": "submitted", "correct": submission_result.get("correct", False)})

            if submission_result.get("url"):
                current_url = submission_result["url"]
                logger.info(f"➡️  Next quiz: {current_url}")
            else:
                logger.info("🏁 Quiz chain complete!")
                break

        except Exception as e:
            logger.error(f"❌ Exception in quiz #{iteration}: {e}")
            logger.error(traceback.format_exc())
            results.append({"quiz_number": iteration, "url": current_url, "error": str(e), "traceback": traceback.format_exc(), "status": "exception"})
            break

    return results


# -------------------------
# Flask endpoints
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
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
                "✅ Comprehensive error handling",
            ],
            "endpoints": {"GET /": "This page", "GET /health": "Health check", "POST /quiz": "Submit quiz task"},
        }
    ), 200


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model": AIMLAPI_MODEL, "base_url": AIMLAPI_BASE_URL, "llm_initialized": client is not None, "email_configured": bool(EMAIL), "secret_configured": bool(SECRET), "api_key_configured": bool(AIMLAPI_API_KEY)}), 200


@app.route("/quiz", methods=["POST"])
def quiz_endpoint():
    start_time = time.time()

    if not request.is_json:
        logger.warning("❌ Received non-JSON request")
        return jsonify({"error": "Invalid JSON"}), 400

    data = request.get_json()
    logger.info("\n" + "=" * 60)
    logger.info("📨 NEW QUIZ REQUEST")
    logger.info(f"📧 Email: {data.get('email')}")
    logger.info(f"🔗 URL: {data.get('url')}")
    logger.info("=" * 60 + "\n")

    if data.get("secret") != SECRET:
        logger.warning("❌ Invalid secret")
        return jsonify({"error": "Invalid secret"}), 403

    if not data.get("email") or not data.get("url"):
        logger.warning("❌ Missing required fields")
        return jsonify({"error": "Missing required fields (email or url)"}), 400

    if not client:
        logger.error("❌ LLM client not initialized")
        return jsonify({"error": "LLM client not initialized. Check AIMLAPI_API_KEY in Space secrets.", "time_taken": round(time.time() - start_time, 2)}), 500

    try:
        results = process_quiz_chain(data["url"], data["email"], data["secret"], start_time)
        response = {"status": "completed", "results": results, "quizzes_attempted": len(results), "quizzes_correct": sum(1 for r in results if r.get("correct")), "time_taken": round(time.time() - start_time, 2)}

        logger.info("\n" + "=" * 60)
        logger.info("✅ QUIZ CHAIN COMPLETED")
        logger.info(f"📊 Attempted: {response['quizzes_attempted']}")
        logger.info(f"✅ Correct: {response['quizzes_correct']}")
        logger.info(f"⏱️  Time: {response['time_taken']}s")
        logger.info("=" * 60 + "\n")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc(), "time_taken": round(time.time() - start_time, 2)}), 500


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    logger.info(f"🚀 Starting LLM Quiz Solver on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
