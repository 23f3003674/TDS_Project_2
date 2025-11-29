#!/usr/bin/env python3
"""
Fully Dynamic Quiz Solver - ENHANCED VERSION
Handles any quiz type with improved data access and reasoning
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

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
EMAIL = os.getenv("STUDENT_EMAIL")
SECRET = os.getenv("STUDENT_SECRET")
AIMLAPI_BASE_URL = os.getenv("AIMLAPI_BASE_URL", "https://aipipe.org/openai/v1")
AIMLAPI_API_KEY = os.getenv("AIMLAPI_API_KEY")
AIMLAPI_MODEL = os.getenv("AIMLAPI_MODEL", "gpt-5-mini")

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


def setup_browser():
    """Initialize headless Chrome"""
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
    driver.set_page_load_timeout(30)
    return driver


def extract_page_content(url, max_wait=7):
    """Extract content from any webpage - FULLY DYNAMIC"""
    logger.info(f"🌐 Extracting: {url}")
    driver = setup_browser()
    
    try:
        driver.get(url)
        
        # Wait for body
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Progressive waiting - check if content stabilizes
        previous_text = ""
        for i in range(max_wait):
            time.sleep(1)
            current_text = driver.find_element(By.TAG_NAME, "body").text
            
            if i > 2 and current_text == previous_text and len(current_text) > 50:
                logger.info(f"✅ Content stabilized after {i}s")
                break
            
            previous_text = current_text
        
        time.sleep(1)
        
        page_source = driver.page_source
        page_text = driver.find_element(By.TAG_NAME, "body").text
        
        try:
            js_text = driver.execute_script("return document.body.innerText;")
            if js_text and len(js_text) > len(page_text):
                page_text = js_text
        except:
            pass
        
        links = []
        for link in driver.find_elements(By.TAG_NAME, "a"):
            href = link.get_attribute("href")
            if href and not href.startswith("javascript:"):
                absolute_url = urljoin(url, href)
                if absolute_url not in links:
                    links.append(absolute_url)
        
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


def download_file(url, timeout=30):
    """Download any file"""
    try:
        logger.info(f"📥 Downloading: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        logger.info(f"✅ Downloaded {len(resp.content)} bytes")
        return resp.content
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        return None


def smart_parse_file(file_content, url):
    """Intelligently parse any file type"""
    result = {"type": "unknown", "content": None, "error": None}
    
    try:
        if file_content[:4] == b"%PDF":
            result["type"] = "pdf"
            result["content"] = parse_pdf(file_content)
        
        elif url.lower().endswith('.csv') or b',' in file_content[:500]:
            result["type"] = "csv"
            result["content"] = parse_csv(file_content)
        
        elif url.lower().endswith(('.xlsx', '.xls')):
            result["type"] = "excel"
            result["content"] = parse_excel(file_content)
        
        elif url.lower().endswith('.json') or file_content[:1] in [b'{', b'[']:
            result["type"] = "json"
            try:
                result["content"] = json.loads(file_content.decode('utf-8'))
            except:
                result["error"] = "Invalid JSON"
        
        else:
            result["type"] = "text"
            try:
                result["content"] = file_content.decode('utf-8')
            except:
                result["error"] = "Could not decode as text"
        
        logger.info(f"✅ Parsed as: {result['type']}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Parse error: {e}")
        result["error"] = str(e)
        return result


def parse_pdf(pdf_bytes):
    """Parse PDF - extract ALL text and tables"""
    try:
        result = {
            "pages": [],
            "all_text": "",
            "all_tables": []
        }
        
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
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
                                df = pd.DataFrame(table[1:], columns=table[0])
                                table_info = {
                                    "index": table_idx,
                                    "data": df.to_dict('records'),
                                    "shape": df.shape
                                }
                                page_data["tables"].append(table_info)
                                result["all_tables"].append({
                                    "page": page_num,
                                    **table_info
                                })
                            except:
                                pass
                
                result["pages"].append(page_data)
        
        logger.info(f"✅ PDF: {len(result['pages'])} pages, {len(result['all_tables'])} tables")
        return result
    except Exception as e:
        logger.error(f"❌ PDF error: {e}")
        return {"error": str(e)}


def parse_csv(csv_content):
    """Parse CSV - COMPLETE analysis with ALL data for accurate calculations"""
    try:
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode("utf-8")
        
        df = pd.read_csv(io.StringIO(csv_content))
        df.columns = [str(col).strip() for col in df.columns]
        
        # Comprehensive column analysis
        column_analysis = {}
        for col in df.columns:
            col_data = df[col]
            analysis = {
                "dtype": str(col_data.dtype),
                "non_null_count": int(col_data.notna().sum()),
                "null_count": int(col_data.isna().sum())
            }
            
            if pd.api.types.is_numeric_dtype(col_data):
                analysis.update({
                    "sum": float(col_data.sum()),
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "std": float(col_data.std()) if len(col_data) > 1 else 0,
                    "variance": float(col_data.var()) if len(col_data) > 1 else 0,
                    "q25": float(col_data.quantile(0.25)),
                    "q75": float(col_data.quantile(0.75)),
                    "count_above_mean": int((col_data > col_data.mean()).sum()),
                    "count_below_mean": int((col_data < col_data.mean()).sum())
                })
                logger.info(f"  '{col}': sum={analysis['sum']}, mean={analysis['mean']:.2f}, median={analysis['median']}, std={analysis['std']:.2f}")
            else:
                unique_vals = col_data.unique()
                analysis["unique_count"] = len(unique_vals)
                analysis["sample_values"] = unique_vals[:10].tolist()
                analysis["value_counts"] = col_data.value_counts().head(10).to_dict()
            
            column_analysis[col] = analysis
        
        # CRITICAL: Include ALL data for precise calculations
        result = {
            "shape": df.shape,
            "row_count": len(df),
            "columns": list(df.columns),
            "column_analysis": column_analysis,
            "first_10_rows": df.head(10).to_dict('records'),
            "last_10_rows": df.tail(10).to_dict('records'),
            # Include ALL data (up to 10000 rows for filtering/calculations)
            "all_data": df.head(10000).to_dict('records')
        }
        
        logger.info(f"✅ CSV: {df.shape[0]} rows × {df.shape[1]} columns")
        return result
    except Exception as e:
        logger.error(f"❌ CSV error: {e}")
        return {"error": str(e)}


def parse_excel(excel_bytes):
    """Parse Excel files"""
    try:
        dfs = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)
        
        result = {"sheets": {}}
        for sheet_name, df in dfs.items():
            csv_str = df.to_csv(index=False)
            result["sheets"][sheet_name] = parse_csv(csv_str.encode())
        
        logger.info(f"✅ Excel: {len(result['sheets'])} sheets")
        return result
    except Exception as e:
        logger.error(f"❌ Excel error: {e}")
        return {"error": str(e)}


def extract_submit_url(content, base_url):
    """Dynamically extract submit URL"""
    if not content:
        return None
    
    url_patterns = [
        r'(https?://[^\s<>"\']+/submit[^\s<>"\']*)',
        r'POST[^\n]*?(https?://[^\s<>"\']+)',
        r'(?:POST|post|Submit|submit)[^\n]*?(\s+/submit[^\s<>"\']*)',
        r'href=["\']([^"\']*submit[^"\']*)["\']',
        r'to\s+([/\w-]*submit[^\s<>"\']*)',
    ]
    
    found_urls = []
    for pattern in url_patterns:
        matches = re.findall(pattern, content, flags=re.IGNORECASE)
        clean_matches = [m.strip() for m in matches if m and m.strip()]
        found_urls.extend(clean_matches)
    
    logger.info(f"🔍 Found URL candidates: {found_urls}")
    
    submit_urls = []
    for url in found_urls:
        if 'submit' in url.lower():
            if url.startswith('/'):
                parsed_base = urlparse(base_url)
                absolute_url = f"{parsed_base.scheme}://{parsed_base.netloc}{url}"
                submit_urls.append(absolute_url)
                logger.info(f"🔗 Converted relative URL: {url} → {absolute_url}")
            elif url.startswith('http'):
                submit_urls.append(url)
                logger.info(f"🔗 Found absolute URL: {url}")
    
    if submit_urls:
        submit_url = max(submit_urls, key=len)
        logger.info(f"✅ Selected submit URL: {submit_url}")
        return submit_url
    
    if 'submit' in content.lower():
        parsed = urlparse(base_url)
        submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
        logger.info(f"⚠️ Fallback constructed submit URL: {submit_url}")
        return submit_url
    
    logger.warning("❌ No submit URL found")
    return None


def solve_with_llm(quiz_page, downloaded_files, quiz_url):
    """
    Solve quiz with LLM - ENHANCED with better prompting and data access
    """
    if not client:
        return {"error": "LLM not initialized"}
    
    page_text = quiz_page.get("text", "")
    
    logger.info(f"📄 Analyzing: {len(page_text)} chars of question")
    logger.info(f"📄 With {len(downloaded_files)} data files")
    
    # Enhanced prompt with explicit operation guidance
    prompt = f"""You are solving a data analysis quiz. Read the question VERY CAREFULLY and perform the EXACT operation requested.

QUESTION PAGE:
{page_text[:4000]}

AVAILABLE DATA:
{json.dumps(downloaded_files, indent=2, default=str)[:20000]}

CRITICAL INSTRUCTIONS - READ CAREFULLY:

1. UNDERSTAND THE OPERATION REQUESTED:
   - "Count how many..." → COUNT operation (number of items matching condition)
   - "Sum of..." → SUM operation (add up all values)
   - "What is the sum of numbers greater than X" → Filter THEN sum
   - "Average/Mean of..." → MEAN calculation
   - "Find the..." → EXTRACT specific value

2. FOR FILTERING + AGGREGATION QUESTIONS:
   Example: "What is the sum of numbers greater than 59873?"
   Step 1: Filter data where value > 59873
   Step 2: Sum those filtered values
   
   The data includes "all_data" with ALL rows - use this to filter and calculate!

3. DO NOT ESTIMATE OR INTERPOLATE:
   - You have the actual data in "all_data"
   - Filter it, then perform the exact calculation
   - Do NOT use percentile estimation or approximation

4. ANSWER FORMAT:
   - If question asks for a number, return just the number
   - If question asks for text, return the text
   - Match the format requested

5. USE THE DATA PROVIDED:
   - "all_data" contains all rows for filtering
   - "column_analysis" has pre-calculated stats (sum, mean, etc.)
   - For filtering questions, iterate through "all_data"

EXAMPLE CALCULATION PATTERNS:
- "Sum of values > X": sum([row[col] for row in all_data if row[col] > X])
- "Count values > X": len([row for row in all_data if row[col] > X])
- "Sum of column Y": column_analysis[Y]["sum"]

Return ONLY valid JSON (no markdown, no code blocks):
{{
    "answer": <calculated_answer>,
    "reasoning": "Step-by-step: what operation I performed and why",
    "confidence": "high/medium/low"
}}

CRITICAL: Perform the EXACT calculation requested. Do not estimate, interpolate, or approximate when you have the actual data!"""

    try:
        logger.info("🤖 Querying LLM...")
        
        response = client.chat.completions.create(
            model=AIMLAPI_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a precise data analyst. Read questions word-by-word. Perform EXACT calculations using the provided data. COUNT ≠ SUM. Filter first, then aggregate."
                },
                {"role": "user", "content": prompt}
            ],
            #temperature=0.1,
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"🤖 LLM response: {response_text[:500]}")
        
        # Parse JSON
        response_text = re.sub(r'```json\s*|\s*```', '', response_text)
        
        try:
            solution = json.loads(response_text)
        except:
            match = re.search(r'\{.*?"answer".*?\}', response_text, re.DOTALL)
            if match:
                solution = json.loads(match.group())
            else:
                return {"error": "Could not parse LLM response", "raw": response_text}
        
        if "answer" not in solution:
            return {"error": "No answer in response", "raw": response_text}
        
        answer = solution["answer"]
        
        # Validate answer
        if answer == "" or answer is None or answer == "N/A":
            logger.warning("⚠️ LLM returned empty/null answer, attempting retry...")
            
            retry_prompt = f"""RETRY - Previous answer was invalid.

QUESTION (READ WORD BY WORD):
{page_text[:2000]}

DATA AVAILABLE:
{json.dumps(downloaded_files, indent=2, default=str)[:15000]}

The question asks you to perform a specific calculation. You have ALL the data needed.

If the question mentions "sum of numbers greater than X":
1. Look in all_data for ALL rows
2. Filter rows where value > X
3. Sum those filtered values

If the question asks "count how many":
1. Filter the data based on condition
2. Return the count

Return JSON with the EXACT calculated answer:
{{"answer": <actual_number_from_calculation>, "reasoning": "I filtered X rows where value > Y, then summed them to get Z"}}"""

            retry_response = client.chat.completions.create(
                model=AIMLAPI_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You have the complete dataset. Filter it, then calculate. Return the exact answer."
                    },
                    {"role": "user", "content": retry_prompt}
                ],
                #temperature=0.1,
            )
            
            retry_text = retry_response.choices[0].message.content.strip()
            retry_text = re.sub(r'```json\s*|\s*```', '', retry_text)
            
            try:
                solution = json.loads(retry_text)
                answer = solution.get("answer")
                logger.info(f"✅ Retry answer: {answer}")
            except:
                logger.error("❌ Retry also failed")
        
        logger.info(f"✅ Final answer: {answer} (confidence: {solution.get('confidence', 'unknown')})")
        return solution
        
    except Exception as e:
        logger.error(f"❌ LLM error: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def submit_answer(submit_url, email, secret, quiz_url, answer):
    """Submit answer with URL validation"""
    
    if not submit_url or not submit_url.startswith('http'):
        logger.error(f"❌ Invalid submit URL: {submit_url}")
        
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
        logger.info(f"📤 Answer: {json.dumps(answer) if isinstance(answer, (dict, list)) else answer}")
        
        resp = requests.post(submit_url, json=payload, timeout=30)
        
        logger.info(f"📥 Status: {resp.status_code}")
        
        try:
            result = resp.json()
            logger.info(f"📥 Result: {json.dumps(result, indent=2)}")
        except:
            result = {"raw": resp.text, "status": resp.status_code}
        
        return result
    except Exception as e:
        logger.error(f"❌ Submit error: {e}")
        return {"error": str(e)}


def process_quiz_chain(initial_url, email, secret, start_time, timeout=170):
    """Process quiz chain - FULLY DYNAMIC"""
    current_url = initial_url
    results = []
    iteration = 0
    
    while current_url and (time.time() - start_time) < timeout:
        iteration += 1
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🎯 QUIZ #{iteration}")
        logger.info(f"📍 URL: {current_url}")
        logger.info(f"⏱️  Time: {elapsed:.1f}s elapsed, {remaining:.1f}s remaining")
        logger.info(f"{'='*70}")
        
        try:
            quiz_page = extract_page_content(current_url)
            if quiz_page.get("error"):
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": quiz_page["error"],
                    "status": "extraction_failed"
                })
                break
            
            combined_content = quiz_page.get("html", "") + "\n" + quiz_page.get("text", "")
            submit_url = extract_submit_url(combined_content, current_url)
            
            if not submit_url:
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "Could not find submit URL",
                    "status": "no_submit_url"
                })
                break
            
            downloaded_files = {}
            for link in quiz_page.get("links", []):
                if 'submit' in link.lower() and 'data' not in link.lower():
                    continue
                
                parsed_link = urlparse(link)
                file_ext = parsed_link.path.split('.')[-1].lower() if '.' in parsed_link.path else ''
                
                if file_ext in ['pdf', 'csv', 'json', 'xlsx', 'xls', 'txt']:
                    file_content = download_file(link)
                    if file_content:
                        parsed = smart_parse_file(file_content, link)
                        downloaded_files[link] = parsed
                else:
                    logger.info(f"🌐 Scraping linked page: {link}")
                    scraped = extract_page_content(link)
                    if not scraped.get("error"):
                        downloaded_files[link] = {
                            "type": "scraped_page",
                            "content": scraped.get("text", "")
                        }
            
            solution = solve_with_llm(quiz_page, downloaded_files, current_url)
            
            if "error" in solution:
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": solution.get("error"),
                    "status": "solving_failed"
                })
                break
            
            answer = solution.get("answer")
            if answer is None:
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "No answer from LLM",
                    "status": "no_answer"
                })
                break
            
            submission = submit_answer(submit_url, email, secret, current_url, answer)
            
            is_correct = submission.get("correct", False)
            
            results.append({
                "quiz_number": iteration,
                "url": current_url,
                "solution": solution,
                "submission_result": submission,
                "status": "correct" if is_correct else "incorrect",
                "correct": is_correct
            })
            
            next_url = submission.get("url")
            if next_url:
                current_url = next_url
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
        "service": "Universal Quiz Solver - Enhanced",
        "version": "5.0",
        "status": "running",
        "model": AIMLAPI_MODEL,
        "capabilities": [
            "Web scraping (JavaScript support)",
            "Statistical analysis (mean, median, std, quartiles)",
            "Filtering + Aggregation (sum/count with conditions)",
            "ML predictions & classifications",
            "Data cleaning & transformation",
            "File parsing (PDF, CSV, Excel, JSON)",
            "Complex calculations with full data access"
        ],
        "max_time_per_quiz_chain": "170 seconds"
    }), 200


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "llm_initialized": client is not None,
        "email_configured": bool(EMAIL),
        "secret_configured": bool(SECRET),
        "model": AIMLAPI_MODEL
    }), 200


@app.route("/quiz", methods=["POST"])
def quiz_endpoint():
    """Main webhook endpoint"""
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
            start_time,
            timeout=170
        )
        
        total_time = time.time() - start_time
        num_correct = sum(1 for r in results if r.get("correct"))
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ COMPLETED: {num_correct}/{len(results)} correct in {total_time:.2f}s")
        logger.info(f"{'='*70}")
        
        return jsonify({
            "status": "completed",
            "results": results,
            "summary": {
                "quizzes_attempted": len(results),
                "quizzes_correct": num_correct,
                "time_taken": round(total_time, 2)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Fatal: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "time_taken": round(time.time() - start_time, 2)
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    logger.info(f"🚀 Enhanced Quiz Solver starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)