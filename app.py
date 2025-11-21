#!/usr/bin/env python3
"""
Enhanced app.py with improved LLM prompting and error handling
"""

import os
import json
import time
import requests
import re
import io
import traceback
from urllib.parse import urljoin

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


def setup_browser():
    """Initialize headless Chrome with better settings"""
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
    
    if CHROME_BINARY and os.path.exists(CHROME_BINARY):
        opts.binary_location = CHROME_BINARY
    
    service = Service(executable_path=CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(30)
    return driver


def extract_quiz_page(url):
    """Extract quiz content with better waiting"""
    logger.info(f"🌐 Extracting: {url}")
    driver = setup_browser()
    
    try:
        driver.get(url)
        
        # Wait for body
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Additional wait for dynamic content
        time.sleep(3)
        
        # Try to wait for common result containers
        for selector in ["#result", ".content", ".quiz-content"]:
            try:
                WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                break
            except:
                pass
        
        page_source = driver.page_source
        page_text = driver.find_element(By.TAG_NAME, "body").text
        
        # Extract all links with better filtering
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


def download_file(url):
    """Download file with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"📥 Downloading (attempt {attempt+1}): {url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            logger.info(f"✅ Downloaded {len(resp.content)} bytes")
            return resp.content
        except Exception as e:
            logger.warning(f"⚠️ Download attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"❌ Download failed after {max_retries} attempts")
                return None
            time.sleep(2)
    return None


def parse_pdf(pdf_bytes):
    """Parse PDF with comprehensive table extraction"""
    try:
        logger.info("📄 Parsing PDF...")
        result = {
            "text_by_page": [],
            "tables_by_page": [],
            "all_text": "",
            "all_tables": []
        }
        
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text()
                if text:
                    result["text_by_page"].append({
                        "page": page_num,
                        "text": text
                    })
                    result["all_text"] += f"\n--- Page {page_num} ---\n{text}"
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 0:
                            try:
                                # Create DataFrame
                                if len(table) > 1:
                                    df = pd.DataFrame(table[1:], columns=table[0])
                                else:
                                    df = pd.DataFrame(table)
                                
                                # Clean column names
                                df.columns = [str(col).strip() if col else f"col_{i}" 
                                            for i, col in enumerate(df.columns)]
                                
                                table_info = {
                                    "page": page_num,
                                    "table_index": table_idx,
                                    "dataframe": df.to_dict('records'),
                                    "columns": list(df.columns),
                                    "shape": df.shape,
                                    "summary": df.describe().to_dict() if df.select_dtypes(include='number').shape[1] > 0 else None
                                }
                                
                                result["tables_by_page"].append(table_info)
                                result["all_tables"].append(table_info)
                                
                                logger.info(f"  Table on page {page_num}: {df.shape}")
                            except Exception as e:
                                logger.warning(f"  Could not parse table: {e}")
        
        logger.info(f"✅ PDF: {len(result['text_by_page'])} pages, {len(result['all_tables'])} tables")
        return result
    except Exception as e:
        logger.error(f"❌ PDF parsing error: {e}")
        return None


def parse_csv(csv_content):
    """Parse CSV with better error handling"""
    try:
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode("utf-8")
        
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        logger.info(f"✅ CSV: {df.shape[0]} rows × {df.shape[1]} columns")
        logger.info(f"  Columns: {list(df.columns)}")
        
        return {
            "data": df.to_dict('records'),
            "columns": list(df.columns),
            "shape": df.shape,
            "summary": df.describe().to_dict() if df.select_dtypes(include='number').shape[1] > 0 else None
        }
    except Exception as e:
        logger.error(f"❌ CSV parsing error: {e}")
        return None


def extract_submit_url(page_content, quiz_url):
    """Extract submit URL with multiple strategies"""
    if not page_content:
        return None
    
    # Try multiple patterns
    patterns = [
        r'Post your answer to\s+(https?://[^\s<>"]+/submit[^\s<>"]*)',
        r'submit.*?to\s+(https?://[^\s<>"]+/submit[^\s<>"]*)',
        r'(https://tds-llm-analysis\.s-anand\.net/submit[^\s<>"]*)',
        r'(https?://[^\s<>"]+/submit[^\s<>"]*)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, page_content, flags=re.IGNORECASE | re.DOTALL)
        if matches:
            url = matches[0].strip()
            logger.info(f"✅ Found submit URL: {url}")
            return url
    
    # Fallback to base URL if mentioned
    if 'submit' in page_content.lower():
        base_url = 'https://tds-llm-analysis.s-anand.net/submit'
        logger.info(f"⚠️ Using fallback submit URL: {base_url}")
        return base_url
    
    logger.warning("❌ No submit URL found")
    return None


def solve_quiz_with_llm(quiz_data, email, secret, quiz_url):
    """Solve quiz with enhanced LLM prompting"""
    if not client:
        return {"error": "LLM not initialized"}
    
    # Download and parse all linked data
    downloaded_data = {}
    data_links = [l for l in quiz_data.get("links", [])[:10] 
                  if not ('submit' in l.lower() and 'data' not in l.lower())]
    
    for link in data_links:
        try:
            file_content = download_file(link)
            if not file_content:
                continue
            
            # Determine file type and parse
            if file_content[:4] == b"%PDF":
                parsed = parse_pdf(file_content)
                if parsed:
                    downloaded_data[link] = {
                        "type": "pdf",
                        "content": parsed
                    }
            elif link.lower().endswith('.csv') or b',' in file_content[:200]:
                parsed = parse_csv(file_content)
                if parsed:
                    downloaded_data[link] = {
                        "type": "csv",
                        "content": parsed
                    }
            elif link.lower().endswith('.json'):
                try:
                    parsed = json.loads(file_content.decode('utf-8'))
                    downloaded_data[link] = {
                        "type": "json",
                        "content": parsed
                    }
                except:
                    pass
            else:
                try:
                    text = file_content.decode('utf-8')
                    downloaded_data[link] = {
                        "type": "text",
                        "content": text[:5000]
                    }
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error processing {link}: {e}")
    
    # Create comprehensive prompt
    prompt = f"""You are a data analyst solving a quiz. Follow these steps:

1. READ THE QUESTION CAREFULLY
2. ANALYZE THE PROVIDED DATA
3. CALCULATE THE EXACT ANSWER
4. RETURN ONLY THE ANSWER IN THE CORRECT FORMAT

QUESTION:
{quiz_data.get('text', '')[:4000]}

DATA FILES DOWNLOADED:
{json.dumps(downloaded_data, indent=2, default=str)[:15000]}

CRITICAL INSTRUCTIONS:
- If question asks for a SUM, calculate the sum of the specified column
- If question asks for a COUNT, count the items
- If question asks for a specific VALUE, extract that exact value
- If question asks for a STRING, return the string (not a number)
- If question asks for a NUMBER, return a number (integer or float)
- If question asks for a BOOLEAN, return true or false
- Pay attention to which PAGE or SECTION the question refers to

Your response MUST be ONLY valid JSON (no markdown, no explanation):
{{
    "answer": <put_the_actual_answer_here>,
    "reasoning": "brief 1-sentence explanation of how you got the answer"
}}

Examples:
- If sum is 12345, return: {{"answer": 12345, "reasoning": "..."}}
- If name is "John", return: {{"answer": "John", "reasoning": "..."}}
- If yes/no, return: {{"answer": true, "reasoning": "..."}}"""

    try:
        logger.info("🤖 Querying LLM...")
        
        response = client.chat.completions.create(
            model=AIMLAPI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise data analyst. Respond ONLY with valid JSON containing 'answer' and 'reasoning' fields. No markdown formatting."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            #temperature=0.1,
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"🤖 LLM Response: {response_text[:300]}")
        
        # Parse JSON response
        try:
            # Remove markdown code blocks if present
            response_text = re.sub(r'```json\s*|\s*```', '', response_text)
            solution = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON object
            match = re.search(r'\{[^{}]*"answer"[^{}]*\}', response_text, re.DOTALL)
            if match:
                solution = json.loads(match.group())
            else:
                return {
                    "error": "Could not parse LLM response as JSON",
                    "raw_response": response_text
                }
        
        if "answer" not in solution:
            return {
                "error": "LLM response missing 'answer' field",
                "raw_response": response_text
            }
        
        logger.info(f"✅ Extracted answer: {solution.get('answer')}")
        return solution
        
    except Exception as e:
        logger.error(f"❌ LLM error: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def submit_answer(submit_url, email, secret, quiz_url, answer):
    """Submit answer with retry logic"""
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"📤 Submitting (attempt {attempt+1}) to: {submit_url}")
            logger.info(f"📤 Payload: {json.dumps(payload, indent=2)}")
            
            resp = requests.post(submit_url, json=payload, timeout=30)
            
            logger.info(f"📥 Status: {resp.status_code}")
            
            try:
                result = resp.json()
                logger.info(f"📥 Response: {json.dumps(result, indent=2)}")
                return result
            except:
                result = {
                    "raw": resp.text,
                    "status": resp.status_code,
                    "error": "Could not parse response as JSON"
                }
                return result
                
        except requests.exceptions.Timeout:
            logger.warning(f"⚠️ Timeout on attempt {attempt+1}")
            if attempt == max_retries - 1:
                return {"error": "Submission timeout after retries"}
            time.sleep(2)
        except Exception as e:
            logger.error(f"❌ Submit error: {e}")
            if attempt == max_retries - 1:
                return {"error": str(e)}
            time.sleep(2)
    
    return {"error": "Failed after all retries"}


def process_quiz_chain(initial_url, email, secret, start_time, timeout=170):
    """Process chain of quizzes"""
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
            # Extract quiz page
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
                quiz_data.get("html", "") + "\n" + quiz_data.get("text", ""),
                current_url
            )
            
            if not submit_url:
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "Could not find submit URL",
                    "status": "no_submit_url"
                })
                break
            
            # Solve with LLM
            solution = solve_quiz_with_llm(quiz_data, email, secret, current_url)
            
            if "error" in solution:
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": solution.get("error"),
                    "raw_response": solution.get("raw_response"),
                    "status": "solving_failed"
                })
                break
            
            answer = solution.get("answer")
            if answer is None:
                results.append({
                    "quiz_number": iteration,
                    "url": current_url,
                    "error": "No answer extracted from LLM",
                    "solution": solution,
                    "status": "no_answer"
                })
                break
            
            # Submit answer
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
            
            # Check for next quiz
            next_url = submission.get("url")
            if next_url:
                current_url = next_url
                logger.info(f"➡️  Moving to next quiz: {current_url}")
            else:
                logger.info("🏁 Quiz chain completed!")
                break
                
        except Exception as e:
            logger.error(f"❌ Exception in quiz {iteration}: {e}")
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


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "LLM Quiz Solver",
        "version": "2.0",
        "status": "running",
        "model": AIMLAPI_MODEL,
        "llm_ready": client is not None,
        "endpoints": {
            "health": "/health",
            "quiz": "/quiz (POST)"
        }
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
    
    # Validate JSON
    if not request.is_json:
        logger.warning("❌ Invalid JSON received")
        return jsonify({"error": "Invalid JSON"}), 400
    
    data = request.get_json()
    logger.info(f"📨 Received request: {json.dumps(data, indent=2)}")
    
    # Validate secret
    if data.get("secret") != SECRET:
        logger.warning("❌ Invalid secret")
        return jsonify({"error": "Invalid secret"}), 403
    
    # Validate required fields
    if not data.get("email") or not data.get("url"):
        logger.warning("❌ Missing required fields")
        return jsonify({"error": "Missing email or url"}), 400
    
    # Check LLM
    if not client:
        logger.error("❌ LLM not initialized")
        return jsonify({"error": "LLM not initialized"}), 500
    
    try:
        logger.info("🚀 Starting quiz chain processing...")
        
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
        logger.info(f"✅ COMPLETED")
        logger.info(f"   Quizzes attempted: {len(results)}")
        logger.info(f"   Correct: {num_correct}")
        logger.info(f"   Time: {total_time:.2f}s")
        logger.info(f"{'='*70}")
        
        return jsonify({
            "status": "completed",
            "results": results,
            "summary": {
                "quizzes_attempted": len(results),
                "quizzes_correct": num_correct,
                "time_taken": round(total_time, 2),
                "success_rate": f"{num_correct}/{len(results)}"
            }
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "time_taken": round(time.time() - start_time, 2)
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    logger.info(f"🚀 Starting server on port {port}")
    logger.info(f"   Model: {AIMLAPI_MODEL}")
    logger.info(f"   Email: {EMAIL}")
    logger.info(f"   LLM Ready: {client is not None}")
    app.run(host="0.0.0.0", port=port, debug=False)