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

# Log configuration status
logger.info("="*60)
logger.info("🔧 Configuration Status:")
logger.info(f"   EMAIL: {'✅ Set' if EMAIL else '❌ Not set'}")
logger.info(f"   SECRET: {'✅ Set' if SECRET else '❌ Not set'}")
logger.info(f"   API_KEY: {'✅ Set' if AIMLAPI_API_KEY else '❌ Not set'}")
logger.info(f"   BASE_URL: {AIMLAPI_BASE_URL}")
logger.info(f"   MODEL: {AIMLAPI_MODEL}")
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
    """Initialize headless Chrome browser"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    chrome_options.binary_location = "/usr/bin/chromium"
    
    service = Service(executable_path="/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(30)
    return driver

def extract_quiz_page(url):
    """Extract quiz content using Selenium with JavaScript rendering"""
    logger.info(f"🌐 Extracting quiz from: {url}")
    driver = setup_browser()
    
    try:
        driver.get(url)
        time.sleep(5)  # Wait for JavaScript to execute
        
        # Wait for body to be present
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Get page content after JavaScript execution
        page_source = driver.page_source
        page_text = driver.find_element(By.TAG_NAME, "body").text
        
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
                tables_html.append(table.get_attribute('outerHTML'))
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
        driver.quit()

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
                text = page.extract_text()
                if text:
                    result["text_by_page"].append({
                        "page": page_num,
                        "text": text
                    })
                
                # Extract tables
                tables = page.extract_tables()
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
        "question": quiz_data["text"][:5000],
        "available_links": quiz_data.get("links", [])[:10],
        "tables_count": len(quiz_data.get("tables_html", [])),
        "downloaded_files": {}
    }
    
    # Auto-download files mentioned in links
    for link in quiz_data.get("links", [])[:5]:  # Limit to 5 files
        try:
            file_ext = link.split('.')[-1].lower()
            
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
                        parsed = json.loads(file_content.decode('utf-8'))
                        context["downloaded_files"][f"json_{link}"] = parsed
                        
        except Exception as e:
            logger.error(f"File processing error for {link}: {e}")
            continue
    
    # Create prompt for LLM
    prompt = f"""You are a data analyst solving a quiz. Analyze the question and data, then provide the answer.

QUESTION:
{context['question']}

AVAILABLE DATA:
- Links found: {len(context['available_links'])}
- Downloaded files: {list(context['downloaded_files'].keys())}

DOWNLOADED FILE DATA:
{json.dumps(context['downloaded_files'], indent=2)[:8000]}

TASK:
1. Read the question carefully
2. Identify what calculation/analysis is needed
3. Use the downloaded data to solve it
4. Provide the exact answer in the required format

Respond in JSON format:
{{
    "answer": the_actual_answer,
    "reasoning": "step-by-step explanation",
    "calculation": "show your work",
    "answer_type": "number/string/boolean/json"
}}

IMPORTANT:
- Be precise with numbers (no rounding unless specified)
- Follow the exact format requested in the question
- If the answer is a sum, provide just the number
- If it's text, provide the exact text"""

    try:
        logger.info("🤖 Sending to LLM...")
        
        response = client.chat.completions.create(
            model=AIMLAPI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise data analyst. Always respond with valid JSON containing the answer."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_completion_tokens=4000  # Changed from max_tokens
        )
        
        response_text = response.choices[0].message.content
        logger.info(f"🤖 LLM Response: {response_text[:500]}...")
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            solution = json.loads(json_match.group())
            logger.info(f"✅ Answer: {solution.get('answer')}")
            return solution
        else:
            logger.error("❌ Could not parse LLM response as JSON")
            return {
                "error": "Could not parse LLM response",
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
        except:
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
            submit_url = extract_submit_url(quiz_data['html'] + quiz_data['text'])
            
            # Step 3: Solve with LLM
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