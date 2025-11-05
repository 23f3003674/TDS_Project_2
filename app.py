import os
import json
import time
import requests
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

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
EMAIL = os.getenv("STUDENT_EMAIL")
SECRET = os.getenv("STUDENT_SECRET")
AIMLAPI_BASE_URL = os.getenv("AIMLAPI_BASE_URL", "https://aipipe.org/openai/v1")
AIMLAPI_API_KEY = os.getenv("AIMLAPI_API_KEY")
AIMLAPI_MODEL = os.getenv("AIMLAPI_MODEL", "gpt-5-nano")

# Initialize OpenAI client with AIPipe
client = OpenAI(
    api_key=AIMLAPI_API_KEY,
    base_url=AIMLAPI_BASE_URL
)

def setup_browser():
    """Initialize headless Chrome browser"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-setuid-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    # Use system Chrome/Chromium
    chrome_options.binary_location = "/usr/bin/chromium"
    
    service = Service(executable_path="/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(30)
    
    return driver

def extract_quiz_content(url):
    """Extract quiz content from the given URL using Selenium"""
    logger.info(f"Extracting content from: {url}")
    driver = setup_browser()
    try:
        driver.get(url)
        # Wait for JavaScript to execute
        time.sleep(3)
        
        # Try to find the result div or any content
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except:
            pass
        
        # Get the full page content
        page_source = driver.page_source
        page_text = driver.find_element(By.TAG_NAME, "body").text
        
        logger.info(f"Extracted {len(page_text)} characters of text")
        
        return {
            "html": page_source,
            "text": page_text,
            "url": url
        }
    except Exception as e:
        logger.error(f"Error extracting content: {str(e)}")
        raise
    finally:
        driver.quit()

def solve_quiz_with_llm(quiz_content, email, secret, quiz_url):
    """Use GPT-5-nano via AIPipe to analyze and solve the quiz"""
    
    prompt = f"""You are a data analysis expert solving a quiz task. Here's the quiz content:

TEXT CONTENT:
{quiz_content['text']}

HTML SOURCE (for reference):
{quiz_content['html'][:5000]}

TASK:
1. Carefully read and understand what the quiz is asking
2. Identify what data needs to be downloaded or accessed
3. Determine what analysis needs to be performed
4. Calculate or determine the correct answer
5. Format the answer according to the instructions

The answer should be submitted to the URL mentioned in the quiz with this JSON structure:
{{
    "email": "{email}",
    "secret": "{secret}",
    "url": "{quiz_url}",
    "answer": <your_answer_here>
}}

IMPORTANT:
- Extract the submit URL from the quiz instructions (do not hardcode)
- The answer can be: boolean, number, string, base64 URI, or JSON object
- Provide your response in this exact JSON format:
{{
    "submit_url": "the URL to submit to",
    "answer": the_actual_answer,
    "reasoning": "brief explanation of your solution",
    "steps": ["step 1", "step 2", ...]
}}

If the task requires downloading files, accessing APIs, or complex processing, describe the steps needed."""

    try:
        logger.info(f"Sending request to {AIMLAPI_MODEL}")
        
        response = client.chat.completions.create(
            model=AIMLAPI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful data analysis assistant. Always respond with valid JSON when requested."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        response_text = response.choices[0].message.content
        logger.info(f"Received response: {response_text[:200]}...")
        
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            solution = json.loads(json_match.group())
            return solution
        else:
            return {
                "error": "Could not parse LLM response",
                "raw_response": response_text
            }
            
    except Exception as e:
        logger.error(f"Error in LLM solving: {str(e)}")
        return {"error": str(e)}

def submit_answer(submit_url, email, secret, quiz_url, answer):
    """Submit the answer to the specified endpoint"""
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer
    }
    
    try:
        logger.info(f"Submitting answer to: {submit_url}")
        logger.info(f"Answer: {answer}")
        
        response = requests.post(submit_url, json=payload, timeout=30)
        result = response.json()
        
        logger.info(f"Submission result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error submitting answer: {str(e)}")
        return {"error": str(e)}

def process_quiz_chain(initial_url, email, secret, start_time, timeout=180):
    """Process a chain of quiz questions"""
    current_url = initial_url
    results = []
    
    while current_url and (time.time() - start_time) < timeout:
        try:
            logger.info(f"Processing quiz: {current_url}")
            
            # Extract quiz content
            quiz_content = extract_quiz_content(current_url)
            
            # Solve with LLM
            solution = solve_quiz_with_llm(quiz_content, email, secret, current_url)
            
            if "error" in solution:
                results.append({
                    "url": current_url,
                    "error": solution["error"],
                    "status": "failed"
                })
                break
            
            # Submit answer
            submit_url = solution.get("submit_url")
            answer = solution.get("answer")
            
            if not submit_url or answer is None:
                results.append({
                    "url": current_url,
                    "error": "Missing submit_url or answer",
                    "solution": solution,
                    "status": "failed"
                })
                break
            
            submission_result = submit_answer(submit_url, email, secret, current_url, answer)
            
            results.append({
                "url": current_url,
                "solution": solution,
                "submission_result": submission_result,
                "status": "submitted"
            })
            
            # Check if there's a next URL
            if submission_result.get("correct") and submission_result.get("url"):
                current_url = submission_result["url"]
            elif not submission_result.get("correct") and submission_result.get("url"):
                # Got next URL even though wrong, can skip or retry
                current_url = submission_result["url"]
            else:
                # No more URLs, quiz complete
                break
                
        except Exception as e:
            logger.error(f"Exception in quiz chain: {str(e)}")
            results.append({
                "url": current_url,
                "error": str(e),
                "status": "exception"
            })
            break
    
    return results

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        "service": "LLM Quiz Solver",
        "status": "running",
        "model": AIMLAPI_MODEL,
        "endpoints": {
            "health": "/health",
            "quiz": "/quiz (POST)"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": AIMLAPI_MODEL,
        "base_url": AIMLAPI_BASE_URL
    }), 200

@app.route('/quiz', methods=['POST'])
def quiz_endpoint():
    """Main endpoint to receive quiz tasks"""
    start_time = time.time()
    
    # Validate request
    if not request.is_json:
        logger.warning("Received non-JSON request")
        return jsonify({"error": "Invalid JSON"}), 400
    
    data = request.get_json()
    logger.info(f"Received quiz request: {data}")
    
    # Validate secret
    if data.get("secret") != SECRET:
        logger.warning(f"Invalid secret received")
        return jsonify({"error": "Invalid secret"}), 403
    
    # Validate required fields
    if not data.get("email") or not data.get("url"):
        logger.warning("Missing required fields")
        return jsonify({"error": "Missing required fields"}), 400
    
    # Process the quiz
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
            "time_taken": time.time() - start_time
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing quiz: {str(e)}")
        return jsonify({
            "error": str(e),
            "time_taken": time.time() - start_time
        }), 500

if __name__ == '__main__':
    # For local development
    port = int(os.getenv("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)