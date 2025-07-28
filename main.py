import os
import json
import requests
from flask import Flask, request, render_template, jsonify, redirect
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Check for required environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")

if not openai_api_key:
    print("WARNING: OPENAI_API_KEY not found in environment variables")
if not claude_api_key:
    print("WARNING: CLAUDE_API_KEY not found in environment variables")

# Global client variable
openai_client = None

# Globals
chat_history = []
pdf_text_memory = ""

# Log file
LOG_FILE = "logs/chat_history.json"
os.makedirs("logs", exist_ok=True)

# Initialize OpenAI client only when needed
def get_openai_client():
    global openai_client
    if openai_client is None and openai_api_key:
        try:
            from openai import OpenAI
            print(f"Attempting to create OpenAI client...")
            openai_client = OpenAI(api_key=openai_api_key)
            print("OpenAI client created successfully")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            return None
    return openai_client

# Direct Claude API call (bypassing problematic SDK)
def call_claude_direct(system_prompt, user_message):
    try:
        print("Calling Claude API directly")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": claude_api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_message}
            ]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["content"][0]["text"]
        else:
            print(f"Claude API error: {response.status_code} - {response.text}")
            return f"Error calling Claude API: {response.status_code}"
            
    except Exception as e:
        print(f"Error in direct Claude call: {e}")
        return f"Error: {str(e)}"

# Save/Load logs
def save_chat_log():
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving chat log: {e}")

def load_chat_log():
    global chat_history
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                chat_history = json.load(f)
    except Exception as e:
        print(f"Error loading chat log: {e}")
        chat_history = []

# PDF extraction - Temporarily disabled
def extract_text_from_pdf(file_path):
    return "PDF upload temporarily disabled - coming soon!"

# Intelligent model switching
def get_model(user_input):
    for keyword in config.get("keywords_for_openai", []):
        if keyword in user_input.lower():
            return config["openai_model"]
    return config["claude_model"]

# Unified call - Claude via direct API, OpenAI via SDK
def call_model(model, system, prompt):
    try:
        if "gpt" in model:
            print(f"Using OpenAI model: {model}")
            client = get_openai_client()
            if not client:
                return "Error: OpenAI API key not configured or client failed to initialize"
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        else:
            print(f"Using Claude via direct API: {model}")
            if not claude_api_key:
                return "Error: Claude API key not configured"
            return call_claude_direct(system, prompt)
    except Exception as e:
        print(f"Error calling model: {e}")
        return f"Error: {str(e)}"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history, pdf_text_memory
    try:
        data = request.get_json()
        user_input = data.get("user_input", "")
        agent = data.get("agent", "")

        print(f"Chat request - User: {user_input}, Agent: {agent}")

        chat_history.append({"role": "user", "content": user_input})

        if agent == "case_assistant":
            system_prompt = "You assist with social work case analysis. Focus on context, safeguarding, and systemic risk."
            model = config["claude_model"]
        elif agent == "research_critic":
            system_prompt = "You are a critical evaluator of research. Be sharp, analytical, and cite relevant frameworks."
            model = config["openai_model"]
        elif agent == "therapy_planner":
            system_prompt = "You are a strategic therapist. Your job is to plan sessions and structure interventions, using systemic and psychoanalytic models."
            model = config["openai_model"]
        else:
            system_prompt = config["claude_system_prompt"]
            model = config["claude_model"]

        if pdf_text_memory:
            system_prompt += f"\n\nReference material:\n{pdf_text_memory[:3000]}"

        print(f"Selected model: {model}")
        response_text = call_model(model, system_prompt, user_input)
        chat_history.append({"role": "assistant", "content": response_text})
        save_chat_log()

        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    global pdf_text_memory
    try:
        if "pdf" not in request.files:
            return redirect("/")
        file = request.files["pdf"]
        if file and file.filename.endswith(".pdf"):
            pdf_text_memory = "PDF processing temporarily disabled - coming soon!"
        return redirect("/")
    except Exception as e:
        print(f"Error in upload endpoint: {e}")
        return redirect("/")

@app.route("/clear", methods=["POST"])
def clear():
    global chat_history, pdf_text_memory
    try:
        chat_history = []
        pdf_text_memory = ""
        save_chat_log()
        return jsonify({"message": "Chat cleared"})
    except Exception as e:
        print(f"Error in clear endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/log", methods=["GET"])
def get_log():
    return jsonify(chat_history)

# Health check endpoint
@app.route("/health")
def health():
    openai_works = False
    claude_works = False
    
    # Test OpenAI
    if openai_api_key:
        try:
            test_client = get_openai_client()
            openai_works = test_client is not None
        except:
            openai_works = False
    
    # Test Claude direct API
    if claude_api_key:
        try:
            test_response = call_claude_direct("You are a test assistant.", "Hello")
            claude_works = not test_response.startswith("Error")
        except:
            claude_works = False
    
    return jsonify({
        "status": "healthy",
        "openai_configured": openai_api_key is not None,
        "openai_client_works": openai_works,
        "claude_configured": claude_api_key is not None,
        "claude_client_works": claude_works,
        "pdf_support": False,
        "note": "Using direct Claude API calls to bypass SDK issues"
    })

if __name__ == "__main__":
    load_chat_log()
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    load_chat_log()
    application = app
