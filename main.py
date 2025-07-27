import os
import json
import fitz
from flask import Flask, request, render_template, jsonify, redirect
from openai import OpenAI
from anthropic import Anthropic
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

# Clients (only initialize if keys are available)
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
claude_client = Anthropic(api_key=claude_api_key) if claude_api_key else None

# Globals
chat_history = []
pdf_text_memory = ""

# Log file
LOG_FILE = "logs/chat_history.json"
os.makedirs("logs", exist_ok=True)

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

# PDF extraction
def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

# Intelligent model switching
def get_model(user_input):
    for keyword in config.get("keywords_for_openai", []):
        if keyword in user_input.lower():
            return config["openai_model"]
    return config["claude_model"]

# Unified call
def call_model(model, system, prompt):
    try:
        if "gpt" in model:
            if not openai_client:
                return "Error: OpenAI API key not configured"
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        else:
            if not claude_client:
                return "Error: Claude API key not configured"
            response = claude_client.messages.create(
                model=model,
                max_tokens=config.get("max_tokens", 1024),
                system=system,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
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
            os.makedirs("uploads", exist_ok=True)
            path = os.path.join("uploads", file.filename)
            file.save(path)
            pdf_text_memory = extract_text_from_pdf(path)
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

# Health check endpoint for Railway
@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "openai_configured": openai_client is not None,
        "claude_configured": claude_client is not None
    })

if __name__ == "__main__":
    load_chat_log()
    # For local development only
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    # For production
    load_chat_log()
    application = app

