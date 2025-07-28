import os
import json
# import fitz  # Commented out for now
from flask import Flask, request, render_template, jsonify, redirect
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Check for required environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("WARNING: OPENAI_API_KEY not found in environment variables")

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
    # TODO: Re-enable when PyMuPDF compilation is fixed
    return "PDF upload temporarily disabled - coming soon!"

# Use OpenAI for everything now
def call_model(system, prompt):
    try:
        print(f"Using OpenAI for all responses")
        client = get_openai_client()
        if not client:
            return "Error: OpenAI API key not configured or client failed to initialize"
        
        response = client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for all responses
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
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

        # All agents now use OpenAI with different system prompts
        if agent == "case_assistant":
            system_prompt = "You are a social work case analysis assistant. Focus on context, safeguarding, and systemic risk. Provide thoughtful, professional guidance for social work cases."
        elif agent == "research_critic":
            system_prompt = "You are a critical evaluator of research. Be sharp, analytical, and cite relevant frameworks. Provide rigorous academic analysis."
        elif agent == "therapy_planner":
            system_prompt = "You are a strategic therapist. Your job is to plan sessions and structure interventions, using systemic and psychoanalytic models. Be practical and evidence-based."
        else:
            # Default therapist role
            system_prompt = "You are a warm, reflective systemic co-therapist with a postmodern, constructivist lens. Provide empathetic, thoughtful therapeutic responses that help clients explore their experiences and relationships."

        if pdf_text_memory:
            system_prompt += f"\n\nReference material:\n{pdf_text_memory[:3000]}"

        response_text = call_model(system_prompt, user_input)
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
            # Temporarily return message instead of processing
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
    
    # Test OpenAI client creation
    if openai_api_key:
        try:
            test_client = get_openai_client()
            openai_works = test_client is not None
        except:
            openai_works = False
    
    return jsonify({
        "status": "healthy",
        "openai_configured": openai_api_key is not None,
        "openai_client_works": openai_works,
        "claude_configured": False,  # Temporarily disabled
        "claude_client_works": False,  # Temporarily disabled
        "pdf_support": False,
        "note": "Running in OpenAI-only mode to bypass Claude SDK issues"
    })

if __name__ == "__main__":
    load_chat_log()
    # For local development only
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    # For production
    load_chat_log()
    application = app
