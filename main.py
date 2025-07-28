import os
import json
import requests
import hashlib
from datetime import datetime
from flask import Flask, request, render_template, jsonify, redirect, session
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-this")

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Check for required environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")
admin_password = os.getenv("ADMIN_PASSWORD", "admin123")  # Set this in your environment

if not openai_api_key:
    print("WARNING: OPENAI_API_KEY not found in environment variables")
if not claude_api_key:
    print("WARNING: CLAUDE_API_KEY not found in environment variables")

# Globals
chat_history = []
user_sessions = {}  # Store per-user chat history

# File paths
LOG_FILE = "logs/chat_history.json"
KNOWLEDGE_BASE_FILE = "core_memory/knowledge_base.json"
CORE_MEMORY_DIR = "core_memory"
UPLOADS_DIR = "uploads"
USER_UPLOADS_DIR = "user_uploads"

# Ensure directories exist
for directory in [CORE_MEMORY_DIR, UPLOADS_DIR, USER_UPLOADS_DIR, "logs"]:
    os.makedirs(directory, exist_ok=True)

# Load knowledge base
def load_knowledge_base():
    try:
        if os.path.exists(KNOWLEDGE_BASE_FILE):
            with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
    return {"documents": [], "last_updated": None, "total_documents": 0}

def save_knowledge_base(knowledge_base):
    try:
        knowledge_base["last_updated"] = datetime.now().isoformat()
        with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving knowledge base: {e}")

knowledge_base = load_knowledge_base()

# Extract text from PDF (placeholder - you'll need to add PyMuPDF back later)
def extract_text_from_pdf(file_path):
    # TODO: Re-add PyMuPDF when compilation is fixed
    # For now, return placeholder
    return f"PDF text extraction placeholder for: {os.path.basename(file_path)}"

def add_document_to_knowledge_base(file_path, filename, is_core=True):
    """Add document to permanent knowledge base (admin only) or temporary session"""
    try:
        text_content = extract_text_from_pdf(file_path)
        
        doc_info = {
            "filename": filename,
            "content": text_content,
            "added_date": datetime.now().isoformat(),
            "file_hash": hashlib.md5(open(file_path, 'rb').read()).hexdigest(),
            "is_core": is_core
        }
        
        if is_core:
            # Add to permanent knowledge base
            knowledge_base["documents"].append(doc_info)
            knowledge_base["total_documents"] = len(knowledge_base["documents"])
            save_knowledge_base(knowledge_base)
            print(f"Added {filename} to core knowledge base")
        
        return doc_info
    except Exception as e:
        print(f"Error adding document: {e}")
        return None

def get_session_id():
    """Get or create session ID for user"""
    if 'session_id' not in session:
        session['session_id'] = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:16]
    return session['session_id']

def get_user_session(session_id):
    """Get user session data"""
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            "chat_history": [],
            "temp_documents": [],
            "created": datetime.now().isoformat()
        }
    return user_sessions[session_id]

# Direct Claude API call
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

# Direct OpenAI API call
def call_openai_direct(system_prompt, user_message):
    try:
        print("Calling OpenAI API directly")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 1024
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"OpenAI API error: {response.status_code} - {response.text}")
            return f"Error calling OpenAI API: {response.status_code}"
            
    except Exception as e:
        print(f"Error in direct OpenAI call: {e}")
        return f"Error: {str(e)}"

def build_context_from_knowledge_base(session_id):
    """Build context from core knowledge + user session docs"""
    context = ""
    
    # Add core knowledge base
    if knowledge_base["documents"]:
        context += "\n=== CORE KNOWLEDGE BASE ===\n"
        for doc in knowledge_base["documents"][-3:]:  # Last 3 core documents
            context += f"From {doc['filename']}:\n{doc['content'][:1000]}...\n\n"
    
    # Add user session documents
    user_session = get_user_session(session_id)
    if user_session["temp_documents"]:
        context += "\n=== SESSION DOCUMENTS ===\n"
        for doc in user_session["temp_documents"]:
            context += f"From {doc['filename']}:\n{doc['content'][:500]}...\n\n"
    
    return context[:4000]  # Limit total context

# Intelligent model switching with knowledge base
def get_model(user_input):
    for keyword in config.get("keywords_for_openai", []):
        if keyword in user_input.lower():
            return config["openai_model"]
    return config["claude_model"]

def call_model(model, system, prompt, session_id):
    try:
        # Add knowledge base context
        knowledge_context = build_context_from_knowledge_base(session_id)
        if knowledge_context:
            system += f"\n\nKNOWLEDGE BASE CONTEXT:\n{knowledge_context}"
        
        if "gpt" in model:
            print(f"Using OpenAI via direct API: {model}")
            if not openai_api_key:
                return "Error: OpenAI API key not configured"
            return call_openai_direct(system, prompt)
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
    session_id = get_session_id()
    return render_template("index.html")

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        password = request.form.get("password")
        if password == admin_password:
            session["is_admin"] = True
            return redirect("/admin")
        else:
            return render_template("admin_login.html", error="Invalid password")
    
    if not session.get("is_admin"):
        return render_template("admin_login.html")
    
    # Show admin dashboard
    return render_template("admin_dashboard.html", 
                         knowledge_base=knowledge_base,
                         total_docs=len(knowledge_base["documents"]))

@app.route("/chat", methods=["POST"])
def chat():
    try:
        session_id = get_session_id()
        user_session = get_user_session(session_id)
        
        data = request.get_json()
        user_input = data.get("user_input", "")
        agent = data.get("agent", "")

        print(f"Chat request - User: {user_input}, Agent: {agent}, Session: {session_id}")

        user_session["chat_history"].append({"role": "user", "content": user_input})

        if agent == "case_assistant":
            system_prompt = "You assist with social work case analysis. Focus on context, safeguarding, and systemic risk. Use the knowledge base to inform your responses with evidence-based practices."
            model = config["claude_model"]
        elif agent == "research_critic":
            system_prompt = "You are a critical evaluator of research. Be sharp, analytical, and cite relevant frameworks. Reference the knowledge base materials in your analysis."
            model = config["openai_model"]
        elif agent == "therapy_planner":
            system_prompt = "You are a strategic therapist. Plan sessions and structure interventions using systemic and psychoanalytic models. Draw on the knowledge base for theoretical grounding."
            model = config["openai_model"]
        else:
            system_prompt = config["claude_system_prompt"] + " Draw upon the uploaded books and materials in your knowledge base to provide theoretically grounded responses."
            model = config["claude_model"]

        response_text = call_model(model, system_prompt, user_input, session_id)
        user_session["chat_history"].append({"role": "assistant", "content": response_text})

        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    try:
        session_id = get_session_id()
        user_session = get_user_session(session_id)
        
        if "pdf" not in request.files:
            return redirect("/")
        
        file = request.files["pdf"]
        if file and file.filename.endswith(".pdf"):
            is_admin = session.get("is_admin", False)
            
            if is_admin:
                # Admin upload - goes to core knowledge base
                file_path = os.path.join(UPLOADS_DIR, file.filename)
                file.save(file_path)
                doc_info = add_document_to_knowledge_base(file_path, file.filename, is_core=True)
                if doc_info:
                    return jsonify({"message": f"Added {file.filename} to core knowledge base", "success": True})
            else:
                # User upload - temporary session only
                file_path = os.path.join(USER_UPLOADS_DIR, f"{session_id}_{file.filename}")
                file.save(file_path)
                doc_info = add_document_to_knowledge_base(file_path, file.filename, is_core=False)
                if doc_info:
                    user_session["temp_documents"].append(doc_info)
                    return jsonify({"message": f"Added {file.filename} to your session (temporary)", "success": True})
        
        return jsonify({"message": "Upload failed", "success": False})
    except Exception as e:
        print(f"Error in upload endpoint: {e}")
        return jsonify({"error": str(e), "success": False})

@app.route("/clear", methods=["POST"])
def clear():
    try:
        session_id = get_session_id()
        user_session = get_user_session(session_id)
        user_session["chat_history"] = []
        user_session["temp_documents"] = []
        return jsonify({"message": "Chat and session documents cleared"})
    except Exception as e:
        print(f"Error in clear endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/log", methods=["GET"])
def get_log():
    session_id = get_session_id()
    user_session = get_user_session(session_id)
    return jsonify(user_session["chat_history"])

@app.route("/knowledge-base", methods=["GET"])
def get_knowledge_base():
    """API endpoint to view knowledge base status"""
    return jsonify({
        "total_documents": len(knowledge_base["documents"]),
        "last_updated": knowledge_base.get("last_updated"),
        "documents": [{"filename": doc["filename"], "added_date": doc["added_date"]} 
                     for doc in knowledge_base["documents"]]
    })

@app.route("/health")
def health():
    openai_works = False
    claude_works = False
    
    # Test OpenAI direct API
    if openai_api_key:
        try:
            test_response = call_openai_direct("You are a test assistant.", "Hello")
            openai_works = not test_response.startswith("Error")
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
        "knowledge_base_documents": len(knowledge_base["documents"]),
        "pdf_support": False,
        "note": "Using direct API calls for both Claude and OpenAI"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    application = app
