import os
import json
import requests
import hashlib
import re
# import fitz  # PyMuPDF - COMMENTED OUT temporarily for deployment
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
admin_password = os.getenv("ADMIN_PASSWORD", "admin123")

if not openai_api_key:
    print("WARNING: OPENAI_API_KEY not found in environment variables")
if not claude_api_key:
    print("WARNING: CLAUDE_API_KEY not found in environment variables")

# Globals
user_sessions = {}

# File paths
KNOWLEDGE_BASE_FILE = "core_memory/knowledge_base.json"
AUTHORIZED_AUTHORS_FILE = "core_memory/authorized_authors.json"
CORE_MEMORY_DIR = "core_memory"
UPLOADS_DIR = "uploads"
USER_UPLOADS_DIR = "user_uploads"

# Ensure directories exist
for directory in [CORE_MEMORY_DIR, UPLOADS_DIR, USER_UPLOADS_DIR, "logs"]:
    os.makedirs(directory, exist_ok=True)

# CONTROLLED KNOWLEDGE SYSTEM
def load_knowledge_base():
    """Load ADMIN knowledge base - curated therapeutic resources"""
    try:
        if os.path.exists(KNOWLEDGE_BASE_FILE):
            with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
    return {"documents": [], "authorized_authors": [], "last_updated": None, "total_documents": 0}

def save_knowledge_base(knowledge_base):
    """Save ADMIN knowledge base"""
    try:
        knowledge_base["last_updated"] = datetime.now().isoformat()
        with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving knowledge base: {e}")

def extract_authors_from_text(text, filename):
    """Extract author names from PDF text and filename"""
    authors = set()
    
    # Common author extraction patterns
    author_patterns = [
        r"by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",  # "by John Smith"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\(\d{4}\)",  # "John Smith (2020)"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*-\s*",  # "John Smith - "
    ]
    
    # Extract from filename (common format: "AuthorName - Title.pdf")
    filename_match = re.match(r"([A-Za-z\s]+)\s*[-–—]\s*", filename)
    if filename_match:
        potential_author = filename_match.group(1).strip()
        if len(potential_author.split()) >= 2:  # At least first + last name
            authors.add(potential_author.title())
    
    # Extract from text content
    for pattern in author_patterns:
        matches = re.findall(pattern, text[:2000])  # Search first 2000 chars
        for match in matches:
            if len(match.split()) >= 2:  # At least first + last name
                authors.add(match.strip())
    
    return list(authors)

def extract_text_from_pdf(file_path):
    """TEMPORARY: Placeholder for PDF extraction - PyMuPDF disabled for deployment"""
    try:
        print(f"PDF extraction temporarily disabled for: {file_path}")
        # For now, return filename-based placeholder
        filename = os.path.basename(file_path)
        return f"PDF content placeholder for: {filename}\n\nPDF text extraction will be re-enabled after deployment with alternative library.\n\nFilename: {filename}"
    except Exception as e:
        error_msg = f"Error with PDF placeholder for {os.path.basename(file_path)}: {str(e)}"
        print(error_msg)
        return error_msg

def add_document_to_knowledge_base(file_path, filename, is_core=True):
    """Add document to ADMIN knowledge base and extract authors"""
    try:
        text_content = extract_text_from_pdf(file_path)
        
        # Extract authors from the document (works with placeholder text too)
        extracted_authors = extract_authors_from_text(text_content, filename)
        
        doc_info = {
            "filename": filename,
            "content": text_content,
            "added_date": datetime.now().isoformat(),
            "file_hash": hashlib.md5(open(file_path, 'rb').read()).hexdigest(),
            "is_core": is_core,
            "character_count": len(text_content),
            "type": "admin_therapeutic_resource",
            "extracted_authors": extracted_authors,
            "pdf_extraction_status": "placeholder_mode"  # Track that we're using placeholder
        }
        
        # Add to global knowledge base
        knowledge_base["documents"].append(doc_info)
        
        # Update authorized authors list
        if "authorized_authors" not in knowledge_base:
            knowledge_base["authorized_authors"] = []
        
        for author in extracted_authors:
            if author not in knowledge_base["authorized_authors"]:
                knowledge_base["authorized_authors"].append(author)
                print(f"Added authorized author: {author}")
        
        knowledge_base["total_documents"] = len(knowledge_base["documents"])
        save_knowledge_base(knowledge_base)
        print(f"Added {filename} to knowledge base ({len(text_content)} characters) - PDF extraction in placeholder mode")
        print(f"Extracted authors: {extracted_authors}")
        
        return doc_info
    except Exception as e:
        print(f"Error adding document to knowledge base: {e}")
        return None

def search_authorized_author_content(query, author_name):
    """Search for content from authorized authors only"""
    try:
        print(f"Searching for content by authorized author: {author_name}")
        
        # This is where you'd implement web search restricted to specific authors
        # For now, we'll return a placeholder
        search_query = f'"{author_name}" {query} therapeutic therapy systemic'
        
        # TODO: Implement controlled web search
        # Could use Google Scholar API, academic databases, or author-specific sites
        
        return f"[Controlled search for '{query}' by authorized author {author_name} would be implemented here]"
        
    except Exception as e:
        print(f"Error searching authorized author content: {e}")
        return None

def check_knowledge_gaps(user_query):
    """Check if query can be answered from knowledge base, or needs authorized author search"""
    # Simple keyword matching to determine if we have relevant content
    query_lower = user_query.lower()
    
    # Search through knowledge base content
    relevant_docs = []
    for doc in knowledge_base["documents"]:
        doc_content_lower = doc["content"].lower()
        
        # Simple relevance scoring
        relevance_score = 0
        query_words = query_lower.split()
        
        for word in query_words:
            if len(word) > 3:  # Skip short words
                relevance_score += doc_content_lower.count(word)
        
        if relevance_score > 0:
            relevant_docs.append((doc, relevance_score))
    
    # Sort by relevance
    relevant_docs.sort(key=lambda x: x[1], reverse=True)
    
    # If we have good coverage, use knowledge base only
    if relevant_docs and relevant_docs[0][1] > 5:
        return "knowledge_base_sufficient", relevant_docs[:3]
    
    # If coverage is poor, suggest authorized author search
    return "needs_authorized_search", relevant_docs

def build_controlled_context(session_id, user_query):
    """Build context ONLY from curated knowledge base + user docs"""
    context = ""
    
    # 1. Check knowledge gaps
    gap_status, relevant_docs = check_knowledge_gaps(user_query)
    
    # 2. ADMIN KNOWLEDGE BASE - Your curated content ONLY
    if relevant_docs:
        context += "\n=== CURATED THERAPEUTIC KNOWLEDGE ===\n"
        for doc, score in relevant_docs:
            context += f"From '{doc['filename']}':\n{doc['content'][:2000]}...\n\n"
    else:
        context += "\n=== AVAILABLE KNOWLEDGE ===\n"
        # If no specific relevance, use recent uploads
        for doc in knowledge_base["documents"][-3:]:
            context += f"From '{doc['filename']}':\n{doc['content'][:1500]}...\n\n"
    
    # 3. USER'S PERSONAL DOCUMENTS
    user_session = get_user_session(session_id)
    if user_session["personal_documents"]:
        context += "\n=== USER'S PERSONAL CONTEXT ===\n"
        for doc in user_session["personal_documents"]:
            context += f"From your document '{doc['filename']}':\n{doc['content'][:1000]}...\n\n"
    
    # 4. AUTHORIZED AUTHORS INFO
    if knowledge_base.get("authorized_authors"):
        context += f"\n=== AUTHORIZED THERAPEUTIC AUTHORS ===\n"
        context += f"You may reference these authorized authors: {', '.join(knowledge_base['authorized_authors'][:10])}\n"
        
        if gap_status == "needs_authorized_search":
            context += "\nNOTE: If the curated knowledge base doesn't fully address this query, you may mention that additional insights could be found from the authorized authors listed above.\n"
    
    return context[:10000], gap_status

knowledge_base = load_knowledge_base()

def is_admin_user():
    return session.get("is_admin", False)

def add_personal_document_to_session(file_path, filename, session_id):
    """Add document to USER's personal session"""
    try:
        text_content = extract_text_from_pdf(file_path)
        
        doc_info = {
            "filename": filename,
            "content": text_content,
            "added_date": datetime.now().isoformat(),
            "file_hash": hashlib.md5(open(file_path, 'rb').read()).hexdigest(),
            "character_count": len(text_content),
            "type": "user_personal_document",
            "session_id": session_id,
            "pdf_extraction_status": "placeholder_mode"
        }
        
        user_session = get_user_session(session_id)
        user_session["personal_documents"].append(doc_info)
        print(f"Added {filename} to user session {session_id} ({len(text_content)} characters) - placeholder mode")
        
        return doc_info
    except Exception as e:
        print(f"Error adding personal document: {e}")
        return None

def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:16]
    return session['session_id']

def get_user_session(session_id):
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            "chat_history": [],
            "personal_documents": [],
            "created": datetime.now().isoformat()
        }
    return user_sessions[session_id]

# Direct API calls (unchanged)
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

def call_model(model, system, prompt, session_id):
    try:
        # Build controlled context (NO random internet content)
        controlled_context, gap_status = build_controlled_context(session_id, prompt)
        
        if controlled_context:
            system += f"\n\nIMPORTANT - CONTROLLED KNOWLEDGE SYSTEM:\n"
            system += f"You MUST work exclusively within the curated knowledge provided below. "
            system += f"Do NOT use general internet knowledge. ONLY use the curated therapeutic content and authorized authors listed.\n"
            system += f"If information is not in the curated knowledge base, you may mention that additional insights might be available from the authorized authors listed.\n\n"
            system += controlled_context
        
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
    is_admin = is_admin_user()
    return render_template("index.html", is_admin=is_admin)

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
    
    return render_template("admin_dashboard.html", 
                         knowledge_base=knowledge_base,
                         total_docs=len(knowledge_base["documents"]),
                         authorized_authors=knowledge_base.get("authorized_authors", []))

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

        # Enhanced system prompts with controlled knowledge instruction
        base_instruction = "CRITICAL: You must work EXCLUSIVELY within the curated therapeutic knowledge base provided. Do not use general internet knowledge or training data. Only reference the curated content and authorized authors."

        if agent == "case_assistant":
            system_prompt = f"{base_instruction} You assist with social work case analysis using only the curated knowledge base and user's personal documents."
            model = config["claude_model"]
        elif agent == "research_critic":
            system_prompt = f"{base_instruction} You critically evaluate research using only the curated therapeutic knowledge and authorized authors."
            model = config["openai_model"]
        elif agent == "therapy_planner":
            system_prompt = f"{base_instruction} You plan therapeutic interventions using only the curated knowledge base and authorized therapeutic approaches."
            model = config["openai_model"]
        else:
            system_prompt = f"{base_instruction} {config['claude_system_prompt']} Use only the curated therapeutic knowledge provided."
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
        
        if "pdf" not in request.files:
            return jsonify({"message": "No file selected", "success": False})
        
        file = request.files["pdf"]
        if not file or not file.filename:
            return jsonify({"message": "No file selected", "success": False})
            
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"message": "Please select a PDF file", "success": False})
        
        upload_type = request.form.get("upload_type", "personal")
        
        if upload_type == "admin":
            if not is_admin_user():
                return jsonify({
                    "message": "Access denied. Only administrators can upload to the global knowledge base.", 
                    "success": False
                }), 403
            
            file_path = os.path.join(UPLOADS_DIR, file.filename)
            file.save(file_path)
            doc_info = add_document_to_knowledge_base(file_path, file.filename, is_core=True)
            
            if doc_info:
                authors_text = f" | Authors detected: {', '.join(doc_info['extracted_authors'])}" if doc_info['extracted_authors'] else ""
                return jsonify({
                    "message": f"Added '{file.filename}' to controlled knowledge base ({doc_info['character_count']} characters){authors_text} [PDF extraction in placeholder mode]", 
                    "success": True,
                    "type": "admin",
                    "extracted_authors": doc_info['extracted_authors']
                })
        else:
            file_path = os.path.join(USER_UPLOADS_DIR, f"{session_id}_{file.filename}")
            file.save(file_path)
            doc_info = add_personal_document_to_session(file_path, file.filename, session_id)
            
            if doc_info:
                return jsonify({
                    "message": f"Added '{file.filename}' to your personal session ({doc_info['character_count']} characters) [PDF extraction in placeholder mode]", 
                    "success": True,
                    "type": "personal"
                })
        
        return jsonify({"message": "Upload processing failed", "success": False})
        
    except Exception as e:
        print(f"Error in upload endpoint: {e}")
        return jsonify({"error": str(e), "success": False})

@app.route("/authorized-authors", methods=["GET"])
def get_authorized_authors():
    """Get list of authorized authors"""
    return jsonify({
        "authorized_authors": knowledge_base.get("authorized_authors", []),
        "total_authors": len(knowledge_base.get("authorized_authors", []))
    })

@app.route("/clear", methods=["POST"])
def clear():
    try:
        session_id = get_session_id()
        user_session = get_user_session(session_id)
        user_session["chat_history"] = []
        return jsonify({"message": "Chat history cleared (personal documents retained)"})
    except Exception as e:
        print(f"Error in clear endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/clear-documents", methods=["POST"])
def clear_documents():
    try:
        session_id = get_session_id()
        user_session = get_user_session(session_id)
        cleared_count = len(user_session["personal_documents"])
        user_session["personal_documents"] = []
        return jsonify({"message": f"Cleared {cleared_count} personal documents"})
    except Exception as e:
        print(f"Error in clear documents endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/log", methods=["GET"])
def get_log():
    session_id = get_session_id()
    user_session = get_user_session(session_id)
    return jsonify(user_session["chat_history"])

@app.route("/personal-documents", methods=["GET"])
def get_personal_documents():
    session_id = get_session_id()
    user_session = get_user_session(session_id)
    return jsonify({
        "personal_documents": [
            {
                "filename": doc["filename"],
                "added_date": doc["added_date"],
                "character_count": doc["character_count"]
            }
            for doc in user_session["personal_documents"]
        ]
    })

@app.route("/knowledge-base", methods=["GET"])
def get_knowledge_base():
    return jsonify({
        "total_documents": len(knowledge_base["documents"]),
        "last_updated": knowledge_base.get("last_updated"),
        "authorized_authors": knowledge_base.get("authorized_authors", []),
        "total_authors": len(knowledge_base.get("authorized_authors", [])),
        "documents": [{"filename": doc["filename"], "added_date": doc["added_date"], "character_count": doc.get("character_count", 0)} 
                     for doc in knowledge_base["documents"]]
    })

@app.route("/health")
def health():
    openai_works = False
    claude_works = False
    
    if openai_api_key:
        try:
            test_response = call_openai_direct("You are a test assistant.", "Hello")
            openai_works = not test_response.startswith("Error")
        except:
            openai_works = False
    
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
        "authorized_authors": len(knowledge_base.get("authorized_authors", [])),
        "controlled_knowledge_system": True,
        "pdf_support": "placeholder_mode",  # Indicates temporary placeholder
        "note": "Controlled knowledge system with placeholder PDF extraction (PyMuPDF disabled for deployment)"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    application = app
