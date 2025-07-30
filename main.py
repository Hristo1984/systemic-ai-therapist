import os
import json
import requests
import hashlib
import re
import sqlite3
import uuid
import secrets
from datetime import datetime, timedelta
from flask import Flask, request, render_template, jsonify, redirect, session
from dotenv import load_dotenv
import pdfplumber
import gc
from contextlib import contextmanager
from functools import wraps

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-this")

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")
admin_password = os.getenv("ADMIN_PASSWORD", "admin123")

if not openai_api_key:
    print("WARNING: OPENAI_API_KEY not found in environment variables")
if not claude_api_key:
    print("WARNING: CLAUDE_API_KEY not found in environment variables")

# File paths
DATABASE_FILE = "therapeutic_ai.db"
KNOWLEDGE_BASE_FILE = "core_memory/knowledge_base.json"
CORE_MEMORY_DIR = "core_memory"
UPLOADS_DIR = "uploads"
USER_UPLOADS_DIR = "user_uploads"

# Ensure directories exist
for directory in [CORE_MEMORY_DIR, UPLOADS_DIR, USER_UPLOADS_DIR, "logs"]:
    os.makedirs(directory, exist_ok=True)

# ================================
# USER AUTHENTICATION FUNCTIONS
# ================================

def hash_password(password):
    """Hash password with salt for secure storage"""
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}:{password_hash.hex()}"

def verify_password(password, hashed):
    """Verify password against hash"""
    try:
        salt, hash_hex = hashed.split(':')
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return hash_hex == password_hash.hex()
    except:
        return False

def require_login(f):
    """Decorator to require user login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or not session.get('authenticated'):
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

def get_authenticated_user():
    """Get current authenticated user or None"""
    if 'user_id' not in session or not session.get('authenticated'):
        return None
    
    with get_db_connection() as conn:
        user = conn.execute(
            'SELECT * FROM users WHERE id = ? AND is_active = 1', 
            (session['user_id'],)
        ).fetchone()
        return dict(user) if user else None

# ================================
# PERSISTENT DATABASE SYSTEM
# ================================

@contextmanager
def get_db_connection():
    """Context manager for database connections with proper cleanup"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize the enhanced database schema with authentication"""
    with get_db_connection() as conn:
        # Enhanced users table with authentication
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                username TEXT UNIQUE,
                password_hash TEXT,
                full_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                preferences JSON,
                therapy_goals TEXT,
                is_active BOOLEAN DEFAULT 1,
                is_verified BOOLEAN DEFAULT 1,
                subscription_type TEXT DEFAULT 'free',
                total_sessions INTEGER DEFAULT 0
            )
        ''')
        
        # Add authentication columns to existing users (migration)
        try:
            conn.execute('ALTER TABLE users ADD COLUMN email TEXT')
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            conn.execute('ALTER TABLE users ADD COLUMN username TEXT')
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute('ALTER TABLE users ADD COLUMN full_name TEXT')
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute('ALTER TABLE users ADD COLUMN is_verified BOOLEAN DEFAULT 1')
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute('ALTER TABLE users ADD COLUMN subscription_type TEXT DEFAULT "free"')
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute('ALTER TABLE users ADD COLUMN total_sessions INTEGER DEFAULT 0')
        except sqlite3.OperationalError:
            pass
        
        # Conversations table for persistent chat history
        conn.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                message_type TEXT, -- 'user' or 'assistant'
                content TEXT,
                agent_type TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_context JSON,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User documents table for persistent personal documents
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_documents (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                filename TEXT,
                content TEXT,
                file_hash TEXT,
                character_count INTEGER,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Therapy progress tracking
        conn.execute('''
            CREATE TABLE IF NOT EXISTS therapy_progress (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mood_score INTEGER,
                progress_notes TEXT,
                therapeutic_insights JSON,
                goals_updated JSON,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Admin knowledge base metadata
        conn.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_base_docs (
                id TEXT PRIMARY KEY,
                filename TEXT,
                file_hash TEXT,
                character_count INTEGER,
                extracted_authors JSON,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        ''')
        
        # User sessions for security
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                session_token TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Password reset tokens
        conn.execute('''
            CREATE TABLE IF NOT EXISTS password_resets (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                reset_token TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                used BOOLEAN DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        print("✅ Enhanced user authentication database initialized")

# ================================
# USER MANAGEMENT SYSTEM (HYBRID)
# ================================

def get_or_create_user(session_id=None):
    """Enhanced user management - supports both old anonymous and new authenticated users"""
    # If user is authenticated, return authenticated user
    if session.get('authenticated'):
        user = get_authenticated_user()
        if user:
            return user
    
    # Legacy support for existing anonymous users
    if 'user_id' not in session:
        if session_id:
            # Try to find existing user by session
            with get_db_connection() as conn:
                user = conn.execute(
                    'SELECT * FROM users WHERE id = ?', (session_id,)
                ).fetchone()
                
                if user:
                    session['user_id'] = user['id']
                    # Update last active
                    conn.execute(
                        'UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE id = ?',
                        (user['id'],)
                    )
                    conn.commit()
                    return dict(user)
        
        # Create new anonymous user (legacy support)
        user_id = str(uuid.uuid4())
        session['user_id'] = user_id
        
        with get_db_connection() as conn:
            conn.execute(
                'INSERT INTO users (id) VALUES (?)', (user_id,)
            )
            conn.commit()
            
        print(f"✅ Created new anonymous user: {user_id}")
        return {'id': user_id, 'created_at': datetime.now(), 'is_active': True}
    
    else:
        # Get existing user
        with get_db_connection() as conn:
            user = conn.execute(
                'SELECT * FROM users WHERE id = ?', (session['user_id'],)
            ).fetchone()
            
            if user:
                # Update last active
                conn.execute(
                    'UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE id = ?',
                    (user['id'],)
                )
                conn.commit()
                return dict(user)
            else:
                # User doesn't exist, create new one
                del session['user_id']
                return get_or_create_user()

# ================================
# CONVERSATION PERSISTENCE
# ================================

def save_conversation_message(user_id, message_type, content, agent_type=None):
    """Save conversation message to persistent storage"""
    message_id = str(uuid.uuid4())
    
    with get_db_connection() as conn:
        conn.execute('''
            INSERT INTO conversations (id, user_id, message_type, content, agent_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (message_id, user_id, message_type, content, agent_type))
        conn.commit()
    
    return message_id

def get_user_conversation_history(user_id, limit=50):
    """Retrieve user's conversation history"""
    with get_db_connection() as conn:
        messages = conn.execute('''
            SELECT * FROM conversations 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit)).fetchall()
    
    return [dict(msg) for msg in reversed(messages)]

def clear_user_conversation(user_id):
    """Clear user's conversation history"""
    with get_db_connection() as conn:
        conn.execute('DELETE FROM conversations WHERE user_id = ?', (user_id,))
        conn.commit()

# ================================
# MEMORY-EFFICIENT PDF PROCESSING
# ================================

def extract_text_from_pdf_efficient(file_path, max_size_mb=100):
    """Memory-efficient PDF extraction with size limits"""
    try:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"Processing PDF: {file_path} ({file_size:.1f}MB)")
        
        if file_size > max_size_mb:
            return f"PDF too large ({file_size:.1f}MB). Maximum size: {max_size_mb}MB"
        
        text_content = ""
        page_count = 0
        
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"PDF has {total_pages} pages")
            
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num + 1} ---\n"
                        text_content += page_text.strip() + "\n"
                        page_count += 1
                    
                    # Memory management: collect garbage every 25 pages
                    if page_num % 25 == 0:
                        gc.collect()
                        
                    # Progress indication for large files
                    if page_num % 50 == 0 and page_num > 0:
                        print(f"Processed {page_num}/{total_pages} pages...")
                        
                except Exception as e:
                    print(f"Error extracting page {page_num + 1}: {e}")
                    continue
        
        if not text_content.strip():
            return f"No text could be extracted from {os.path.basename(file_path)}"
        
        print(f"✅ Successfully extracted {len(text_content)} characters from {page_count} pages")
        return text_content
        
    except Exception as e:
        error_msg = f"Error extracting text from {os.path.basename(file_path)}: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg
    finally:
        # Force garbage collection
        gc.collect()

def extract_authors_from_text(text, filename):
    """Extract author names from PDF text and filename"""
    authors = set()
    
    # Common author extraction patterns
    author_patterns = [
        r"by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\(\d{4}\)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*[-–—]\s*",
        r"Author[s]?:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
    ]
    
    # Extract from filename
    filename_match = re.match(r"([A-Za-z\s]+)\s*[-–—]\s*", filename)
    if filename_match:
        potential_author = filename_match.group(1).strip()
        if len(potential_author.split()) >= 2:
            authors.add(potential_author.title())
    
    # Extract from text content (first 3000 chars for efficiency)
    search_text = text[:3000] if len(text) > 3000 else text
    for pattern in author_patterns:
        matches = re.findall(pattern, search_text)
        for match in matches:
            if len(match.split()) >= 2:
                authors.add(match.strip())
    
    return list(authors)

# ================================
# KNOWLEDGE BASE SYSTEM
# ================================

def load_knowledge_base():
    """Load admin knowledge base with fallback"""
    try:
        if os.path.exists(KNOWLEDGE_BASE_FILE):
            with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
    
    return {
        "documents": [],
        "authorized_authors": [],
        "last_updated": None,
        "total_documents": 0,
        "total_characters": 0
    }

def save_knowledge_base(knowledge_base):
    """Save admin knowledge base with metadata"""
    try:
        knowledge_base["last_updated"] = datetime.now().isoformat()
        knowledge_base["total_documents"] = len(knowledge_base["documents"])
        knowledge_base["total_characters"] = sum(
            len(doc.get("content", "")) for doc in knowledge_base["documents"]
        )
        
        with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Knowledge base saved: {knowledge_base['total_documents']} documents")
    except Exception as e:
        print(f"❌ Error saving knowledge base: {e}")

def add_document_to_knowledge_base(file_path, filename, is_core=True):
    """Add document to admin knowledge base with database tracking"""
    try:
        # Extract text efficiently
        text_content = extract_text_from_pdf_efficient(file_path)
        
        if "Error" in text_content or "too large" in text_content:
            return {"error": text_content}
        
        # Extract authors
        extracted_authors = extract_authors_from_text(text_content, filename)
        
        # Create document info
        file_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        doc_id = str(uuid.uuid4())
        
        doc_info = {
            "id": doc_id,
            "filename": filename,
            "content": text_content,
            "added_date": datetime.now().isoformat(),
            "file_hash": file_hash,
            "is_core": is_core,
            "character_count": len(text_content),
            "type": "admin_therapeutic_resource",
            "extracted_authors": extracted_authors,
            "pdf_extraction_status": "success"
        }
        
        # Add to knowledge base
        knowledge_base = load_knowledge_base()
        knowledge_base["documents"].append(doc_info)
        
        # Update authorized authors
        if "authorized_authors" not in knowledge_base:
            knowledge_base["authorized_authors"] = []
        
        for author in extracted_authors:
            if author not in knowledge_base["authorized_authors"]:
                knowledge_base["authorized_authors"].append(author)
                print(f"✅ Added authorized author: {author}")
        
        save_knowledge_base(knowledge_base)
        
        # Track in database
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO knowledge_base_docs 
                (id, filename, file_hash, character_count, extracted_authors)
                VALUES (?, ?, ?, ?, ?)
            ''', (doc_id, filename, file_hash, len(text_content), json.dumps(extracted_authors)))
            conn.commit()
        
        print(f"✅ Added {filename} to knowledge base ({len(text_content)} characters)")
        return doc_info
        
    except Exception as e:
        error_msg = f"Error adding document: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}
    finally:
        gc.collect()

# ================================
# USER DOCUMENT PERSISTENCE
# ================================

def add_personal_document_persistent(file_path, filename, user_id):
    """Add document to user's persistent personal collection"""
    try:
        text_content = extract_text_from_pdf_efficient(file_path, max_size_mb=50)
        
        if "Error" in text_content or "too large" in text_content:
            return {"error": text_content}
        
        file_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        doc_id = str(uuid.uuid4())
        
        # Save to database
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO user_documents 
                (id, user_id, filename, content, file_hash, character_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (doc_id, user_id, filename, text_content, file_hash, len(text_content)))
            conn.commit()
        
        doc_info = {
            "id": doc_id,
            "filename": filename,
            "content": text_content,
            "file_hash": file_hash,
            "character_count": len(text_content),
            "upload_date": datetime.now().isoformat(),
            "type": "user_personal_document"
        }
        
        print(f"✅ Added {filename} to user {user_id} ({len(text_content)} characters)")
        return doc_info
        
    except Exception as e:
        error_msg = f"Error adding personal document: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}
    finally:
        gc.collect()

def get_user_documents(user_id):
    """Get user's persistent personal documents"""
    with get_db_connection() as conn:
        docs = conn.execute('''
            SELECT * FROM user_documents 
            WHERE user_id = ? AND is_active = 1
            ORDER BY upload_date DESC
        ''', (user_id,)).fetchall()
    
    return [dict(doc) for doc in docs]

def clear_user_documents(user_id):
    """Clear user's personal documents"""
    with get_db_connection() as conn:
        result = conn.execute(
            'UPDATE user_documents SET is_active = 0 WHERE user_id = ?', (user_id,)
        )
        conn.commit()
        return result.rowcount

# ================================
# INTELLIGENT CONTEXT BUILDING
# ================================

def build_therapeutic_context(user_id, user_query, limit_kb_docs=3):
    """Build intelligent context from knowledge base + user history + personal docs"""
    context = ""
    
    # 1. Get user's conversation history for continuity
    conversation_history = get_user_conversation_history(user_id, limit=10)
    if conversation_history:
        context += "\n=== CONVERSATION CONTINUITY ===\n"
        context += "Previous conversation context (for therapeutic continuity):\n"
        for msg in conversation_history[-5:]:  # Last 5 messages
            role = "User" if msg['message_type'] == 'user' else "Therapist"
            context += f"{role}: {msg['content'][:200]}...\n"
    
    # 2. Search knowledge base for relevant content
    knowledge_base = load_knowledge_base()
    relevant_docs = []
    
    if knowledge_base["documents"]:
        query_words = user_query.lower().split()
        
        for doc in knowledge_base["documents"]:
            relevance_score = 0
            doc_content_lower = doc["content"].lower()
            
            for word in query_words:
                if len(word) > 3:
                    relevance_score += doc_content_lower.count(word)
            
            if relevance_score > 0:
                relevant_docs.append((doc, relevance_score))
        
        # Sort by relevance and take top docs
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        relevant_docs = relevant_docs[:limit_kb_docs]
    
    # Add knowledge base content
    if relevant_docs:
        context += "\n=== CURATED THERAPEUTIC KNOWLEDGE ===\n"
        for doc, score in relevant_docs:
            context += f"From '{doc['filename']}':\n{doc['content'][:2000]}...\n\n"
            
            # Track access in database
            with get_db_connection() as conn:
                conn.execute('''
                    UPDATE knowledge_base_docs 
                    SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE id = ?
                ''', (doc.get('id', ''),))
                conn.commit()
    
    # 3. Add user's personal documents
    user_docs = get_user_documents(user_id)
    if user_docs:
        context += "\n=== USER'S PERSONAL CONTEXT ===\n"
        for doc in user_docs[:3]:  # Limit to recent docs
            context += f"From your document '{doc['filename']}':\n{doc['content'][:1000]}...\n\n"
    
    # 4. Add authorized authors
    if knowledge_base.get("authorized_authors"):
        context += f"\n=== AUTHORIZED THERAPEUTIC AUTHORS ===\n"
        context += f"You may reference: {', '.join(knowledge_base['authorized_authors'][:10])}\n"
    
    return context[:15000]  # Limit total context size

# ================================
# API FUNCTIONS
# ================================

def call_claude_direct(system_prompt, user_message):
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": claude_api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}]
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
            print(f"Claude API error: {response.status_code}")
            return f"Error calling Claude API: {response.status_code}"
            
    except Exception as e:
        print(f"Error in Claude call: {e}")
        return f"Error: {str(e)}"

def call_openai_direct(system_prompt, user_message):
    try:
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
            print(f"OpenAI API error: {response.status_code}")
            return f"Error calling OpenAI API: {response.status_code}"
            
    except Exception as e:
        print(f"Error in OpenAI call: {e}")
        return f"Error: {str(e)}"

def call_model_with_context(model, system, prompt, user_id):
    """Enhanced model calling with persistent context"""
    try:
        # Build intelligent context
        context = build_therapeutic_context(user_id, prompt)
        
        if context:
            system += f"\n\nTHERAPEUTIC CONTEXT SYSTEM:\n"
            system += f"You have access to this user's therapeutic history and curated knowledge. "
            system += f"Provide continuity and reference previous conversations when appropriate.\n\n"
            system += context
        
        # Call appropriate model
        if "gpt" in model:
            if not openai_api_key:
                return "Error: OpenAI API key not configured"
            return call_openai_direct(system, prompt)
        else:
            if not claude_api_key:
                return "Error: Claude API key not configured"
            return call_claude_direct(system, prompt)
            
    except Exception as e:
        print(f"Error calling model: {e}")
        return f"Error: {str(e)}"

# ================================
# ADMIN FUNCTIONS
# ================================

def is_admin_user():
    return session.get("is_admin", False)

# ================================
# AUTHENTICATION ROUTES
# ================================

@app.route("/welcome", methods=["GET"])
def welcome():
    """Public landing page for new visitors"""
    return render_template("welcome.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration"""
    if request.method == "POST":
        try:
            data = request.get_json() if request.is_json else request.form
            
            email = data.get("email", "").strip().lower()
            username = data.get("username", "").strip()
            password = data.get("password", "")
            full_name = data.get("full_name", "").strip()
            
            # Validation
            if not email or not username or not password or not full_name:
                return jsonify({"error": "All fields are required"}), 400
            
            if len(password) < 8:
                return jsonify({"error": "Password must be at least 8 characters"}), 400
                
            if "@" not in email or "." not in email:
                return jsonify({"error": "Please enter a valid email address"}), 400
            
            # Check if user exists
            with get_db_connection() as conn:
                existing_user = conn.execute(
                    'SELECT id FROM users WHERE email = ? OR username = ?', 
                    (email, username)
                ).fetchone()
                
                if existing_user:
                    return jsonify({"error": "Email or username already exists"}), 400
                
                # Create new user
                user_id = str(uuid.uuid4())
                password_hash = hash_password(password)
                
                conn.execute('''
                    INSERT INTO users (id, email, username, password_hash, full_name, is_verified)
                    VALUES (?, ?, ?, ?, ?, 1)
                ''', (user_id, email, username, password_hash, full_name))
                conn.commit()
                
                # Log them in
                session['user_id'] = user_id
                session['authenticated'] = True
                session['username'] = username
                
                print(f"✅ New user registered: {username} ({email})")
                return jsonify({
                    "success": True, 
                    "message": "Account created successfully!",
                    "redirect": "/"
                })
                
        except Exception as e:
            print(f"Registration error: {e}")
            return jsonify({"error": "Registration failed. Please try again."}), 500
    
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """User login"""
    if request.method == "POST":
        try:
            data = request.get_json() if request.is_json else request.form
            
            email_or_username = data.get("email_or_username", "").strip().lower()
            password = data.get("password", "")
            
            if not email_or_username or not password:
                return jsonify({"error": "Email/username and password are required"}), 400
            
            # Find user
            with get_db_connection() as conn:
                user = conn.execute('''
                    SELECT * FROM users 
                    WHERE (email = ? OR username = ?) AND is_active = 1
                ''', (email_or_username, email_or_username)).fetchone()
                
                if not user or not user['password_hash'] or not verify_password(password, user['password_hash']):
                    return jsonify({"error": "Invalid email/username or password"}), 401
                
                # Update last active
                conn.execute(
                    'UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE id = ?',
                    (user['id'],)
                )
                conn.commit()
                
                # Log them in
                session['user_id'] = user['id']
                session['authenticated'] = True
                session['username'] = user['username']
                
                print(f"✅ User logged in: {user['username']}")
                return jsonify({
                    "success": True,
                    "message": f"Welcome back, {user['full_name'] or user['username']}!",
                    "redirect": "/"
                })
                
        except Exception as e:
            print(f"Login error: {e}")
            return jsonify({"error": "Login failed. Please try again."}), 500
    
    return render_template("login.html")

@app.route("/logout", methods=["GET", "POST"])
def logout():
    """User logout"""
    username = session.get('username', 'User')
    session.clear()
    
    if request.method == "POST":
        return jsonify({"success": True, "message": "Logged out successfully"})
    
    return redirect('/welcome')

@app.route("/profile", methods=["GET", "POST"])
@require_login
def profile():
    """User profile management"""
    user = get_authenticated_user()
    if not user:
        return redirect('/login')
    
    if request.method == "POST":
        try:
            data = request.get_json() if request.is_json else request.form
            
            full_name = data.get("full_name", "").strip()
            therapy_goals = data.get("therapy_goals", "").strip()
            
            with get_db_connection() as conn:
                conn.execute('''
                    UPDATE users 
                    SET full_name = ?, therapy_goals = ?
                    WHERE id = ?
                ''', (full_name, therapy_goals, user['id']))
                conn.commit()
            
            return jsonify({"success": True, "message": "Profile updated successfully"})
            
        except Exception as e:
            return jsonify({"error": "Failed to update profile"}), 500
    
    # Get user stats
    with get_db_connection() as conn:
        stats = conn.execute('''
            SELECT 
                COUNT(DISTINCT c.id) as total_conversations,
                COUNT(DISTINCT d.id) as total_documents,
                MAX(c.timestamp) as last_session
            FROM users u
            LEFT JOIN conversations c ON u.id = c.user_id
            LEFT JOIN user_documents d ON u.id = d.user_id AND d.is_active = 1
            WHERE u.id = ?
        ''', (user['id'],)).fetchone()
    
    return render_template("profile.html", user=user, stats=dict(stats))

# ================================
# MAIN APPLICATION ROUTES
# ================================

@app.route("/", methods=["GET"])
def root():
    """Root route - hybrid support for authenticated and anonymous users"""
    # If user is authenticated, go to chat
    if session.get('authenticated'):
        user = get_authenticated_user()
        if user:
            is_admin = is_admin_user()
            return render_template("index.html", 
                                 is_admin=is_admin, 
                                 user_id=user['id'],
                                 username=user.get('username', ''),
                                 full_name=user.get('full_name', user.get('username', '')))
    
    # Legacy support for anonymous users OR redirect to welcome
    if 'user_id' in session:
        # Existing anonymous user - continue their session
        user = get_or_create_user()
        is_admin = is_admin_user()
        return render_template("index.html", is_admin=is_admin, user_id=user['id'])
    else:
        # New visitor - redirect to welcome page
        return redirect('/welcome')

@app.route("/admin", methods=["GET", "POST"])
def admin():
    """Admin panel"""
    if request.method == "POST":
        password = request.form.get("password")
        if password == admin_password:
            session["is_admin"] = True
            return redirect("/admin")
        else:
            return render_template("admin_login.html", error="Invalid password")
    
    if not session.get("is_admin"):
        return render_template("admin_login.html")
    
    knowledge_base = load_knowledge_base()
    return render_template("admin_dashboard.html", 
                         knowledge_base=knowledge_base,
                         total_docs=len(knowledge_base["documents"]),
                         authorized_authors=knowledge_base.get("authorized_authors", []))

@app.route("/admin/logout", methods=["GET"])
def admin_logout():
    """Admin logout route"""
    session.pop("is_admin", None)
    return redirect("/")

@app.route("/chat", methods=["POST"])
def chat():
    """Chat endpoint - supports both authenticated and anonymous users"""
    try:
        user = get_or_create_user()
        if not user:
            return jsonify({"error": "User session required"}), 401
        
        user_id = user['id']
        
        # Increment session counter for authenticated users
        if session.get('authenticated'):
            with get_db_connection() as conn:
                conn.execute(
                    'UPDATE users SET total_sessions = total_sessions + 1 WHERE id = ?',
                    (user_id,)
                )
                conn.commit()
        
        data = request.get_json()
        user_input = data.get("user_input", "")
        agent = data.get("agent", "")

        print(f"Chat request - User: {user_id}, Input: {user_input[:100]}..., Agent: {agent}")

        # Save user message
        save_conversation_message(user_id, "user", user_input)

        # Build system prompt based on agent
        user_name = user.get('full_name') or user.get('username') or 'User'
        base_instruction = f"You are a professional therapeutic AI supporting {user_name}. You have access to curated knowledge and conversation history. Provide continuity and reference previous conversations when appropriate."

        agent_prompts = {
            "case_assistant": f"{base_instruction} You assist with social work case analysis.",
            "research_critic": f"{base_instruction} You critically evaluate research using evidence-based approaches.",
            "therapy_planner": f"{base_instruction} You plan therapeutic interventions using proven methodologies.",
            "therapist": f"{base_instruction} {config.get('claude_system_prompt', 'You provide therapeutic support.')}"
        }

        system_prompt = agent_prompts.get(agent, agent_prompts["therapist"])
        model = config.get("claude_model", "claude-3-haiku-20240307")

        # Get AI response with full context
        response_text = call_model_with_context(model, system_prompt, user_input, user_id)
        
        # Save AI response
        save_conversation_message(user_id, "assistant", response_text, agent)

        return jsonify({
            "response": response_text, 
            "user_id": user_id,
            "username": user.get('username', '')
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    """File upload endpoint - supports both authenticated and anonymous users"""
    try:
        user = get_or_create_user()
        if not user:
            return jsonify({"error": "User session required"}), 401
        
        user_id = user['id']
        
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
                    "message": "Access denied. Admin privileges required.", 
                    "success": False
                }), 403
            
            # Admin upload to global knowledge base
            file_path = os.path.join(UPLOADS_DIR, file.filename)
            file.save(file_path)
            
            doc_info = add_document_to_knowledge_base(file_path, file.filename, is_core=True)
            
            if "error" in doc_info:
                return jsonify({"message": doc_info["error"], "success": False})
            
            authors_text = f" | Authors: {', '.join(doc_info['extracted_authors'])}" if doc_info['extracted_authors'] else ""
            return jsonify({
                "message": f"✅ Added '{file.filename}' to global knowledge base ({doc_info['character_count']} characters){authors_text}", 
                "success": True,
                "type": "admin",
                "extracted_authors": doc_info['extracted_authors']
            })
        else:
            # User personal document
            file_path = os.path.join(USER_UPLOADS_DIR, f"{user_id}_{file.filename}")
            file.save(file_path)
            
            doc_info = add_personal_document_persistent(file_path, file.filename, user_id)
            
            if "error" in doc_info:
                return jsonify({"message": doc_info["error"], "success": False})
            
            return jsonify({
                "message": f"✅ Added '{file.filename}' to your persistent documents ({doc_info['character_count']} characters)", 
                "success": True,
                "type": "personal"
            })
        
    except Exception as e:
        print(f"Error in upload endpoint: {e}")
        return jsonify({"error": str(e), "success": False})

@app.route("/clear", methods=["POST"])
def clear():
    """Clear chat history"""
    try:
        user = get_or_create_user()
        if not user:
            return jsonify({"error": "User session required"}), 401
        
        user_id = user['id']
        clear_user_conversation(user_id)
        return jsonify({"message": "Chat history cleared (documents and progress retained)"})
    except Exception as e:
        print(f"Error in clear endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/clear-documents", methods=["POST"])
def clear_documents():
    """Clear personal documents"""
    try:
        user = get_or_create_user()
        if not user:
            return jsonify({"error": "User session required"}), 401
        
        user_id = user['id']
        cleared_count = clear_user_documents(user_id)
        return jsonify({"message": f"Cleared {cleared_count} personal documents"})
    except Exception as e:
        print(f"Error in clear documents endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/conversation-history", methods=["GET"])
def get_conversation_history():
    """Get user's persistent conversation history"""
    try:
        user = get_or_create_user()
        if not user:
            return jsonify({"error": "User session required"}), 401
        
        user_id = user['id']
        history = get_user_conversation_history(user_id, limit=100)
        return jsonify({"conversation_history": history, "user_id": user_id})
    except Exception as e:
        print(f"Error in conversation history endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/personal-documents", methods=["GET"])
def get_personal_documents():
    """Get user's persistent personal documents"""
    try:
        user = get_or_create_user()
        if not user:
            return jsonify({"error": "User session required"}), 401
        
        user_id = user['id']
        docs = get_user_documents(user_id)
        
        # Return summary info (not full content for performance)
        doc_summaries = [
            {
                "id": doc["id"],
                "filename": doc["filename"],
                "upload_date": doc["upload_date"],
                "character_count": doc["character_count"]
            }
            for doc in docs
        ]
        
        return jsonify({"personal_documents": doc_summaries, "user_id": user_id})
    except Exception as e:
        print(f"Error in personal documents endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/knowledge-base", methods=["GET"])
def get_knowledge_base():
    """Get knowledge base status"""
    try:
        knowledge_base = load_knowledge_base()
        
        # Get database stats
        with get_db_connection() as conn:
            total_users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
            active_users = conn.execute(
                'SELECT COUNT(*) as count FROM users WHERE last_active > datetime("now", "-30 days")'
            ).fetchone()['count']
            total_conversations = conn.execute('SELECT COUNT(*) as count FROM conversations').fetchone()['count']
        
        return jsonify({
            "total_documents": len(knowledge_base["documents"]),
            "total_characters": knowledge_base.get("total_characters", 0),
            "last_updated": knowledge_base.get("last_updated"),
            "authorized_authors": knowledge_base.get("authorized_authors", []),
            "total_authors": len(knowledge_base.get("authorized_authors", [])),
            "system_stats": {
                "total_users": total_users,
                "active_users_30d": active_users,
                "total_conversations": total_conversations
            }
        })
    except Exception as e:
        print(f"Error in knowledge base endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/user-stats", methods=["GET"])
def get_user_stats():
    """Get current user's statistics"""
    try:
        user = get_or_create_user()
        if not user:
            return jsonify({"error": "User session required"}), 401
        
        user_id = user['id']
        
        with get_db_connection() as conn:
            conversation_count = conn.execute(
                'SELECT COUNT(*) as count FROM conversations WHERE user_id = ?', (user_id,)
            ).fetchone()['count']
            
            document_count = conn.execute(
                'SELECT COUNT(*) as count FROM user_documents WHERE user_id = ? AND is_active = 1', (user_id,)
            ).fetchone()['count']
            
            days_active = conn.execute('''
                SELECT CAST((julianday('now') - julianday(created_at)) AS INTEGER) as days
                FROM users WHERE id = ?
            ''', (user_id,)).fetchone()['days']
        
        return jsonify({
            "user_id": user_id,
            "conversation_messages": conversation_count,
            "personal_documents": document_count,
            "days_active": days_active,
            "created_at": user.get('created_at'),
            "last_active": user.get('last_active')
        })
    except Exception as e:
        print(f"Error in user stats endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/log", methods=["GET"])
def get_log():
    """Download user's conversation history"""
    try:
        user = get_or_create_user()
        if not user:
            return jsonify({"error": "User session required"}), 401
        
        user_id = user['id']
        
        # Get full conversation history
        history = get_user_conversation_history(user_id, limit=1000)
        
        # Get user documents
        docs = get_user_documents(user_id)
        
        # Format for download
        export_data = {
            "user_id": user_id,
            "username": user.get('username', ''),
            "full_name": user.get('full_name', ''),
            "export_date": datetime.now().isoformat(),
            "conversation_history": history,
            "personal_documents": [
                {
                    "filename": doc["filename"],
                    "upload_date": doc["upload_date"],
                    "character_count": doc["character_count"]
                }
                for doc in docs
            ],
            "stats": {
                "total_messages": len(history),
                "total_documents": len(docs)
            }
        }
        
        return jsonify(export_data)
    except Exception as e:
        print(f"Error in log endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    """System health check with memory usage"""
    try:
        # Test API connections
        openai_works = False
        claude_works = False
        
        if openai_api_key:
            try:
                test_response = call_openai_direct("You are a test.", "Hello")
                openai_works = not test_response.startswith("Error")
            except:
                pass
        
        if claude_api_key:
            try:
                test_response = call_claude_direct("You are a test.", "Hello")
                claude_works = not test_response.startswith("Error")
            except:
                pass
        
        # Database stats
        with get_db_connection() as conn:
            total_users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
            authenticated_users = conn.execute('SELECT COUNT(*) as count FROM users WHERE email IS NOT NULL').fetchone()['count']
            total_conversations = conn.execute('SELECT COUNT(*) as count FROM conversations').fetchone()['count']
            total_user_docs = conn.execute('SELECT COUNT(*) as count FROM user_documents WHERE is_active = 1').fetchone()['count']
        
        knowledge_base = load_knowledge_base()
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "apis": {
                "openai_configured": openai_api_key is not None,
                "openai_working": openai_works,
                "claude_configured": claude_api_key is not None,
                "claude_working": claude_works
            },
            "database": {
                "total_users": total_users,
                "authenticated_users": authenticated_users,
                "anonymous_users": total_users - authenticated_users,
                "total_conversations": total_conversations,
                "total_user_documents": total_user_docs,
                "knowledge_base_documents": len(knowledge_base["documents"]),
                "authorized_authors": len(knowledge_base.get("authorized_authors", []))
            },
            "features": {
                "user_authentication": True,
                "anonymous_sessions": True,
                "persistent_memory": True,
                "conversation_continuity": True,
                "personal_documents": True,
                "knowledge_base": True,
                "pdf_extraction": "pdfplumber",
                "controlled_knowledge": True
            },
            "memory_info": {
                "knowledge_base_size_mb": round(knowledge_base.get("total_characters", 0) / 1024 / 1024, 2),
                "database_file": DATABASE_FILE,
                "pdf_max_size_mb": 100
            }
        })
    except Exception as e:
        print(f"Error in health endpoint: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

# ================================
# DATABASE MAINTENANCE
# ================================

@app.route("/admin/cleanup", methods=["POST"])
def admin_cleanup():
    """Admin endpoint for database maintenance"""
    if not is_admin_user():
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        days_old = int(request.json.get("days_old", 90))
        
        with get_db_connection() as conn:
            # Clean old inactive users (anonymous only)
            old_users = conn.execute('''
                DELETE FROM users 
                WHERE last_active < datetime('now', '-{} days') 
                AND email IS NULL
                AND id NOT IN (SELECT DISTINCT user_id FROM conversations WHERE timestamp > datetime('now', '-30 days'))
            '''.format(days_old))
            
            # Clean orphaned conversations
            orphaned_convs = conn.execute('''
                DELETE FROM conversations 
                WHERE user_id NOT IN (SELECT id FROM users)
            ''')
            
            # Clean orphaned documents
            orphaned_docs = conn.execute('''
                DELETE FROM user_documents 
                WHERE user_id NOT IN (SELECT id FROM users)
            ''')
            
            conn.commit()
        
        # Force garbage collection
        gc.collect()
        
        return jsonify({
            "message": "Database cleanup completed",
            "removed": {
                "old_anonymous_users": old_users.rowcount,
                "orphaned_conversations": orphaned_convs.rowcount,
                "orphaned_documents": orphaned_docs.rowcount
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================================
# INITIALIZATION AND STARTUP
# ================================

def initialize_system():
    """Initialize the therapeutic AI system with hybrid authentication"""
    print("🚀 Initializing Therapeutic AI with Hybrid Authentication...")
    
    # Initialize database
    init_database()
    
    # Load knowledge base
    knowledge_base = load_knowledge_base()
    print(f"📚 Knowledge base loaded: {len(knowledge_base['documents'])} documents")
    
    # Check database health
    with get_db_connection() as conn:
        total_users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
        authenticated_users = conn.execute('SELECT COUNT(*) as count FROM users WHERE email IS NOT NULL').fetchone()['count']
        total_conversations = conn.execute('SELECT COUNT(*) as count FROM conversations').fetchone()['count']
        print(f"👥 Database: {total_users} users ({authenticated_users} registered, {total_users - authenticated_users} anonymous), {total_conversations} conversations")
    
    print("✅ Therapeutic AI system with hybrid authentication initialized successfully!")

if __name__ == "__main__":
    initialize_system()
    app.run(host="0.0.0.0", port=5000, debug=True)
else:
    initialize_system()
    application = app
