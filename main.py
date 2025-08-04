import os
import json
import requests
import hashlib
import re
import sqlite3
import uuid
import secrets
import tempfile
import threading
import time
import resource
import signal
import sys
from datetime import datetime, timedelta
from flask import Flask, request, render_template, jsonify, redirect, session
from dotenv import load_dotenv
import pdfplumber
import gc
from contextlib import contextmanager
from functools import wraps
from typing import Dict, List, Tuple, Any
from collections import Counter
import traceback

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-this")

# CRITICAL: Memory and upload safety settings
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max request
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
app.config['JSON_AS_ASCII'] = False

# Memory limit enforcement (2GB limit for Render Standard tier)
def limit_memory():
    try:
        # Set memory limit to 2GB (Render Standard tier safe limit)
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024))
        print("✅ Memory limit set to 2GB")
    except Exception as e:
        print(f"⚠️ Could not set memory limit: {e}")

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
    print(f"✅ Created/verified directory: {directory}")

# ================================
# ERROR HANDLERS FOR MEMORY ISSUES
# ================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return jsonify({
        "error": "File too large. Maximum size is 100MB.",
        "success": False
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    """Handle server errors with memory cleanup"""
    print(f"🚨 Server error: {error}")
    gc.collect()  # Force cleanup on error
    return jsonify({
        "error": "Internal server error. Please try with a smaller file.",
        "success": False
    }), 500

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
                message_type TEXT,
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
# IMPROVED PDF PROCESSING
# ================================

def extract_text_from_pdf_improved(file_path, max_size_mb=200):
    """Improved PDF extraction with better error handling and progress"""
    try:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"🔍 PDF PROCESSING: {file_path} ({file_size:.1f}MB)")
        
        if file_size > max_size_mb:
            return f"PDF too large ({file_size:.1f}MB). Maximum size: {max_size_mb}MB"
        
        text_content = ""
        page_count = 0
        
        # Force garbage collection before starting
        gc.collect()
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"📄 PDF has {total_pages} pages")
                
                # Process all pages up to reasonable limit
                max_pages = min(total_pages, 2000)  # Increased from 1000
                if total_pages > max_pages:
                    print(f"⚠️ Large PDF detected - processing first {max_pages} pages only")
                
                for page_num in range(max_pages):
                    try:
                        page = pdf.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            # Better page separation
                            text_content += f"\n\n=== PAGE {page_num + 1} ===\n"
                            text_content += page_text.strip() + "\n"
                            page_count += 1
                        
                        # Memory management every 50 pages
                        if page_num % 50 == 0:
                            gc.collect()
                            
                        # Progress indication every 100 pages
                        if page_num % 100 == 0 and page_num > 0:
                            print(f"📄 Processed {page_num}/{max_pages} pages...")
                            
                        # Text size safety check (100MB text limit)
                        if len(text_content) > 100 * 1024 * 1024:
                            print("⚠️ Text content limit reached - stopping extraction")
                            break
                            
                    except Exception as e:
                        print(f"⚠️ Error extracting page {page_num + 1}: {e}")
                        continue
                        
        except Exception as pdf_error:
            print(f"❌ PDF processing error: {pdf_error}")
            return f"Error processing PDF: {str(pdf_error)}"
        
        if not text_content.strip():
            return f"❌ No text could be extracted from {os.path.basename(file_path)}"
        
        print(f"✅ Successfully extracted {len(text_content):,} characters from {page_count} pages")
        return text_content
        
    except Exception as e:
        error_msg = f"Error extracting text from {os.path.basename(file_path)}: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg
    finally:
        # ALWAYS force garbage collection
        gc.collect()

def extract_authors_from_text_improved(text, filename):
    """Improved author extraction with more patterns"""
    authors = set()
    
    # Enhanced author extraction patterns
    author_patterns = [
        r"by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\(\d{4}\)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*[-–—]\s*",
        r"Author[s]?:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        r"Written by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\n",  # First line author
        r"Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
    ]
    
    # Extract from filename first (higher priority)
    filename_patterns = [
        r"([A-Za-z\s]+)\s*[-–—]\s*",  # "Author Name - Book Title"
        r"^([A-Za-z\s]+)\s+\(",  # "Author Name (Year)"
        r"^([A-Za-z\s]+)\s+\d{4}",  # "Author Name 2023"
    ]
    
    for pattern in filename_patterns:
        filename_match = re.match(pattern, filename)
        if filename_match:
            potential_author = filename_match.group(1).strip()
            if len(potential_author.split()) >= 2 and len(potential_author) < 50:
                authors.add(potential_author.title())
    
    # Extract from text content (first 5000 chars for efficiency)
    search_text = text[:5000] if len(text) > 5000 else text
    for pattern in author_patterns:
        matches = re.findall(pattern, search_text, re.MULTILINE)
        for match in matches:
            if isinstance(match, str) and len(match.split()) >= 2 and len(match) < 50:
                authors.add(match.strip())
    
    return list(authors)

# ================================
# IMPROVED KNOWLEDGE BASE SYSTEM
# ================================

def load_knowledge_base():
    """Load admin knowledge base with better error handling"""
    try:
        if os.path.exists(KNOWLEDGE_BASE_FILE):
            with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
                kb = json.load(f)
                print(f"📚 Loaded knowledge base: {len(kb.get('documents', []))} documents")
                return kb
    except Exception as e:
        print(f"❌ Error loading knowledge base: {e}")
    
    # Return empty knowledge base with proper structure
    empty_kb = {
        "documents": [],
        "authorized_authors": [],
        "last_updated": None,
        "total_documents": 0,
        "total_characters": 0
    }
    
    # Try to create the file
    try:
        save_knowledge_base(empty_kb)
    except Exception as e:
        print(f"⚠️ Could not create knowledge base file: {e}")
    
    return empty_kb

def save_knowledge_base(knowledge_base):
    """Save admin knowledge base with metadata"""
    try:
        knowledge_base["last_updated"] = datetime.now().isoformat()
        knowledge_base["total_documents"] = len(knowledge_base["documents"])
        
        # Calculate total characters from all documents
        total_chars = 0
        for doc in knowledge_base["documents"]:
            total_chars += len(doc.get("content", ""))
        
        knowledge_base["total_characters"] = total_chars
        
        with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Knowledge base saved: {knowledge_base['total_documents']} documents, {total_chars:,} characters")
    except Exception as e:
        print(f"❌ Error saving knowledge base: {e}")

def add_document_to_knowledge_base(file_path, filename, is_core=True):
    """Add document to admin knowledge base - IMPROVED VERSION"""
    try:
        print(f"🔍 ADMIN UPLOAD: Processing {filename}")
        
        # Extract text with improved method
        text_content = extract_text_from_pdf_improved(file_path)
        
        if text_content.startswith("Error") or "too large" in text_content:
            return {"error": text_content}
        
        # Extract authors with improved method
        extracted_authors = extract_authors_from_text_improved(text_content, filename)
        
        # Create document info
        file_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        doc_id = str(uuid.uuid4())
        
        doc_info = {
            "id": doc_id,
            "filename": filename,
            "content": text_content,  # Store full text
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
        
        print(f"✅ Added '{filename}' to knowledge base ({len(text_content):,} characters)")
        return doc_info
        
    except Exception as e:
        error_msg = f"Error adding document: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return {"error": error_msg}
    finally:
        gc.collect()

# ================================
# USER DOCUMENT PERSISTENCE (IMPROVED)
# ================================

def add_personal_document_improved(file_path, filename, user_id):
    """Add document to user's persistent personal collection - IMPROVED VERSION"""
    try:
        print(f"🔍 PERSONAL UPLOAD: Processing {filename} for user {user_id}")
        
        text_content = extract_text_from_pdf_improved(file_path, max_size_mb=100)
        
        if text_content.startswith("Error") or "too large" in text_content:
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
        
        print(f"✅ Added '{filename}' to user {user_id} ({len(text_content):,} characters)")
        return doc_info
        
    except Exception as e:
        error_msg = f"Error adding personal document: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
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
# COMPLETELY REWRITTEN CONTEXT BUILDING
# ================================

def build_therapeutic_context_improved(user_id, user_query, limit_kb_docs=5):
    """COMPLETELY IMPROVED context building with advanced search and debugging"""
    context = ""
    
    print(f"🔍 CONTEXT BUILD: Query = '{user_query[:100]}...'")
    print(f"🔍 CONTEXT BUILD: User ID = {user_id}")
    
    # 1. Get user's conversation history for continuity
    conversation_history = get_user_conversation_history(user_id, limit=10)
    if conversation_history:
        context += "\n=== RECENT CONVERSATION CONTEXT ===\n"
        context += "Previous conversation for therapeutic continuity:\n"
        for msg in conversation_history[-5:]:  # Last 5 messages
            role = "User" if msg['message_type'] == 'user' else "Therapist"
            context += f"{role}: {msg['content'][:300]}...\n"
    
    # 2. ADVANCED KNOWLEDGE BASE SEARCH
    knowledge_base = load_knowledge_base()
    print(f"🔍 CONTEXT BUILD: Knowledge base has {len(knowledge_base['documents'])} documents")
    
    if not knowledge_base["documents"]:
        print("❌ CONTEXT BUILD: No documents in knowledge base!")
        context += "\n=== NO KNOWLEDGE BASE AVAILABLE ===\n"
        context += "No therapeutic resources have been uploaded to the knowledge base.\n"
        return context
    
    # Advanced search algorithm
    query_words = user_query.lower().split()
    book_title_patterns = [
        r'"([^"]+)"',  # "Book Title in quotes"
        r"book[:\s]+([A-Za-z\s]+)",  # "book: Title" or "book Title"
        r"titled?\s+([A-Za-z\s]+)",  # "titled Title"
    ]
    
    # Extract potential book titles from query
    potential_titles = []
    for pattern in book_title_patterns:
        matches = re.findall(pattern, user_query, re.IGNORECASE)
        potential_titles.extend(matches)
    
    print(f"🔍 CONTEXT BUILD: Search words = {query_words}")
    print(f"🔍 CONTEXT BUILD: Potential book titles = {potential_titles}")
    
    # Score each document
    scored_docs = []
    
    for i, doc in enumerate(knowledge_base["documents"]):
        filename = doc.get("filename", "").lower()
        doc_content = doc.get("content", "").lower()
        
        # Scoring system
        title_exact_match = 0
        filename_word_matches = 0
        content_relevance = 0
        
        # 1. Check for exact title matches (highest priority)
        for title in potential_titles:
            if title.lower() in filename:
                title_exact_match += 50
                print(f"✅ EXACT TITLE MATCH: '{title}' found in '{doc.get('filename')}'")
        
        # 2. Check filename word matches
        for word in query_words:
            if len(word) > 2:  # Lowered from 3 to 2
                if word in filename:
                    filename_word_matches += 10
                    print(f"✅ FILENAME MATCH: '{word}' in '{doc.get('filename')}'")
        
        # 3. Content relevance (for non-title searches)
        if not potential_titles:  # Only do content search if not looking for specific titles
            for word in query_words:
                if len(word) > 3:
                    content_relevance += min(doc_content.count(word), 10)  # Cap at 10 to prevent spam
        
        total_score = title_exact_match + filename_word_matches + content_relevance
        
        if total_score > 0:
            scored_docs.append({
                "doc": doc,
                "score": total_score,
                "title_match": title_exact_match,
                "filename_match": filename_word_matches,
                "content_match": content_relevance
            })
        
        print(f"🔍 CONTEXT BUILD: '{doc.get('filename')}' = {total_score} points (title:{title_exact_match}, filename:{filename_word_matches}, content:{content_relevance})")
    
    # Sort by relevance
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    top_docs = scored_docs[:limit_kb_docs]
    
    print(f"🔍 CONTEXT BUILD: Selected {len(top_docs)} documents")
    
    # Build context from selected documents
    if top_docs:
        context += "\n=== THERAPEUTIC KNOWLEDGE BASE ===\n"
        context += f"Found {len(top_docs)} relevant therapeutic resources:\n\n"
        
        for i, doc_data in enumerate(top_docs):
            doc = doc_data["doc"]
            score = doc_data["score"]
            
            # Use much more content per document
            content_to_include = min(len(doc.get("content", "")), 1000000)  # Up to 1MB per doc
            doc_content = doc.get("content", "")[:content_to_include]
            
            context += f"=== RESOURCE {i+1}: '{doc['filename']}' (Relevance: {score}) ===\n"
            context += f"{doc_content}\n\n"
            
            print(f"✅ CONTEXT BUILD: Added {len(doc_content):,} chars from '{doc['filename']}'")
            
            # Track access in database
            try:
                with get_db_connection() as conn:
                    conn.execute('''
                        UPDATE knowledge_base_docs 
                        SET last_accessed = CURRENT_TIMESTAMP, access_count = access_count + 1
                        WHERE id = ?
                    ''', (doc.get('id', ''),))
                    conn.commit()
            except Exception as e:
                print(f"⚠️ Could not update access tracking: {e}")
    
    else:
        print("❌ CONTEXT BUILD: No relevant documents found!")
        context += "\n=== NO RELEVANT KNOWLEDGE FOUND ===\n"
        context += f"Search query: '{user_query}'\n"
        context += "Available documents in knowledge base:\n"
        for doc in knowledge_base["documents"][:10]:  # Show first 10
            context += f"- '{doc.get('filename', 'Unknown')}'\n"
    
    # 3. Add user's personal documents
    user_docs = get_user_documents(user_id)
    if user_docs:
        context += "\n=== YOUR PERSONAL DOCUMENTS ===\n"
        context += f"You have {len(user_docs)} personal documents:\n\n"
        for doc in user_docs[:3]:  # Include up to 3 recent docs
            personal_content = doc['content'][:500000]  # 500k chars per personal doc
            context += f"=== YOUR DOCUMENT: '{doc['filename']}' ===\n"
            context += f"{personal_content}\n\n"
            print(f"✅ CONTEXT BUILD: Added {len(personal_content):,} chars from personal doc '{doc['filename']}'")
    
    # 4. Add authorized authors list
    if knowledge_base.get("authorized_authors"):
        context += f"\n=== AUTHORIZED THERAPEUTIC AUTHORS ===\n"
        context += f"Therapeutic experts you can reference: {', '.join(knowledge_base['authorized_authors'][:15])}\n"
    
    # 5. Add metadata
    context += f"\n=== CONTEXT METADATA ===\n"
    context += f"Total knowledge base documents: {len(knowledge_base['documents'])}\n"
    context += f"User's personal documents: {len(user_docs) if user_docs else 0}\n"
    context += f"Context built at: {datetime.now().isoformat()}\n"
    
    final_context_size = len(context)
    print(f"🔍 CONTEXT BUILD: Final context = {final_context_size:,} characters")
    
    # Return full context - let Claude handle the size
    return context

# ================================
# IMPROVED API FUNCTIONS
# ================================

def call_claude_improved(system_prompt, user_message):
    """Improved Claude API call with better error handling"""
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": claude_api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Use larger context window
        data = {
            "model": config.get("claude_model", "claude-3-haiku-20240307"),
            "max_tokens": config.get("max_tokens", 2048),
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}]
        }
        
        print(f"🔍 CLAUDE API: Sending {len(system_prompt):,} system chars + {len(user_message):,} user chars")
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=60  # Increased timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result["content"][0]["text"]
            print(f"✅ CLAUDE API: Received {len(response_text):,} chars response")
            return response_text
        else:
            error_msg = f"Claude API error: {response.status_code} - {response.text}"
            print(f"❌ CLAUDE API: {error_msg}")
            return f"Error calling Claude API: {response.status_code}"
            
    except Exception as e:
        error_msg = f"Error in Claude call: {str(e)}"
        print(f"❌ CLAUDE API: {error_msg}")
        return error_msg

def call_openai_improved(system_prompt, user_message):
    """Improved OpenAI API call with better error handling"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        
        data = {
            "model": config.get("openai_model", "gpt-4"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": config.get("max_tokens", 2048)
        }
        
        print(f"🔍 OPENAI API: Sending {len(system_prompt):,} system chars + {len(user_message):,} user chars")
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            print(f"✅ OPENAI API: Received {len(response_text):,} chars response")
            return response_text
        else:
            error_msg = f"OpenAI API error: {response.status_code} - {response.text}"
            print(f"❌ OPENAI API: {error_msg}")
            return f"Error calling OpenAI API: {response.status_code}"
            
    except Exception as e:
        error_msg = f"Error in OpenAI call: {str(e)}"
        print(f"❌ OPENAI API: {error_msg}")
        return error_msg

def call_model_with_improved_context(model, system, prompt, user_id):
    """Enhanced model calling with improved context and debugging"""
    try:
        print(f"🔍 MODEL CALL: Model = {model}, User = {user_id}")
        
        # Build intelligent context with improved search
        context = build_therapeutic_context_improved(user_id, prompt)
        
        if context:
            enhanced_system = f"""
{system}

THERAPEUTIC CONTEXT SYSTEM:
You have access to this user's therapeutic history and curated knowledge base. 
Provide continuity and reference previous conversations when appropriate.
When referencing knowledge base resources, mention the specific document name.

{context}
"""
        else:
            enhanced_system = system
        
        print(f"🔍 MODEL CALL: Final system prompt = {len(enhanced_system):,} characters")
        
        # Call appropriate model with improved functions
        if "gpt" in model.lower():
            if not openai_api_key:
                return "Error: OpenAI API key not configured"
            return call_openai_improved(enhanced_system, prompt)
        else:
            if not claude_api_key:
                return "Error: Claude API key not configured"
            return call_claude_improved(enhanced_system, prompt)
            
    except Exception as e:
        error_msg = f"Error calling model: {str(e)}"
        print(f"❌ MODEL CALL: {error_msg}")
        traceback.print_exc()
        return error_msg

# ================================
# ADMIN FUNCTIONS
# ================================

def is_admin_user():
    return session.get("is_admin", False)

# ================================
# DEBUG AND DIAGNOSTIC ROUTES
# ================================

@app.route("/debug-knowledge", methods=["POST"])
def debug_knowledge():
    """Debug endpoint to test knowledge base search - FOR ADMIN ONLY"""
    if not is_admin_user():
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        data = request.get_json()
        search_query = data.get("query", "")
        
        if not search_query:
            return jsonify({"error": "No search query provided"}), 400
        
        # Load knowledge base
        knowledge_base = load_knowledge_base()
        
        # Test the improved search
        context = build_therapeutic_context_improved("debug-user", search_query, limit_kb_docs=10)
        
        # Get document list
        all_docs = []
        for i, doc in enumerate(knowledge_base["documents"]):
            all_docs.append({
                "index": i,
                "filename": doc.get("filename", "Unknown"),
                "character_count": doc.get("character_count", 0),
                "authors": doc.get("extracted_authors", []),
                "content_preview": doc.get("content", "")[:300] + "..." if doc.get("content") else "NO CONTENT"
            })
        
        return jsonify({
            "search_query": search_query,
            "total_documents": len(knowledge_base["documents"]),
            "all_documents": all_docs,
            "generated_context_length": len(context),
            "context_preview": context[:1000] + "..." if len(context) > 1000 else context,
            "knowledge_base_stats": {
                "total_characters": knowledge_base.get("total_characters", 0),
                "total_authors": len(knowledge_base.get("authorized_authors", [])),
                "last_updated": knowledge_base.get("last_updated")
            }
        })
        
    except Exception as e:
        print(f"❌ Debug knowledge error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

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
                
                print(f"✅ New user registered and logged in: {username} ({email})")
                return jsonify({
                    "success": True, 
                    "message": "Account created successfully!",
                    "redirect": "/"
                })
                
        except Exception as e:
            print(f"❌ Registration error: {e}")
            traceback.print_exc()
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
            print(f"❌ Login error: {e}")
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
    
    # Legacy support for anonymous users OR create new anonymous user
    if 'user_id' in session:
        # Existing anonymous user - continue their session
        user = get_or_create_user()
        is_admin = is_admin_user()
        return render_template("index.html", is_admin=is_admin, user_id=user['id'])
    else:
        # New visitor - create anonymous user and go directly to chat
        user = get_or_create_user()
        is_admin = is_admin_user()
        return render_template("index.html", is_admin=is_admin, user_id=user['id'])

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

@app.route("/admin/batch-upload", methods=["GET"])
def admin_batch_upload():
    """Batch upload interface for multiple large PDFs"""
    if not session.get("is_admin"):
        return redirect("/admin")
    
    return render_template("batch_upload.html")

# ================================
# IMPROVED CHAT ROUTE
# ================================

@app.route("/chat", methods=["POST"])
def chat():
    """IMPROVED Chat endpoint with better context and debugging"""
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

        print(f"🔍 CHAT REQUEST: User={user_id}, Input='{user_input[:100]}...', Agent={agent}")

        # Save user message
        save_conversation_message(user_id, "user", user_input)

        # Build enhanced system prompt based on agent
        user_name = user.get('full_name') or user.get('username') or 'User'
        base_instruction = f"""You are a professional therapeutic AI supporting {user_name}. 

IMPORTANT: You have access to extensive curated therapeutic knowledge and user conversation history. 
When you reference information from the knowledge base, ALWAYS mention the specific document name.
When you find relevant information, say something like "According to [Document Name]..." or "As mentioned in [Book Title]...".

If asked about specific books or resources, check if they are available in your knowledge base and reference them specifically."""

        agent_prompts = {
            "case_assistant": f"{base_instruction} You assist with social work case analysis using evidence-based approaches.",
            "research_critic": f"{base_instruction} You critically evaluate research using evidence-based approaches and cite specific sources.",
            "therapy_planner": f"{base_instruction} You plan therapeutic interventions using proven methodologies from your knowledge base.",
            "therapist": f"{base_instruction} {config.get('claude_system_prompt', 'You provide therapeutic support using systemic approaches.')}"
        }

        system_prompt = agent_prompts.get(agent, agent_prompts["therapist"])
        model = config.get("claude_model", "claude-3-haiku-20240307")

        # Get AI response with improved context
        response_text = call_model_with_improved_context(model, system_prompt, user_input, user_id)
        
        # Save AI response
        save_conversation_message(user_id, "assistant", response_text, agent)

        print(f"✅ CHAT RESPONSE: Generated {len(response_text):,} chars for user {user_id}")

        return jsonify({
            "response": response_text, 
            "user_id": user_id,
            "username": user.get('username', ''),
            "context_included": True
        })
        
    except Exception as e:
        print(f"❌ CHAT ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ================================
# IMPROVED UPLOAD ROUTE
# ================================

@app.route("/upload", methods=["POST"])
def upload():
    """IMPROVED file upload endpoint with better processing and debugging"""
    print(f"🔍 UPLOAD START: Processing file upload")
    
    try:
        user = get_or_create_user()
        if not user:
            print("❌ UPLOAD: No user session")
            return jsonify({"error": "User session required"}), 401
        
        user_id = user['id']
        print(f"🔍 UPLOAD: User ID = {user_id}")
        
        if "pdf" not in request.files:
            print("❌ UPLOAD: No PDF in request.files")
            return jsonify({"message": "No file selected", "success": False})
        
        file = request.files["pdf"]
        
        if not file or not file.filename:
            print("❌ UPLOAD: No file or filename")
            return jsonify({"message": "No file selected", "success": False})
            
        if not file.filename.lower().endswith(".pdf"):
            print("❌ UPLOAD: Not a PDF file")
            return jsonify({"message": "Please select a PDF file", "success": False})
        
        upload_type = request.form.get("upload_type", "personal")
        
        # Check file size before processing
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"🔍 UPLOAD: File '{file.filename}' = {file_size_mb:.1f}MB, Type = {upload_type}")
        
        # Enforce file size limits
        if upload_type == "admin" and file_size_mb > 200:
            return jsonify({
                "message": f"Admin file too large ({file_size_mb:.1f}MB). Maximum: 200MB",
                "success": False
            })
        elif upload_type == "personal" and file_size_mb > 100:
            return jsonify({
                "message": f"Personal file too large ({file_size_mb:.1f}MB). Maximum: 100MB", 
                "success": False
            })
        
        if upload_type == "admin":
            # ADMIN UPLOAD
            if not is_admin_user():
                print("❌ UPLOAD: Not admin user")
                return jsonify({
                    "message": "Access denied. Admin privileges required.", 
                    "success": False
                }), 403
            
            print("🔍 UPLOAD: Processing admin upload...")
            file_path = os.path.join(UPLOADS_DIR, f"admin_{int(time.time())}_{file.filename}")
            
            # Ensure directory exists
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            
            # Save file
            file.save(file_path)
            print(f"✅ UPLOAD: Admin file saved to {file_path}")
            
            # Process document with improved method
            doc_info = add_document_to_knowledge_base(file_path, file.filename, is_core=True)
            
            if "error" in doc_info:
                print(f"❌ UPLOAD: Admin processing error - {doc_info['error']}")
                # Clean up uploaded file
                try:
                    os.remove(file_path)
                except:
                    pass
                return jsonify({"message": doc_info["error"], "success": False})
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
                print(f"🔍 UPLOAD: Cleaned up admin temp file")
            except Exception as e:
                print(f"⚠️ UPLOAD: Could not remove admin temp file: {e}")
            
            authors_text = f" | Authors: {', '.join(doc_info['extracted_authors'])}" if doc_info['extracted_authors'] else ""
            
            return jsonify({
                "message": f"✅ Added '{file.filename}' to global knowledge base ({doc_info['character_count']:,} characters){authors_text}", 
                "success": True,
                "type": "admin",
                "extracted_authors": doc_info['extracted_authors'],
                "character_count": doc_info['character_count']
            })
        
        else:
            # PERSONAL UPLOAD
            print("🔍 UPLOAD: Processing personal upload...")
            file_path = os.path.join(USER_UPLOADS_DIR, f"user_{user_id}_{int(time.time())}_{file.filename}")
            
            # Ensure directory exists
            os.makedirs(USER_UPLOADS_DIR, exist_ok=True)
            
            # Save file
            file.save(file_path)
            print(f"✅ UPLOAD: Personal file saved to {file_path}")
            
            doc_info = add_personal_document_improved(file_path, file.filename, user_id)
            
            if "error" in doc_info:
                print(f"❌ UPLOAD: Personal processing error - {doc_info['error']}")
                # Clean up uploaded file
                try:
                    os.remove(file_path)
                except:
                    pass
                return jsonify({"message": doc_info["error"], "success": False})
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
                print(f"🔍 UPLOAD: Cleaned up personal temp file")
            except Exception as e:
                print(f"⚠️ UPLOAD: Could not remove personal temp file: {e}")
            
            return jsonify({
                "message": f"✅ Added '{file.filename}' to your persistent documents ({doc_info['character_count']:,} characters)", 
                "success": True,
                "type": "personal",
                "character_count": doc_info['character_count']
            })
        
    except Exception as e:
        print(f"❌ UPLOAD EXCEPTION: {str(e)}")
        traceback.print_exc()
        gc.collect()  # Force memory cleanup on error
        return jsonify({
            "error": f"Upload failed: {str(e)}", 
            "success": False
        }), 500

# ================================
# REMAINING UTILITY ROUTES
# ================================

@app.route("/clear", methods=["POST"])
def clear():
    """Clear chat history"""
    try:
        user = get_or_create_user()
        if not user:
            return jsonify({"error": "User session required"}), 401
        
        user_id = user['id']
        clear_user_conversation(user_id)
        print(f"✅ CLEAR: Cleared conversation history for user {user_id}")
        return jsonify({"message": "Chat history cleared (documents and progress retained)"})
    except Exception as e:
        print(f"❌ CLEAR ERROR: {e}")
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
        print(f"✅ CLEAR DOCS: Cleared {cleared_count} personal documents for user {user_id}")
        return jsonify({"message": f"Cleared {cleared_count} personal documents"})
    except Exception as e:
        print(f"❌ CLEAR DOCS ERROR: {e}")
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
        print(f"❌ CONVERSATION HISTORY ERROR: {e}")
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
        print(f"❌ PERSONAL DOCS ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/knowledge-base", methods=["GET"])
def get_knowledge_base():
    """Get knowledge base status with improved stats"""
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
        print(f"❌ KNOWLEDGE BASE ERROR: {e}")
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
        print(f"❌ USER STATS ERROR: {e}")
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
        print(f"❌ LOG ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    """System health check with improved diagnostics"""
    try:
        # Test API connections
        openai_works = False
        claude_works = False
        
        if openai_api_key:
            try:
                test_response = call_openai_improved("You are a test.", "Hello")
                openai_works = not test_response.startswith("Error")
            except:
                pass
        
        if claude_api_key:
            try:
                test_response = call_claude_improved("You are a test.", "Hello")
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
            "version": "improved_v2.0",
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
                "pdf_extraction": "pdfplumber_improved",
                "controlled_knowledge": True,
                "large_pdf_support": True,
                "improved_search": True,
                "batch_upload": True,
                "memory_safety": True,
                "debug_tools": True
            },
            "knowledge_base": {
                "total_documents": len(knowledge_base["documents"]),
                "total_characters": knowledge_base.get("total_characters", 0),
                "size_mb": round(knowledge_base.get("total_characters", 0) / 1024 / 1024, 2),
                "last_updated": knowledge_base.get("last_updated"),
                "authorized_authors": len(knowledge_base.get("authorized_authors", [])),
                "files": [
                    {
                        "filename": doc.get("filename", "Unknown"),
                        "characters": doc.get("character_count", 0),
                        "authors": len(doc.get("extracted_authors", []))
                    }
                    for doc in knowledge_base["documents"][:5]  # Show first 5
                ]
            },
            "memory_info": {
                "database_file": DATABASE_FILE,
                "pdf_max_size_admin_mb": 200,
                "pdf_max_size_personal_mb": 100,
                "memory_limit_mb": 2048,
                "tier": "render_standard",
                "context_processing": "improved_search_algorithm"
            }
        })
    except Exception as e:
        print(f"❌ HEALTH CHECK ERROR: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500

# ================================
# BATCH UPLOAD SYSTEM
# ================================

@app.route("/admin/batch-process", methods=["POST"])
def admin_batch_process():
    """Process batch upload via API - IMPROVED VERSION"""
    if not is_admin_user():
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        # Get uploaded files
        files = request.files.getlist('pdfs')
        if not files:
            return jsonify({"error": "No files uploaded"}), 400
        
        print(f"🔍 BATCH UPLOAD: Processing {len(files)} files")
        results = []
        
        for i, file in enumerate(files):
            if file.filename and file.filename.lower().endswith('.pdf'):
                try:
                    print(f"🔍 BATCH UPLOAD: Processing file {i+1}/{len(files)}: {file.filename}")
                    
                    # Save file temporarily
                    temp_path = os.path.join(UPLOADS_DIR, f"batch_{int(time.time())}_{i}_{file.filename}")
                    file.save(temp_path)
                    
                    file_size_mb = os.path.getsize(temp_path) / 1024 / 1024
                    
                    # Process with improved method
                    doc_info = add_document_to_knowledge_base(temp_path, file.filename, is_core=True)
                    
                    if not doc_info.get('error'):
                        results.append({
                            'filename': file.filename,
                            'status': 'success',
                            'size_mb': round(file_size_mb, 2),
                            'character_count': doc_info.get('character_count', 0),
                            'authors': doc_info.get('extracted_authors', [])
                        })
                        print(f"✅ BATCH UPLOAD: Successfully processed {file.filename}")
                    else:
                        results.append({
                            'filename': file.filename,
                            'status': 'failed',
                            'error': doc_info.get('error', 'Processing failed')
                        })
                        print(f"❌ BATCH UPLOAD: Failed to process {file.filename}: {doc_info.get('error')}")
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                        
                except Exception as e:
                    results.append({
                        'filename': file.filename,
                        'status': 'failed',
                        'error': str(e)
                    })
                    print(f"❌ BATCH UPLOAD: Exception processing {file.filename}: {e}")
        
        # Calculate summary stats
        successful = len([r for r in results if r['status'] == 'success'])
        failed = len([r for r in results if r['status'] == 'failed'])
        
        print(f"✅ BATCH UPLOAD: Completed - {successful} successful, {failed} failed")
        
        return jsonify({
            "success": True,
            "message": f"Batch upload completed: {successful} successful, {failed} failed",
            "results": results,
            "summary": {
                "total": len(results),
                "successful": successful,
                "failed": failed
            }
        })
        
    except Exception as e:
        print(f"❌ BATCH UPLOAD ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ================================
# ADMIN DEBUG TOOLS
# ================================

@app.route("/admin/debug-system", methods=["GET"])
def debug_system():
    """Debug system status - ADMIN ONLY"""
    if not is_admin_user():
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        # Check file system
        directories = {}
        for dir_name in [CORE_MEMORY_DIR, UPLOADS_DIR, USER_UPLOADS_DIR]:
            directories[dir_name] = {
                "exists": os.path.exists(dir_name),
                "writable": os.access(dir_name, os.W_OK) if os.path.exists(dir_name) else False,
                "contents": len(os.listdir(dir_name)) if os.path.exists(dir_name) else 0
            }
        
        # Check knowledge base file
        kb_status = {
            "file_exists": os.path.exists(KNOWLEDGE_BASE_FILE),
            "file_size": os.path.getsize(KNOWLEDGE_BASE_FILE) if os.path.exists(KNOWLEDGE_BASE_FILE) else 0,
            "readable": os.access(KNOWLEDGE_BASE_FILE, os.R_OK) if os.path.exists(KNOWLEDGE_BASE_FILE) else False
        }
        
        # Load and analyze knowledge base
        knowledge_base = load_knowledge_base()
        kb_analysis = {
            "total_docs": len(knowledge_base["documents"]),
            "total_chars": knowledge_base.get("total_characters", 0),
            "authors": len(knowledge_base.get("authorized_authors", [])),
            "last_updated": knowledge_base.get("last_updated")
        }
        
        # Database stats
        with get_db_connection() as conn:
            db_stats = {
                "users": conn.execute('SELECT COUNT(*) as c FROM users').fetchone()['c'],
                "conversations": conn.execute('SELECT COUNT(*) as c FROM conversations').fetchone()['c'],
                "user_docs": conn.execute('SELECT COUNT(*) as c FROM user_documents WHERE is_active = 1').fetchone()['c'],
                "kb_docs": conn.execute('SELECT COUNT(*) as c FROM knowledge_base_docs').fetchone()['c']
            }
        
        return jsonify({
            "system_status": "operational",
            "timestamp": datetime.now().isoformat(),
            "directories": directories,
            "knowledge_base_file": kb_status,
            "knowledge_base_analysis": kb_analysis,
            "database_stats": db_stats,
            "memory_usage": {
                "python_process": "unknown",  # Could add psutil here if needed
                "gc_collections": gc.get_count()
            }
        })
        
    except Exception as e:
        print(f"❌ DEBUG SYSTEM ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ================================
# INITIALIZATION SYSTEM
# ================================

def initialize_improved_system():
    """Initialize system with all improvements"""
    print("🚀 Initializing IMPROVED Therapeutic AI System...")
    
    # Set memory limits
    limit_memory()
    
    # Initialize database
    init_database()
    
    # Load and verify knowledge base
    knowledge_base = load_knowledge_base()
    print(f"📚 Knowledge base status: {len(knowledge_base['documents'])} documents, {knowledge_base.get('total_characters', 0):,} characters")
    
    # Show available documents for debugging
    if knowledge_base["documents"]:
        print("📚 Available documents:")
        for i, doc in enumerate(knowledge_base["documents"][:5]):  # Show first 5
            print(f"  {i+1}. '{doc.get('filename', 'Unknown')}' ({doc.get('character_count', 0):,} chars)")
        if len(knowledge_base["documents"]) > 5:
            print(f"  ... and {len(knowledge_base['documents']) - 5} more documents")
    
    # Check database health
    try:
        with get_db_connection() as conn:
            total_users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
            authenticated_users = conn.execute('SELECT COUNT(*) as count FROM users WHERE email IS NOT NULL').fetchone()['count']
            total_conversations = conn.execute('SELECT COUNT(*) as count FROM conversations').fetchone()['count']
            print(f"👥 Database health: {total_users} users ({authenticated_users} registered), {total_conversations} conversations")
    except Exception as e:
        print(f"⚠️ Database check failed: {e}")
    
    # Test API connections
    api_status = []
    if claude_api_key:
        api_status.append("Claude API configured")
    if openai_api_key:
        api_status.append("OpenAI API configured")
    
    if api_status:
        print(f"🔑 APIs: {', '.join(api_status)}")
    else:
        print("⚠️ No API keys configured")
    
    # Force initial garbage collection
    gc.collect()
    
    print("✅ IMPROVED Therapeutic AI System initialized!")
    print("🎯 New features active:")
    print("   - Improved PDF extraction (up to 2000 pages)")
    print("   - Advanced search algorithm with title matching")
    print("   - Enhanced context building (up to 1MB per document)")
    print("   - Better author extraction")
    print("   - Debug logging throughout")
    print("   - Improved error handling and memory management")
    print("   - Enhanced API calls with longer timeouts")
    print("   - Better user authentication and profile management")
    print("   - Admin debug tools")
    print()
    print("🔍 Debug features:")
    print("   - Search debug logs in console")
    print("   - Context building logs")
    print("   - Upload processing logs")
    print("   - API call logging")
    print("   - Admin debug endpoints")

# ================================
# APPLICATION STARTUP
# ================================

if __name__ == "__main__":
    initialize_improved_system()
    app.run(host="0.0.0.0", port=5000, debug=False)
else:
    initialize_improved_system()
    application = app
