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
import tiktoken
from datetime import datetime, timedelta
from flask import Flask, request, render_template, jsonify, redirect, session
from dotenv import load_dotenv
import pdfplumber
import gc
from contextlib import contextmanager
from functools import wraps
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import traceback

# New imports for embeddings and vector store
import openai
import chromadb
from chromadb.config import Settings
import numpy as np

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
admin_password = os.getenv("ADMIN_PASSWORD")

if not openai_api_key:
    print("WARNING: OPENAI_API_KEY not found in environment variables")
if not claude_api_key:
    print("WARNING: CLAUDE_API_KEY not found in environment variables")
if not admin_password:
    print("WARNING: ADMIN_PASSWORD not found in environment variables")

# OpenAI client initialized per-request in get_openai_embedding function

# File paths
DATABASE_FILE = "therapeutic_ai.db"
KNOWLEDGE_BASE_FILE = "core_memory/knowledge_base.json"
CORE_MEMORY_DIR = "core_memory"
UPLOADS_DIR = "uploads"
USER_UPLOADS_DIR = "user_uploads"
CHROMA_PERSIST_DIR = "chroma_db"

# Ensure directories exist
for directory in [CORE_MEMORY_DIR, UPLOADS_DIR, USER_UPLOADS_DIR, "logs", CHROMA_PERSIST_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"✅ Created/verified directory: {directory}")

# ================================
# ENHANCED EMBEDDINGS & VECTOR STORE SYSTEM
# ================================

# Initialize ChromaDB
def init_chroma():
    """Initialize ChromaDB with persistent storage"""
    try:
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        print("✅ ChromaDB initialized with persistent storage")
        return client
    except Exception as e:
        print(f"❌ ChromaDB initialization failed: {e}")
        return None

# Global ChromaDB client
chroma_client = init_chroma()

# Token counting for budget management
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimation (4 chars per token)
        return len(text) // 4

def get_openai_embedding(text: str, model: str = "text-embedding-3-small") -> Optional[List[float]]:
    """Get embedding from OpenAI"""
    try:
        if not openai_api_key:
            print("❌ OpenAI API key not available for embeddings")
            return None

        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)

        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding

    except Exception as e:
        print(f"❌ OpenAI embedding error: {e}")
        return None

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[Dict]:
    """Split text into overlapping chunks with metadata"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "text": chunk_text,
            "chunk_index": len(chunks),
            "start_word": i,
            "end_word": i + len(chunk_words),
            "word_count": len(chunk_words),
            "char_count": len(chunk_text)
        })
        
        # Stop if this chunk contains the last words
        if i + chunk_size >= len(words):
            break
    
    return chunks

def get_or_create_collection(collection_name: str) -> Optional[Any]:
    """Get or create a ChromaDB collection"""
    try:
        if not chroma_client:
            return None
            
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": f"Therapeutic AI {collection_name} collection"}
        )
        return collection
    except Exception as e:
        print(f"❌ Error with collection {collection_name}: {e}")
        return None

def add_document_to_vector_store(
    doc_id: str, 
    filename: str, 
    content: str, 
    collection_name: str = "admin_kb",
    user_id: Optional[str] = None
) -> bool:
    """Add document chunks to vector store with embeddings"""
    try:
        collection = get_or_create_collection(collection_name)
        if not collection:
            return False
            
        print(f"🔍 VECTOR STORE: Adding {filename} to {collection_name}")
        
        # Create chunks
        chunks = chunk_text(content)
        print(f"📄 Created {len(chunks)} chunks from {filename}")
        
        # Generate embeddings and add to collection
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for chunk in chunks:
            chunk_id = f"{doc_id}_chunk_{chunk['chunk_index']}"
            
            # Get embedding
            embedding = get_openai_embedding(chunk["text"])
            if not embedding:
                print(f"⚠️ Failed to get embedding for chunk {chunk['chunk_index']}")
                continue
                
            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(chunk["text"])
            
            metadata = {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": chunk["chunk_index"],
                "word_count": chunk["word_count"],
                "char_count": chunk["char_count"],
                "collection_type": collection_name
            }
            
            if user_id:
                metadata["user_id"] = user_id
                
            metadatas.append(metadata)
        
        if ids:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            print(f"✅ Added {len(ids)} chunks to {collection_name}")
            return True
        else:
            print(f"❌ No chunks could be embedded for {filename}")
            return False
            
    except Exception as e:
        print(f"❌ Vector store error for {filename}: {e}")
        traceback.print_exc()
        return False

def query_vector_store(
    query: str, 
    collection_name: str = "admin_kb",
    user_id: Optional[str] = None,
    n_results: int = 8
) -> List[Dict]:
    """Query vector store for relevant chunks"""
    try:
        collection = get_or_create_collection(collection_name)
        if not collection:
            return []
            
        # Get query embedding
        query_embedding = get_openai_embedding(query)
        if not query_embedding:
            return []
            
        # Build where clause for user filtering
        where_clause = None
        if user_id and collection_name == "user_docs":
            where_clause = {"user_id": user_id}
            
        # Query collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results and results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i] if results['distances'] else 1.0
                similarity = 1 - distance  # Convert distance to similarity
                
                formatted_results.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i],
                    "similarity": similarity,
                    "distance": distance
                })
        
        return formatted_results
        
    except Exception as e:
        print(f"❌ Vector store query error: {e}")
        return []

def get_document_summary(doc_content: str, max_length: int = 200) -> str:
    """Create a brief summary of document content"""
    try:
        # Take first few sentences up to max_length
        sentences = doc_content.split('. ')
        summary = ""
        for sentence in sentences:
            if len(summary + sentence) > max_length:
                break
            summary += sentence + ". "
        
        return summary.strip() or doc_content[:max_length] + "..."
    except:
        return doc_content[:max_length] + "..."

# ================================
# ENHANCED RETRIEVAL SYSTEM
# ================================

def retrieve_relevant_context(
    user_id: str, 
    query: str, 
    max_tokens: int = 20000
) -> Dict[str, Any]:
    """
    Enhanced retrieval system with token budget management
    Returns context with debug information
    """
    debug_info = {
        "query": query,
        "retrieval_timestamp": datetime.now().isoformat(),
        "admin_chunks_found": 0,
        "user_chunks_found": 0,
        "total_context_tokens": 0,
        "chunks_used": [],
        "token_breakdown": {}
    }
    
    context_parts = []
    token_count = 0
    
    print(f"🔍 RETRIEVAL: Starting enhanced retrieval for query: '{query[:100]}...'")
    
    # 1. Get conversation history summary
    conversation_history = get_user_conversation_history(user_id, limit=10)
    history_summary = ""
    if conversation_history:
        # Summarize recent conversation if too long
        recent_messages = conversation_history[-5:]  # Last 5 messages
        history_text = "\n".join([
            f"{'User' if msg['message_type'] == 'user' else 'AI'}: {msg['content'][:200]}..."
            for msg in recent_messages
        ])
        
        history_tokens = count_tokens(history_text)
        if history_tokens < 1000:  # If short enough, include full history
            history_summary = f"=== RECENT CONVERSATION CONTEXT ===\n{history_text}\n\n"
            token_count += history_tokens
            debug_info["token_breakdown"]["conversation_history"] = history_tokens
    
    # 2. Query admin knowledge base
    admin_results = query_vector_store(query, "admin_kb", n_results=8)
    debug_info["admin_chunks_found"] = len(admin_results)
    
    print(f"📚 Found {len(admin_results)} admin chunks")
    
    # 3. Query user documents
    user_results = query_vector_store(query, "user_docs", user_id, n_results=6)
    debug_info["user_chunks_found"] = len(user_results)
    
    print(f"📁 Found {len(user_results)} user chunks")
    
    # 4. Combine and sort by relevance
    all_results = []
    
    # Add admin results with source marking
    for result in admin_results:
        result["source_type"] = "admin_kb"
        all_results.append(result)
    
    # Add user results with source marking  
    for result in user_results:
        result["source_type"] = "user_docs"
        all_results.append(result)
    
    # Sort by similarity score
    all_results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # 5. Build context within token budget
    admin_context = ""
    user_context = ""
    chunks_used = 0
    
    # Reserve tokens for system prompt and response
    available_tokens = max_tokens - token_count - 2000  # Reserve 2k for system prompt
    
    for result in all_results:
        chunk_tokens = count_tokens(result["text"])
        
        # Check if we have room for this chunk
        if token_count + chunk_tokens > available_tokens:
            print(f"⚠️ Token budget exhausted. Used {chunks_used} chunks, {token_count} tokens")
            break
        
        # Add chunk to appropriate section
        if result["source_type"] == "admin_kb":
            if not admin_context:
                admin_context += "=== THERAPEUTIC KNOWLEDGE BASE ===\n"
            
            filename = result["metadata"].get("filename", "Unknown")
            chunk_idx = result["metadata"].get("chunk_index", 0)
            admin_context += f"\n--- From: {filename} (Chunk {chunk_idx}, Relevance: {result['similarity']:.3f}) ---\n"
            admin_context += result["text"] + "\n"
            
        else:  # user_docs
            if not user_context:
                user_context += "\n=== YOUR PERSONAL DOCUMENTS ===\n"
            
            filename = result["metadata"].get("filename", "Unknown") 
            chunk_idx = result["metadata"].get("chunk_index", 0)
            user_context += f"\n--- From your document: {filename} (Chunk {chunk_idx}, Relevance: {result['similarity']:.3f}) ---\n"
            user_context += result["text"] + "\n"
        
        token_count += chunk_tokens
        chunks_used += 1
        
        debug_info["chunks_used"].append({
            "filename": result["metadata"].get("filename", "Unknown"),
            "chunk_index": result["metadata"].get("chunk_index", 0),
            "similarity": result["similarity"],
            "source_type": result["source_type"],
            "tokens": chunk_tokens
        })
    
    # 6. Assemble final context
    final_context = history_summary + admin_context + user_context
    
    # Add metadata
    if admin_context or user_context:
        final_context += f"\n=== CONTEXT METADATA ===\n"
        final_context += f"Retrieved {chunks_used} most relevant chunks\n"
        final_context += f"Admin KB chunks: {debug_info['admin_chunks_found']}\n"
        final_context += f"User doc chunks: {debug_info['user_chunks_found']}\n"
        final_context += f"Total context tokens: {token_count}\n"
        final_context += f"Generated at: {datetime.now().isoformat()}\n"
    
    debug_info["total_context_tokens"] = token_count
    debug_info["token_breakdown"]["admin_context"] = count_tokens(admin_context)
    debug_info["token_breakdown"]["user_context"] = count_tokens(user_context)
    debug_info["chunks_included"] = chunks_used
    
    print(f"✅ RETRIEVAL COMPLETE: {chunks_used} chunks, {token_count} tokens")
    
    return {
        "context": final_context,
        "debug_info": debug_info,
        "token_count": token_count,
        "chunks_used": chunks_used
    }

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
    conn = sqlite3.connect(DATABASE_FILE, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
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
        columns_to_add = [
            'email TEXT', 'username TEXT', 'password_hash TEXT', 'full_name TEXT',
            'is_verified BOOLEAN DEFAULT 1', 'subscription_type TEXT DEFAULT "free"',
            'total_sessions INTEGER DEFAULT 0'
        ]
        
        for column in columns_to_add:
            try:
                conn.execute(f'ALTER TABLE users ADD COLUMN {column}')
            except sqlite3.OperationalError:
                pass  # Column already exists
        
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
                vector_indexed BOOLEAN DEFAULT 0,
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
                access_count INTEGER DEFAULT 0,
                vector_indexed BOOLEAN DEFAULT 0
            )
        ''')
        
        # Add vector_indexed columns if missing
        try:
            conn.execute('ALTER TABLE user_documents ADD COLUMN vector_indexed BOOLEAN DEFAULT 0')
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute('ALTER TABLE knowledge_base_docs ADD COLUMN vector_indexed BOOLEAN DEFAULT 0')
        except sqlite3.OperationalError:
            pass
        
        conn.commit()
        print("✅ Enhanced database with vector indexing support initialized")

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
# IMPROVED KNOWLEDGE BASE SYSTEM WITH VECTOR INDEXING
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
    """Add document to admin knowledge base with vector indexing"""
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
        
        # Add to vector store
        vector_success = add_document_to_vector_store(
            doc_id, filename, text_content, "admin_kb"
        )
        
        # Track in database
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO knowledge_base_docs 
                (id, filename, file_hash, character_count, extracted_authors, vector_indexed)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (doc_id, filename, file_hash, len(text_content), 
                  json.dumps(extracted_authors), vector_success))
            conn.commit()
        
        print(f"✅ Added '{filename}' to knowledge base ({len(text_content):,} characters)")
        print(f"🔍 Vector indexing: {'Success' if vector_success else 'Failed'}")
        
        return doc_info
        
    except Exception as e:
        error_msg = f"Error adding document: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return {"error": error_msg}
    finally:
        gc.collect()

# ================================
# USER DOCUMENT PERSISTENCE WITH VECTOR INDEXING
# ================================

def add_personal_document_improved(file_path, filename, user_id):
    """Add document to user's persistent personal collection with vector indexing"""
    try:
        print(f"🔍 PERSONAL UPLOAD: Processing {filename} for user {user_id}")
        
        text_content = extract_text_from_pdf_improved(file_path, max_size_mb=100)
        
        if text_content.startswith("Error") or "too large" in text_content:
            return {"error": text_content}
        
        file_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        doc_id = str(uuid.uuid4())
        
        # Add to vector store
        vector_success = add_document_to_vector_store(
            doc_id, filename, text_content, "user_docs", user_id
        )
        
        # Save to database
        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO user_documents 
                (id, user_id, filename, content, file_hash, character_count, vector_indexed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (doc_id, user_id, filename, text_content, file_hash, 
                  len(text_content), vector_success))
            conn.commit()
        
        doc_info = {
            "id": doc_id,
            "filename": filename,
            "content": text_content,
            "file_hash": file_hash,
            "character_count": len(text_content),
            "upload_date": datetime.now().isoformat(),
            "type": "user_personal_document",
            "vector_indexed": vector_success
        }
        
        print(f"✅ Added '{filename}' to user {user_id} ({len(text_content):,} characters)")
        print(f"🔍 Vector indexing: {'Success' if vector_success else 'Failed'}")
        
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
    """Clear user's personal documents and vector store entries"""
    try:
        # Get user documents before clearing
        docs = get_user_documents(user_id)
        
        # Remove from vector store
        user_collection = get_or_create_collection("user_docs")
        if user_collection and docs:
            doc_ids = [doc['id'] for doc in docs]
            # Remove chunks for these documents
            for doc_id in doc_ids:
                try:
                    # Get all chunk IDs for this document
                    results = user_collection.get(
                        where={"doc_id": doc_id, "user_id": user_id}
                    )
                    if results and results['ids']:
                        user_collection.delete(ids=results['ids'])
                        print(f"🗑️ Removed vector chunks for {doc_id}")
                except Exception as e:
                    print(f"⚠️ Error removing vector chunks: {e}")
        
        # Clear from database
        with get_db_connection() as conn:
            result = conn.execute(
                'UPDATE user_documents SET is_active = 0 WHERE user_id = ?', (user_id,)
            )
            conn.commit()
            return result.rowcount
            
    except Exception as e:
        print(f"❌ Error clearing user documents: {e}")
        return 0

# ================================
# ENHANCED API FUNCTIONS WITH CLAUDE PRIMARY
# ================================

def call_claude_improved(system_prompt, user_message):
    """Improved Claude API call with better error handling - PRIMARY MODEL"""
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": claude_api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Use larger context window for Claude
        data = {
            "model": config.get("claude_model", "claude-3-haiku-20240307"),
            "max_tokens": config.get("max_tokens", 2048),
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}]
        }
        
        print(f"🔍 CLAUDE API (PRIMARY): Sending {len(system_prompt):,} system chars + {len(user_message):,} user chars")
        
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
            return None  # Return None to trigger fallback
            
    except Exception as e:
        error_msg = f"Error in Claude call: {str(e)}"
        print(f"❌ CLAUDE API: {error_msg}")
        return None  # Return None to trigger fallback

def call_openai_improved(system_prompt, user_message):
    """Improved OpenAI API call with better error handling - FALLBACK MODEL"""
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
        
        print(f"🔍 OPENAI API (FALLBACK): Sending {len(system_prompt):,} system chars + {len(user_message):,} user chars")
        
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

def call_model_with_enhanced_retrieval(model, system, prompt, user_id):
    """Enhanced model calling with retrieval system and Claude primary"""
    try:
        print(f"🔍 MODEL CALL: Model = {model}, User = {user_id}")
        
        # Use enhanced retrieval system
        retrieval_result = retrieve_relevant_context(user_id, prompt)
        context = retrieval_result["context"]
        debug_info = retrieval_result["debug_info"]
        
        if context:
            enhanced_system = f"""
{system}

THERAPEUTIC CONTEXT WITH VECTOR RETRIEVAL:
You have access to this user's therapeutic history and curated knowledge base through advanced semantic search.
When referencing knowledge base resources, mention the specific document name and relevance score.
Provide continuity based on conversation history when appropriate.

{context}
"""
        else:
            enhanced_system = system
        
        print(f"🔍 MODEL CALL: Final system prompt = {len(enhanced_system):,} characters")
        print(f"🔍 RETRIEVAL DEBUG: {debug_info['chunks_included']} chunks, {debug_info['total_context_tokens']} tokens")
        
        # Try Claude first (PRIMARY), fallback to OpenAI
        if claude_api_key:
            claude_response = call_claude_improved(enhanced_system, prompt)
            if claude_response:  # Claude succeeded
                return claude_response
            else:
                print("⚠️ Claude failed, falling back to OpenAI...")
        
        # Fallback to OpenAI
        if openai_api_key:
            return call_openai_improved(enhanced_system, prompt)
        else:
            return "Error: No working API keys available"
            
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
# NEW VECTOR STORE ADMIN ROUTES
# ================================

@app.route("/rebuild-index", methods=["POST"])
def rebuild_vector_index():
    """ADMIN ONLY: Rebuild vector index for all documents"""
    if not is_admin_user():
        return jsonify({"error": "Admin access required"}), 403

    try:
        print("🔧 REBUILD INDEX: Starting complete vector index rebuild...")

        # Get all documents from database
        with get_db_connection() as conn:
            admin_docs = conn.execute('SELECT * FROM knowledge_base_docs').fetchall()
            user_docs = conn.execute('SELECT * FROM user_documents WHERE is_active = 1').fetchall()

        # Load knowledge base for admin documents content
        knowledge_base = load_knowledge_base()
        admin_content_map = {doc['id']: doc for doc in knowledge_base['documents']}

        rebuild_stats = {
            "admin_docs_processed": 0,
            "user_docs_processed": 0,
            "admin_docs_failed": 0,
            "user_docs_failed": 0,
            "total_chunks_created": 0,
            "errors": []
        }

        # Clear existing collections
        try:
            if chroma_client:
                chroma_client.delete_collection("admin_kb")
                chroma_client.delete_collection("user_docs")
                print("🗑️ Cleared existing vector collections")
        except Exception as e:
            print(f"⚠️ Error clearing collections: {e}")

        # Rebuild admin knowledge base
        print(f"📚 Rebuilding admin KB: {len(admin_docs)} documents")
        for doc_row in admin_docs:
            doc_id = doc_row['id']
            filename = doc_row['filename']

            if doc_id in admin_content_map:
                content = admin_content_map[doc_id]['content']
                success = add_document_to_vector_store(doc_id, filename, content, "admin_kb")

                if success:
                    rebuild_stats["admin_docs_processed"] += 1
                    with get_db_connection() as conn:
                        conn.execute(
                            'UPDATE knowledge_base_docs SET vector_indexed = 1 WHERE id = ?',
                            (doc_id,)
                        )
                        conn.commit()
                else:
                    rebuild_stats["admin_docs_failed"] += 1
                    rebuild_stats["errors"].append(f"Failed to index admin doc: {filename}")

        # Rebuild user documents
        print(f"👤 Rebuilding user docs: {len(user_docs)} documents")
        for doc_row in user_docs:
            doc_id = doc_row['id']
            filename = doc_row['filename']
            content = doc_row['content']
            user_id = doc_row['user_id']

            success = add_document_to_vector_store(doc_id, filename, content, "user_docs", user_id)

            if success:
                rebuild_stats["user_docs_processed"] += 1
                with get_db_connection() as conn:
                    conn.execute(
                        'UPDATE user_documents SET vector_indexed = 1 WHERE id = ?',
                        (doc_id,)
                    )
                    conn.commit()
            else:
                rebuild_stats["user_docs_failed"] += 1
                rebuild_stats["errors"].append(f"Failed to index user doc: {filename}")

        print(f"✅ REBUILD COMPLETE: {rebuild_stats}")

        return jsonify({
            "success": True,
            "message": "Vector index rebuild completed",
            "stats": rebuild_stats
        })

    except Exception as e:
        error_msg = f"Rebuild index error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

    @app.route("/admin/debug-system", methods=["GET"])
def debug_system_status():
    """ADMIN ONLY: Complete system diagnostic"""
    if not is_admin_user():
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        import psutil
        import gc
        
        debug_info = {
            "system_status": "operational",
            "timestamp": datetime.now().isoformat(),
            "version": "enhanced_vector_v3.0"
        }
        
        # Check directories
        directories = {}
        for directory in [CORE_MEMORY_DIR, UPLOADS_DIR, USER_UPLOADS_DIR, "logs", CHROMA_PERSIST_DIR]:
            try:
                directories[directory] = {
                    "exists": os.path.exists(directory),
                    "writable": os.access(directory, os.W_OK) if os.path.exists(directory) else False,
                    "contents": len(os.listdir(directory)) if os.path.exists(directory) else 0
                }
            except Exception as e:
                directories[directory] = {"exists": False, "error": str(e)}
        
        debug_info["directories"] = directories
        
        # Check knowledge base file
        try:
            kb_file_exists = os.path.exists(KNOWLEDGE_BASE_FILE)
            kb_file_size = os.path.getsize(KNOWLEDGE_BASE_FILE) if kb_file_exists else 0
            debug_info["knowledge_base_file"] = {
                "file_exists": kb_file_exists,
                "file_size": kb_file_size,
                "readable": os.access(KNOWLEDGE_BASE_FILE, os.R_OK) if kb_file_exists else False
            }
        except Exception as e:
            debug_info["knowledge_base_file"] = {"error": str(e)}
        
        # Knowledge base analysis
        try:
            knowledge_base = load_knowledge_base()
            debug_info["knowledge_base_analysis"] = {
                "total_docs": len(knowledge_base.get("documents", [])),
                "total_chars": knowledge_base.get("total_characters", 0),
                "authors": len(knowledge_base.get("authorized_authors", [])),
                "last_updated": knowledge_base.get("last_updated")
            }
        except Exception as e:
            debug_info["knowledge_base_analysis"] = {"error": str(e)}
        
        # Database stats
        try:
            with get_db_connection() as conn:
                users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
                conversations = conn.execute('SELECT COUNT(*) as count FROM conversations').fetchone()['count']
                user_docs = conn.execute('SELECT COUNT(*) as count FROM user_documents WHERE is_active = 1').fetchone()['count']
                kb_docs = conn.execute('SELECT COUNT(*) as count FROM knowledge_base_docs').fetchone()['count']
                
                debug_info["database_stats"] = {
                    "users": users,
                    "conversations": conversations,
                    "user_docs": user_docs,
                    "kb_docs": kb_docs
                }
        except Exception as e:
            debug_info["database_stats"] = {"error": str(e)}
        
        # Memory usage
        try:
            process = psutil.Process()
            debug_info["memory_usage"] = {
                "python_process": f"{process.memory_info().rss / 1024 / 1024:.1f} MB",
                "gc_collections": {
                    "gen0": gc.get_count()[0],
                    "gen1": gc.get_count()[1], 
                    "gen2": gc.get_count()[2]
                }
            }
        except Exception as e:
            debug_info["memory_usage"] = {"error": str(e)}
        
        return jsonify(debug_info)
        
    except Exception as e:
        print(f"❌ Debug system error: {e}")
        traceback.print_exc()
        return jsonify({
            "system_status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 500

@app.route("/debug-knowledge", methods=["POST"])
def debug_knowledge_search():
    """ADMIN ONLY: Debug knowledge base search functionality"""
    if not is_admin_user():
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        data = request.get_json()
        query = data.get("query", "")
        
        if not query:
            return jsonify({"error": "Query parameter required"}), 400
        
        print(f"🔍 DEBUG KNOWLEDGE: Testing search for '{query}'")
        
        # Load knowledge base
        knowledge_base = load_knowledge_base()
        
        # Test vector search
        admin_results = query_vector_store(query, "admin_kb", n_results=5)
        
        # Test retrieval system
        test_user_id = "debug-test-user"
        retrieval_result = retrieve_relevant_context(test_user_id, query, max_tokens=5000)
        
        debug_response = {
            "search_query": query,
            "timestamp": datetime.now().isoformat(),
            "total_documents": len(knowledge_base.get("documents", [])),
            "vector_results_count": len(admin_results),
            "generated_context_length": len(retrieval_result.get("context", "")),
            "all_documents": [],
            "vector_results": [],
            "context_preview": retrieval_result.get("context", "")[:1000] + "..." if len(retrieval_result.get("context", "")) > 1000 else retrieval_result.get("context", ""),
            "knowledge_base_stats": {
                "total_characters": knowledge_base.get("total_characters", 0),
                "total_authors": len(knowledge_base.get("authorized_authors", [])),
                "last_updated": knowledge_base.get("last_updated")
            }
        }
        
        # Add document summaries
        for doc in knowledge_base.get("documents", [])[:10]:  # First 10 docs
            doc_summary = {
                "filename": doc.get("filename", "Unknown"),
                "character_count": doc.get("character_count", 0),
      
    except Exception as e:
        error_msg = f"Rebuild index error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500
        @app.route("/debug-knowledge", methods=["POST"])
def debug_knowledge_search():
    """ADMIN ONLY: Debug knowledge base search functionality"""
    if not is_admin_user():
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        data = request.get_json()
        query = data.get("query", "")
        
        if not query:
            return jsonify({"error": "Query parameter required"}), 400
        
        print(f"🔍 DEBUG KNOWLEDGE: Testing search for '{query}'")
        
        # Load knowledge base
        knowledge_base = load_knowledge_base()
        
        # Test vector search
        admin_results = query_vector_store(query, "admin_kb", n_results=5)
        
        # Test retrieval system
        test_user_id = "debug-test-user"
        retrieval_result = retrieve_relevant_context(test_user_id, query, max_tokens=5000)
        
        debug_response = {
            "search_query": query,
            "timestamp": datetime.now().isoformat(),
            "total_documents": len(knowledge_base.get("documents", [])),
            "vector_results_count": len(admin_results),
            "generated_context_length": len(retrieval_result.get("context", "")),
            "all_documents": [],
            "vector_results": [],
            "context_preview": retrieval_result.get("context", "")[:1000] + "..." if len(retrieval_result.get("context", "")) > 1000 else retrieval_result.get("context", ""),
            "knowledge_base_stats": {
                "total_characters": knowledge_base.get("total_characters", 0),
                "total_authors": len(knowledge_base.get("authorized_authors", [])),
                "last_updated": knowledge_base.get("last_updated")
            }
        }
        
        # Add document summaries
        for doc in knowledge_base.get("documents", [])[:10]:  # First 10 docs
            doc_summary = {
                "filename": doc.get("filename", "Unknown"),
                "character_count": doc.get("character_count", 0),
                "authors": doc.get("extracted_authors", []),
                "content_preview": (doc.get("content", "")[:200] + "...") if len(doc.get("content", "")) > 200 else doc.get("content", "")
            }
            debug_response["all_documents"].append(doc_summary)
        
        # Add vector search results
        for result in admin_results[:5]:
            vector_result = {
                "filename": result["metadata"].get("filename", "Unknown"),
                "similarity": result["similarity"],
                "chunk_index": result["metadata"].get("chunk_index", 0),
                "preview": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
            }
            debug_response["vector_results"].append(vector_result)
        
        return jsonify(debug_response)
        
    except Exception as e:
        print(f"❌ Debug knowledge search error: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/debug-retrieval", methods=["GET"])
def debug_retrieval():
    """ADMIN ONLY: Test retrieval system without running chat"""
    if not is_admin_user():
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        query = request.args.get('query', '')
        if not query:
            return jsonify({"error": "Missing 'query' parameter"}), 400
        
        print(f"🔍 DEBUG RETRIEVAL: Testing query '{query}'")
        
        # Test admin KB retrieval
        admin_results = query_vector_store(query, "admin_kb", n_results=8)
        
        # Test user docs retrieval (use first user if any exists)
        user_results = []
        with get_db_connection() as conn:
            first_user = conn.execute('SELECT id FROM users LIMIT 1').fetchone()
            if first_user:
                user_results = query_vector_store(query, "user_docs", first_user['id'], n_results=6)
        
        # Get full retrieval context for a test user
        test_user_id = first_user['id'] if first_user else "test-user"
        retrieval_result = retrieve_relevant_context(test_user_id, query)
        
        debug_output = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "admin_results": {
                "count": len(admin_results),
                "results": [
                    {
                        "filename": r["metadata"].get("filename", "Unknown"),
                        "similarity": r["similarity"],
                        "chunk_index": r["metadata"].get("chunk_index", 0),
                        "preview": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
                    }
                    for r in admin_results
                ]
            },
            "user_results": {
                "count": len(user_results),
                "results": [
                    {
                        "filename": r["metadata"].get("filename", "Unknown"),
                        "similarity": r["similarity"],
                        "chunk_index": r["metadata"].get("chunk_index", 0),
                        "preview": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
                    }
                    for r in user_results
                ]
            },
            "full_retrieval": {
                "context_length": len(retrieval_result["context"]),
                "token_count": retrieval_result["token_count"],
                "chunks_used": retrieval_result["chunks_used"],
                "debug_info": retrieval_result["debug_info"]
            },
            "collections_status": {
                "admin_kb_exists": get_or_create_collection("admin_kb") is not None,
                "user_docs_exists": get_or_create_collection("user_docs") is not None
            }
        }
        
        return jsonify(debug_output)
        
    except Exception as e:
        error_msg = f"Debug retrieval error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

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
# ENHANCED CHAT ROUTE WITH VECTOR RETRIEVAL
# ================================

@app.route("/chat", methods=["POST"])
def chat():
    """ENHANCED Chat endpoint with vector retrieval and Claude primary"""
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

IMPORTANT: You have access to extensive curated therapeutic knowledge and user conversation history through advanced semantic vector search. 
When you reference information from the knowledge base, ALWAYS mention the specific document name and relevance score.
When you find relevant information, say something like "According to [Document Name] (relevance: X.XXX)..." or "As mentioned in [Book Title] with high relevance...".

If asked about specific books or resources, the vector search system will automatically find and provide the most relevant content."""

        agent_prompts = {
            "case_assistant": f"{base_instruction} You assist with social work case analysis using evidence-based approaches.",
            "research_critic": f"{base_instruction} You critically evaluate research using evidence-based approaches and cite specific sources with relevance scores.",
            "therapy_planner": f"{base_instruction} You plan therapeutic interventions using proven methodologies from your knowledge base.",
            "therapist": f"{base_instruction} {config.get('claude_system_prompt', 'You provide therapeutic support using systemic approaches.')}"
        }

        system_prompt = agent_prompts.get(agent, agent_prompts["therapist"])

        # Get AI response with enhanced retrieval (Claude primary, OpenAI fallback)
        response_text = call_model_with_enhanced_retrieval("claude", system_prompt, user_input, user_id)
        
        # Save AI response
        save_conversation_message(user_id, "assistant", response_text, agent)

        print(f"✅ CHAT RESPONSE: Generated {len(response_text):,} chars for user {user_id}")

        return jsonify({
            "response": response_text, 
            "user_id": user_id,
            "username": user.get('username', ''),
            "context_included": True,
            "retrieval_enhanced": True
        })
        
    except Exception as e:
        print(f"❌ CHAT ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ================================
# ENHANCED UPLOAD ROUTE WITH VECTOR INDEXING
# ================================

@app.route("/upload", methods=["POST"])
def upload():
    """ENHANCED file upload endpoint with vector indexing"""
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
            # ADMIN UPLOAD WITH VECTOR INDEXING
            if not is_admin_user():
                print("❌ UPLOAD: Not admin user")
                return jsonify({
                    "message": "Access denied. Admin privileges required.", 
                    "success": False
                }), 403
            
            print("🔍 UPLOAD: Processing admin upload with vector indexing...")
            file_path = os.path.join(UPLOADS_DIR, f"admin_{int(time.time())}_{file.filename}")
            
            # Ensure directory exists
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            
            # Save file
            file.save(file_path)
            print(f"✅ UPLOAD: Admin file saved to {file_path}")
            
            # Process document with improved method + vector indexing
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
                "message": f"✅ Added '{file.filename}' to global knowledge base with vector indexing ({doc_info['character_count']:,} characters){authors_text}", 
                "success": True,
                "type": "admin",
                "extracted_authors": doc_info['extracted_authors'],
                "character_count": doc_info['character_count'],
                "vector_indexed": True
            })
        
        else:
            # PERSONAL UPLOAD WITH VECTOR INDEXING
            print("🔍 UPLOAD: Processing personal upload with vector indexing...")
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
                "message": f"✅ Added '{file.filename}' to your persistent documents with vector indexing ({doc_info['character_count']:,} characters)", 
                "success": True,
                "type": "personal",
                "character_count": doc_info['character_count'],
                "vector_indexed": doc_info.get('vector_indexed', False)
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
    """Clear personal documents and vector indexes"""
    try:
        user = get_or_create_user()
        if not user:
            return jsonify({"error": "User session required"}), 401
        
        user_id = user['id']
        cleared_count = clear_user_documents(user_id)
        print(f"✅ CLEAR DOCS: Cleared {cleared_count} personal documents and vector indexes for user {user_id}")
        return jsonify({"message": f"Cleared {cleared_count} personal documents and vector indexes"})
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
                "character_count": doc["character_count"],
                "vector_indexed": doc.get("vector_indexed", False)
            }
            for doc in docs
        ]
        
        return jsonify({"personal_documents": doc_summaries, "user_id": user_id})
    except Exception as e:
        print(f"❌ PERSONAL DOCS ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/knowledge-base", methods=["GET"])
def get_knowledge_base():
    """Get knowledge base status with vector indexing stats"""
    try:
        knowledge_base = load_knowledge_base()
        
        # Get database stats including vector indexing
        with get_db_connection() as conn:
            total_users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
            active_users = conn.execute(
                'SELECT COUNT(*) as count FROM users WHERE last_active > datetime("now", "-30 days")'
            ).fetchone()['count']
            total_conversations = conn.execute('SELECT COUNT(*) as count FROM conversations').fetchone()['count']
            
            # Vector indexing stats
            admin_indexed = conn.execute(
                'SELECT COUNT(*) as count FROM knowledge_base_docs WHERE vector_indexed = 1'
            ).fetchone()['count']
            user_indexed = conn.execute(
                'SELECT COUNT(*) as count FROM user_documents WHERE vector_indexed = 1 AND is_active = 1'
            ).fetchone()['count']
        
        return jsonify({
            "total_documents": len(knowledge_base["documents"]),
            "total_characters": knowledge_base.get("total_characters", 0),
            "last_updated": knowledge_base.get("last_updated"),
            "authorized_authors": knowledge_base.get("authorized_authors", []),
            "total_authors": len(knowledge_base.get("authorized_authors", [])),
            "vector_indexing": {
                "admin_docs_indexed": admin_indexed,
                "user_docs_indexed": user_indexed,
                "total_indexed": admin_indexed + user_indexed
            },
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
            
            vector_indexed_count = conn.execute(
                'SELECT COUNT(*) as count FROM user_documents WHERE user_id = ? AND is_active = 1 AND vector_indexed = 1', (user_id,)
            ).fetchone()['count']
            
            days_active = conn.execute('''
                SELECT CAST((julianday('now') - julianday(created_at)) AS INTEGER) as days
                FROM users WHERE id = ?
            ''', (user_id,)).fetchone()['days']
        
        return jsonify({
            "user_id": user_id,
            "conversation_messages": conversation_count,
            "personal_documents": document_count,
            "vector_indexed_documents": vector_indexed_count,
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
                    "character_count": doc["character_count"],
                    "vector_indexed": doc.get("vector_indexed", False)
                }
                for doc in docs
            ],
            "stats": {
                "total_messages": len(history),
                "total_documents": len(docs),
                "vector_indexed_docs": len([d for d in docs if d.get("vector_indexed", False)])
            }
        }
        
        return jsonify(export_data)
    except Exception as e:
        print(f"❌ LOG ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    """System health check with vector store diagnostics"""
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
                claude_works = test_response is not None and not test_response.startswith("Error")
            except:
                pass
        
        # Database stats
        with get_db_connection() as conn:
            total_users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
            authenticated_users = conn.execute('SELECT COUNT(*) as count FROM users WHERE email IS NOT NULL').fetchone()['count']
            total_conversations = conn.execute('SELECT COUNT(*) as count FROM conversations').fetchone()['count']
            total_user_docs = conn.execute('SELECT COUNT(*) as count FROM user_documents WHERE is_active = 1').fetchone()['count']
            admin_indexed = conn.execute('SELECT COUNT(*) as count FROM knowledge_base_docs WHERE vector_indexed = 1').fetchone()['count']
            user_indexed = conn.execute('SELECT COUNT(*) as count FROM user_documents WHERE vector_indexed = 1').fetchone()['count']
        
        knowledge_base = load_knowledge_base()
        
        # Test vector store
        vector_store_status = {
            "chromadb_initialized": chroma_client is not None,
            "admin_collection_exists": get_or_create_collection("admin_kb") is not None,
            "user_collection_exists": get_or_create_collection("user_docs") is not None
        }
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "enhanced_vector_v3.0",
            "apis": {
                "openai_configured": openai_api_key is not None,
                "openai_working": openai_works,
                "claude_configured": claude_api_key is not None,
                "claude_working": claude_works,
                "primary_model": "claude",
                "fallback_model": "openai"
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
            "vector_store": {
                **vector_store_status,
                "admin_docs_indexed": admin_indexed,
                "user_docs_indexed": user_indexed,
                "total_indexed": admin_indexed + user_indexed,
                "embedding_model": "text-embedding-3-small"
            },
            "features": {
                "user_authentication": True,
                "anonymous_sessions": True,
                "persistent_memory": True,
                "conversation_continuity": True,
                "personal_documents": True,
                "knowledge_base": True,
                "vector_search": True,
                "semantic_retrieval": True,
                "claude_primary": True,
                "openai_fallback": True,
                "pdf_extraction": "pdfplumber_improved",
                "auto_chunking": True,
                "token_budget_management": True,
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
            "retrieval_config": {
                "max_tokens": 20000,
                "admin_chunks_per_query": 8,
                "user_chunks_per_query": 6,
                "chunk_size_words": 1000,
                "chunk_overlap_words": 150,
                "embedding_dimensions": 1536
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
    """Process batch upload via API with vector indexing"""
    if not is_admin_user():
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        # Get uploaded files
        files = request.files.getlist('pdfs')
        if not files:
            return jsonify({"error": "No files uploaded"}), 400
        
        print(f"🔍 BATCH UPLOAD: Processing {len(files)} files with vector indexing")
        results = []
        
        for i, file in enumerate(files):
            if file.filename and file.filename.lower().endswith('.pdf'):
                try:
                    print(f"🔍 BATCH UPLOAD: Processing file {i+1}/{len(files)}: {file.filename}")
                    
                    # Save file temporarily
                    temp_path = os.path.join(UPLOADS_DIR, f"batch_{int(time.time())}_{i}_{file.filename}")
                    file.save(temp_path)
                    
                    file_size_mb = os.path.getsize(temp_path) / 1024 / 1024
                    
                    # Process with improved method + vector indexing
                    doc_info = add_document_to_knowledge_base(temp_path, file.filename, is_core=True)
                    
                    if not doc_info.get('error'):
                        results.append({
                            'filename': file.filename,
                            'status': 'success',
                            'size_mb': round(file_size_mb, 2),
                            'character_count': doc_info.get('character_count', 0),
                            'authors': doc_info.get('extracted_authors', []),
                            'vector_indexed': True
                        })
                        print(f"✅ BATCH UPLOAD: Successfully processed {file.filename} with vector indexing")
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
            "message": f"Batch upload completed with vector indexing: {successful} successful, {failed} failed",
            "results": results,
            "summary": {
                "total": len(results),
                "successful": successful,
                "failed": failed,
                "vector_indexed": successful
            }
        })
        
    except Exception as e:
        print(f"❌ BATCH UPLOAD ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ================================
# INITIALIZATION SYSTEM
# ================================

def initialize_enhanced_system():
    """Initialize system with vector store enhancements"""
    print("🚀 Initializing ENHANCED Vector-Powered Therapeutic AI System...")
    
    # Set memory limits
    limit_memory()
    
    # Initialize database
    init_database()
    
    # Initialize ChromaDB
    global chroma_client
    chroma_client = init_chroma()
    
    if chroma_client:
        print("✅ ChromaDB vector store initialized")
        
        # Test collections
        admin_collection = get_or_create_collection("admin_kb")
        user_collection = get_or_create_collection("user_docs")
        
        if admin_collection and user_collection:
            print("✅ Vector collections ready: admin_kb, user_docs")
        else:
            print("⚠️ Some vector collections failed to initialize")
    else:
        print("❌ ChromaDB initialization failed - vector search will not work")
    
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
    
    # Check database health including vector indexing
    try:
        with get_db_connection() as conn:
            total_users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
            authenticated_users = conn.execute('SELECT COUNT(*) as count FROM users WHERE email IS NOT NULL').fetchone()['count']
            total_conversations = conn.execute('SELECT COUNT(*) as count FROM conversations').fetchone()['count']
            admin_indexed = conn.execute('SELECT COUNT(*) as count FROM knowledge_base_docs WHERE vector_indexed = 1').fetchone()['count']
            user_indexed = conn.execute('SELECT COUNT(*) as count FROM user_documents WHERE vector_indexed = 1').fetchone()['count']
            print(f"👥 Database health: {total_users} users ({authenticated_users} registered), {total_conversations} conversations")
            print(f"🔍 Vector indexing: {admin_indexed} admin docs, {user_indexed} user docs indexed")
    except Exception as e:
        print(f"⚠️ Database check failed: {e}")
    
    # Test API connections
    api_status = []
    if claude_api_key:
        api_status.append("Claude API configured (PRIMARY)")
    if openai_api_key:
        api_status.append("OpenAI API configured (FALLBACK + Embeddings)")
    
    if api_status:
        print(f"🔑 APIs: {', '.join(api_status)}")
    else:
        print("⚠️ No API keys configured")
    
    # Force initial garbage collection
    gc.collect()
    
    print("✅ ENHANCED Vector-Powered Therapeutic AI System initialized!")
    print("🎯 New features active:")
    print("   - Claude as primary therapist voice")
    print("   - OpenAI GPT-4 as fallback model")
    print("   - OpenAI text-embedding-3-small for semantic search")
    print("   - ChromaDB persistent vector store")
    print("   - Two collections: admin_kb + user_docs")
    print("   - Auto-chunking: ~1,000 words, 150-word overlap")
    print("   - Retrieval system: top 8-10 most relevant chunks")
    print("   - Token budget control: ~20-25k tokens max")
    print("   - Debug logging with relevance scores")
    print("   - /rebuild-index admin route")
    print("   - /debug-retrieval admin route")
    print("   - Enhanced PDF processing (up to 2000 pages)")
    print("   - Improved author extraction")
    print("   - Better error handling and memory management")
    print()
    print("🔍 Vector Search Features:")
    print("   - Semantic similarity search with OpenAI embeddings")
    print("   - Automatic relevance scoring")
    print("   - Context-aware chunk selection")
    print("   - Token budget management")
    print("   - Conversation history integration")
    print("   - Document-specific source attribution")
    print()
    print("🛠️ Admin Debug Tools:")
    print("   - POST /rebuild-index - Rebuild all vector indexes")
    print("   - GET /debug-retrieval?query=... - Test retrieval system")
    print("   - Enhanced health endpoint with vector diagnostics")
    print("   - Real-time chunk and relevance logging")

# ================================
# APPLICATION STARTUP
# ================================

if __name__ == "__main__":
    initialize_enhanced_system()
    app.run(host="0.0.0.0", port=5000, debug=False)
else:
    initialize_enhanced_system()
    application = app
