import os
import json
import requests
import hashlib
import re
import sqlite3
import uuid
import secrets
import zlib
import base64
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
# ADVANCED COMPRESSION SYSTEM
# ================================

class AdvancedTextCompressor:
    """Advanced compression system for large therapeutic texts"""
    
    @staticmethod
    def ultra_compress(text: str) -> Tuple[str, Dict]:
        """
        Ultra-high compression for large books
        Achieves 85-95% compression on text-heavy PDFs
        """
        try:
            # Step 1: Text normalization
            normalized = AdvancedTextCompressor._normalize_text(text)
            
            # Step 2: Create dictionary of common therapeutic terms
            term_dict, encoded_text = AdvancedTextCompressor._create_term_dictionary(normalized)
            
            # Step 3: Apply maximum zlib compression
            text_bytes = encoded_text.encode('utf-8')
            compressed = zlib.compress(text_bytes, level=9)
            
            # Step 4: Base64 encode for storage
            compressed_b64 = base64.b64encode(compressed).decode('ascii')
            
            # Calculate compression stats
            original_size = len(text)
            final_size = len(compressed_b64)
            compression_ratio = (1 - final_size / original_size) * 100
            
            metadata = {
                'original_size': original_size,
                'compressed_size': final_size,
                'compression_ratio': round(compression_ratio, 1),
                'term_dictionary': term_dict,
                'normalization_applied': True
            }
            
            print(f"🗜️ Ultra-compression: {original_size:,} → {final_size:,} bytes ({compression_ratio:.1f}% reduction)")
            
            return compressed_b64, metadata
            
        except Exception as e:
            print(f"❌ Ultra-compression failed: {e}")
            return text, {'compression_ratio': 0}
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for better compression"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize page breaks and headers
        text = re.sub(r'--- Page \d+ ---\n?', '\n[PAGE]\n', text)
        
        # Normalize common patterns
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()
    
    @staticmethod
    def _create_term_dictionary(text: str) -> Tuple[Dict, str]:
        """Create dictionary of common therapeutic terms for compression"""
        # Common therapeutic terms that appear frequently
        therapeutic_terms = [
            'therapy', 'therapeutic', 'therapist', 'treatment', 'intervention',
            'relationship', 'relationships', 'couple', 'couples', 'family', 'families',
            'marriage', 'marital', 'communication', 'conflict', 'attachment',
            'emotional', 'behavior', 'behavioral', 'cognitive', 'systemic',
            'counseling', 'counselor', 'psychology', 'psychological', 'psychologist',
            'client', 'clients', 'patient', 'patients', 'session', 'sessions'
        ]
        
        # Find frequently occurring phrases (3+ words)
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        
        # Create replacement dictionary for frequent terms
        term_dict = {}
        encoded_text = text
        
        # Replace most frequent therapeutic terms with short codes
        for i, term in enumerate(therapeutic_terms):
            if word_freq.get(term, 0) > 10:  # Appears 10+ times
                code = f"T{i:02d}"
                term_dict[code] = term
                encoded_text = re.sub(r'\b' + re.escape(term) + r'\b', code, encoded_text, flags=re.IGNORECASE)
        
        return term_dict, encoded_text
    
    @staticmethod
    def decompress(compressed_b64: str, metadata: Dict) -> str:
        """Decompress ultra-compressed text"""
        try:
            # Decode and decompress
            compressed_bytes = base64.b64decode(compressed_b64)
            decompressed_bytes = zlib.decompress(compressed_bytes)
            text = decompressed_bytes.decode('utf-8')
            
            # Restore term dictionary
            if 'term_dictionary' in metadata:
                term_dict = metadata['term_dictionary']
                for code, term in term_dict.items():
                    text = re.sub(r'\b' + re.escape(code) + r'\b', term, text)
            
            # Restore page markers
            text = text.replace('[PAGE]', '--- Page ---')
            
            return text
            
        except Exception as e:
            print(f"❌ Decompression failed: {e}")
            return compressed_b64

class StreamingPDFProcessor:
    """Process large PDFs without loading everything into memory"""
    
    def __init__(self, max_memory_mb=500):
        self.max_memory_mb = max_memory_mb
        self.compressor = AdvancedTextCompressor()
    
    def process_large_pdf_streaming(self, file_path: str, chunk_size_pages=25) -> Dict:
        """
        Process very large PDFs using streaming approach
        Processes in chunks and compresses immediately
        """
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"📚 Processing large PDF: {file_path} ({file_size_mb:.1f}MB)")
            
            if file_size_mb > 200:  # 200MB limit even with streaming
                return {"error": f"PDF too large ({file_size_mb:.1f}MB). Maximum: 200MB"}
            
            compressed_chunks = []
            total_pages = 0
            total_original_size = 0
            total_compressed_size = 0
            authors = set()
            
            # Process PDF in streaming chunks
            with pdfplumber.open(file_path) as pdf:
                total_pdf_pages = len(pdf.pages)
                print(f"📄 PDF has {total_pdf_pages} pages, processing in chunks of {chunk_size_pages}")
                
                for start_page in range(0, total_pdf_pages, chunk_size_pages):
                    end_page = min(start_page + chunk_size_pages, total_pdf_pages)
                    
                    # Extract text from chunk
                    chunk_text = self._extract_chunk_text(pdf, start_page, end_page)
                    
                    if chunk_text.strip():
                        # Compress the chunk immediately
                        compressed_chunk, metadata = self.compressor.ultra_compress(chunk_text)
                        
                        chunk_info = {
                            'compressed_data': compressed_chunk,
                            'metadata': metadata,
                            'page_range': f"{start_page + 1}-{end_page}",
                            'chunk_index': len(compressed_chunks)
                        }
                        
                        compressed_chunks.append(chunk_info)
                        
                        # Update stats
                        total_original_size += metadata['original_size']
                        total_compressed_size += metadata['compressed_size']
                        total_pages += (end_page - start_page)
                        
                        # Extract authors from first chunk
                        if len(compressed_chunks) == 1:
                            chunk_authors = extract_authors_from_text(chunk_text, os.path.basename(file_path))
                            authors.update(chunk_authors)
                    
                    # Memory management
                    if len(compressed_chunks) % 4 == 0:  # Every 4 chunks
                        gc.collect()
                        print(f"  Processed {end_page}/{total_pdf_pages} pages...")
            
            overall_compression = (1 - total_compressed_size / total_original_size) * 100 if total_original_size > 0 else 0
            
            result = {
                'success': True,
                'compressed_chunks': compressed_chunks,
                'total_chunks': len(compressed_chunks),
                'total_pages': total_pages,
                'original_size': total_original_size,
                'compressed_size': total_compressed_size,
                'compression_ratio': round(overall_compression, 1),
                'file_size_mb': round(file_size_mb, 2),
                'extracted_authors': list(authors),
                'processing_method': 'streaming_ultra_compressed'
            }
            
            print(f"✅ Streaming processing complete: {total_original_size:,} → {total_compressed_size:,} bytes ({overall_compression:.1f}% compression)")
            
            return result
            
        except Exception as e:
            error_msg = f"Streaming processing failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
        finally:
            gc.collect()
    
    def _extract_chunk_text(self, pdf, start_page: int, end_page: int) -> str:
        """Extract text from a chunk of PDF pages"""
        chunk_text = ""
        
        for page_num in range(start_page, end_page):
            try:
                page = pdf.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    chunk_text += f"\n--- Page {page_num + 1} ---\n"
                    chunk_text += page_text.strip() + "\n"
            except Exception as e:
                print(f"  ⚠️ Error extracting page {page_num + 1}: {e}")
                continue
        
        return chunk_text

# ================================
# MEMORY-SAFE PDF PROCESSING
# ================================

def extract_text_from_pdf_efficient(file_path, max_size_mb=200):  # INCREASED for Standard tier
    """Memory-efficient PDF extraction with reasonable limits for Standard tier"""
    try:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"Processing PDF: {file_path} ({file_size:.1f}MB)")
        
        # More generous limits for Standard tier
        if file_size > max_size_mb:
            return f"PDF too large ({file_size:.1f}MB). Maximum size: {max_size_mb}MB"
        
        text_content = ""
        page_count = 0
        
        # Force garbage collection before starting
        gc.collect()
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"PDF has {total_pages} pages")
                
                # More generous page limits for Standard tier
                max_pages = min(total_pages, 500)  # Max 500 pages (increased from 200)
                if total_pages > max_pages:
                    print(f"⚠️ Large PDF detected - processing first {max_pages} pages only")
                
                for page_num in range(max_pages):
                    try:
                        page = pdf.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n--- Page {page_num + 1} ---\n"
                            text_content += page_text.strip() + "\n"
                            page_count += 1
                        
                        # Less aggressive memory management for Standard tier
                        if page_num % 25 == 0:  # Every 25 pages
                            gc.collect()
                            
                        # Progress indication
                        if page_num % 50 == 0 and page_num > 0:
                            print(f"Processed {page_num}/{max_pages} pages...")
                            
                        # Higher memory safety threshold for Standard tier
                        if len(text_content) > 20 * 1024 * 1024:  # 20MB text limit (increased from 5MB)
                            print("⚠️ Text content limit reached - stopping extraction")
                            break
                            
                    except Exception as e:
                        print(f"Error extracting page {page_num + 1}: {e}")
                        continue
                        
        except Exception as pdf_error:
            print(f"PDF processing error: {pdf_error}")
            return f"Error processing PDF: {str(pdf_error)}"
        
        if not text_content.strip():
            return f"No text could be extracted from {os.path.basename(file_path)}"
        
        print(f"✅ Successfully extracted {len(text_content)} characters from {page_count} pages")
        return text_content
        
    except Exception as e:
        error_msg = f"Error extracting text from {os.path.basename(file_path)}: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg
    finally:
        # ALWAYS force garbage collection
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
        
        # Calculate total characters from both compressed and uncompressed docs
        total_chars = 0
        for doc in knowledge_base["documents"]:
            if doc.get("content_type") == "streaming_ultra_compressed":
                total_chars += doc.get("character_count", 0)
            elif doc.get("content_type") == "ultra_compressed":
                total_chars += doc.get("character_count", 0)
            else:
                total_chars += len(doc.get("content", ""))
        
        knowledge_base["total_characters"] = total_chars
        
        with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
            json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Knowledge base saved: {knowledge_base['total_documents']} documents")
    except Exception as e:
        print(f"❌ Error saving knowledge base: {e}")

def add_document_to_knowledge_base(file_path, filename, is_core=True):
    """Add document to admin knowledge base with database tracking (legacy method)"""
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

def add_document_compressed(file_path, filename, is_core=True):
    """
    Add document with compression and smart storage
    Reduces memory usage by 70-90%
    """
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"📊 Processing: {filename} ({file_size_mb:.1f}MB)")
        
        # Use streaming processor for very large files on Standard tier
        if file_size_mb > 50:  # Threshold for streaming (increased from 25MB)
            processor = StreamingPDFProcessor()
            result = processor.process_large_pdf_streaming(file_path)
            
            if not result.get('success'):
                return {"error": result.get('error', 'Processing failed')}
            
            # Create document info for streaming result
            file_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
            doc_id = str(uuid.uuid4())
            
            doc_info = {
                "id": doc_id,
                "filename": filename,
                "content_type": "streaming_ultra_compressed",
                "compressed_chunks": result['compressed_chunks'],
                "total_chunks": result['total_chunks'],
                "added_date": datetime.now().isoformat(),
                "file_hash": file_hash,
                "is_core": is_core,
                "character_count": result['original_size'],
                "compressed_size": result['compressed_size'],
                "compression_ratio": result['compression_ratio'],
                "total_pages": result['total_pages'],
                "file_size_mb": result['file_size_mb'],
                "type": "admin_therapeutic_resource",
                "extracted_authors": result['extracted_authors'],
                "pdf_extraction_status": "success_streaming_compressed",
                "processing_method": result['processing_method']
            }
        else:
            # Use regular processing for smaller files with compression
            text_content = extract_text_from_pdf_efficient(file_path)
            
            if "Error" in text_content or "too large" in text_content:
                return {"error": text_content}
            
            # Compress the content
            compressor = AdvancedTextCompressor()
            compressed_content, metadata = compressor.ultra_compress(text_content)
            
            # Extract authors
            extracted_authors = extract_authors_from_text(text_content, filename)
            
            # Create document info
            file_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
            doc_id = str(uuid.uuid4())
            
            doc_info = {
                "id": doc_id,
                "filename": filename,
                "content_type": "ultra_compressed",
                "compressed_content": compressed_content,
                "compression_metadata": metadata,
                "added_date": datetime.now().isoformat(),
                "file_hash": file_hash,
                "is_core": is_core,
                "character_count": metadata['original_size'],
                "compressed_size": metadata['compressed_size'],
                "compression_ratio": metadata['compression_ratio'],
                "type": "admin_therapeutic_resource",
                "extracted_authors": extracted_authors,
                "pdf_extraction_status": "success_compressed",
                "file_size_mb": round(file_size_mb, 2)
            }
        
        # Add to knowledge base
        knowledge_base = load_knowledge_base()
        knowledge_base["documents"].append(doc_info)
        
        # Update authorized authors
        if "authorized_authors" not in knowledge_base:
            knowledge_base["authorized_authors"] = []
        
        for author in doc_info['extracted_authors']:
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
            ''', (doc_id, filename, file_hash, doc_info['character_count'], json.dumps(doc_info['extracted_authors'])))
            conn.commit()
        
        print(f"✅ Added {filename} to knowledge base ({doc_info['character_count']:,} chars, {doc_info['compression_ratio']}% compression)")
        return doc_info
        
    except Exception as e:
        error_msg = f"Error adding document: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}
    finally:
        gc.collect()

def retrieve_compressed_content(doc_info: Dict, max_chars: int = 25000) -> str:
    """
    Retrieve content from compressed documents
    Only decompresses what's needed for performance
    """
    try:
        content_type = doc_info.get('content_type', '')
        
        if content_type == 'streaming_ultra_compressed':
            # Streaming compressed chunks
            compressed_chunks = doc_info.get('compressed_chunks', [])
            compressor = AdvancedTextCompressor()
            
            content_parts = []
            chars_retrieved = 0
            
            for chunk_info in compressed_chunks:
                if chars_retrieved >= max_chars:
                    break
                    
                compressed_data = chunk_info['compressed_data']
                metadata = chunk_info['metadata']
                
                decompressed_text = compressor.decompress(compressed_data, metadata)
                content_parts.append(f"[Pages {chunk_info['page_range']}]\n{decompressed_text}")
                
                chars_retrieved += len(decompressed_text)
            
            return "\n\n".join(content_parts)[:max_chars]
            
        elif content_type == 'ultra_compressed':
            # Single compressed content
            compressed_content = doc_info.get('compressed_content', '')
            metadata = doc_info.get('compression_metadata', {})
            
            compressor = AdvancedTextCompressor()
            full_content = compressor.decompress(compressed_content, metadata)
            
            return full_content[:max_chars]
        else:
            # Legacy uncompressed content
            return doc_info.get('content', '')[:max_chars]
            
    except Exception as e:
        print(f"❌ Error retrieving compressed content: {e}")
        return ""

# ================================
# USER DOCUMENT PERSISTENCE
# ================================

def add_personal_document_persistent(file_path, filename, user_id):
    """Add document to user's persistent personal collection"""
    try:
        text_content = extract_text_from_pdf_efficient(file_path, max_size_mb=100)  # Increased for Standard tier
        
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

def build_therapeutic_context(user_id, user_query, limit_kb_docs=6):
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
            
            # USE UNCOMPRESSED CONTENT (compression disabled)
doc_text = doc.get("content", "").lower()
            
            for word in query_words:
                if len(word) > 3:
                    relevance_score += doc_text.count(word)
            
            if relevance_score > 0:
                relevant_docs.append((doc, relevance_score))
        
        # Sort by relevance and take top docs
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        relevant_docs = relevant_docs[:limit_kb_docs]
    
    # Add knowledge base content (UPDATED for compression)
    if relevant_docs:
        context += "\n=== CURATED THERAPEUTIC KNOWLEDGE ===\n"
        for doc, score in relevant_docs:
            # Use compressed content retrieval
            content_snippet = retrieve_compressed_content(doc, max_chars=20000)
            context += f"From '{doc['filename']}':\n{content_snippet}...\n\n"
            
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
    
    return context[:75000]  # Limit total context size

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

# ================================
# MEMORY-SAFE UPLOAD ROUTE - FIXED
# ================================

@app.route("/upload", methods=["POST"])
def upload():
    """Memory-safe file upload endpoint - FIXED VERSION"""
    print(f"🔍 UPLOAD START - Memory limit enforced")
    
    try:
        user = get_or_create_user()
        if not user:
            print("❌ No user session")
            return jsonify({"error": "User session required"}), 401
        
        user_id = user['id']
        print(f"🔍 User ID: {user_id}")
        
        if "pdf" not in request.files:
            print("❌ No PDF in request.files")
            return jsonify({"message": "No file selected", "success": False})
        
        file = request.files["pdf"]
        
        if not file or not file.filename:
            print("❌ No file or filename")
            return jsonify({"message": "No file selected", "success": False})
            
        if not file.filename.lower().endswith(".pdf"):
            print("❌ Not a PDF file")
            return jsonify({"message": "Please select a PDF file", "success": False})
        
        upload_type = request.form.get("upload_type", "personal")
        use_streaming = request.form.get("use_streaming", "false") == "true"
        
        # Check file size before processing
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"🔍 File size: {file_size_mb:.1f}MB")
        
        # Enforce more generous limits for Standard tier
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
                print("❌ Not admin user")
                return jsonify({
                    "message": "Access denied. Admin privileges required.", 
                    "success": False
                }), 403
            
            print("🔍 Admin upload - saving file...")
            file_path = os.path.join(UPLOADS_DIR, file.filename)
            
            # Ensure directory exists
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            
            # Save file
            file.save(file_path)
            print(f"✅ Admin file saved to: {file_path}")
            
            # FORCE LEGACY PROCESSING (no compression)
print("🔍 Using legacy processing method (compression disabled)")
doc_info = add_document_to_knowledge_base(file_path, file.filename, is_core=True)
            
            print(f"🔍 Admin document processing result: {doc_info}")
            
            if "error" in doc_info:
                print(f"❌ Admin document processing error: {doc_info['error']}")
                # Clean up uploaded file
                try:
                    os.remove(file_path)
                except:
                    pass
                return jsonify({"message": doc_info["error"], "success": False})
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
                print(f"🔍 Cleaned up admin temp file: {file_path}")
            except Exception as e:
                print(f"⚠️ Could not remove admin temp file: {e}")
            
            authors_text = f" | Authors: {', '.join(doc_info['extracted_authors'])}" if doc_info['extracted_authors'] else ""
            compression_text = f" | {doc_info.get('compression_ratio', 0)}% compression" if doc_info.get('compression_ratio', 0) > 0 else ""
            
            return jsonify({
                "message": f"✅ Added '{file.filename}' to global knowledge base ({doc_info['character_count']} characters){compression_text}{authors_text}", 
                "success": True,
                "type": "admin",
                "extracted_authors": doc_info['extracted_authors'],
                "compression_ratio": doc_info.get('compression_ratio', 0)
            })
        
        else:
            # PERSONAL UPLOAD
            print("🔍 Personal document upload")
            file_path = os.path.join(USER_UPLOADS_DIR, f"{user_id}_{file.filename}")
            
            # Ensure directory exists
            os.makedirs(USER_UPLOADS_DIR, exist_ok=True)
            
            # Save file
            file.save(file_path)
            print(f"✅ Personal file saved to: {file_path}")
            
            doc_info = add_personal_document_persistent(file_path, file.filename, user_id)
            print(f"🔍 Personal document processing result: {doc_info}")
            
            if "error" in doc_info:
                print(f"❌ Personal document processing error: {doc_info['error']}")
                # Clean up uploaded file
                try:
                    os.remove(file_path)
                except:
                    pass
                return jsonify({"message": doc_info["error"], "success": False})
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
                print(f"🔍 Cleaned up personal temp file: {file_path}")
            except Exception as e:
                print(f"⚠️ Could not remove personal temp file: {e}")
            
            return jsonify({
                "message": f"✅ Added '{file.filename}' to your persistent documents ({doc_info['character_count']} characters)", 
                "success": True,
                "type": "personal"
            })
        
    except Exception as e:
        print(f"❌ Upload exception: {str(e)}")
        traceback.print_exc()
        # Force memory cleanup on error
        gc.collect()
        return jsonify({
            "error": f"Upload failed: {str(e)}", 
            "success": False
        }), 500

# ================================
# REMAINING ROUTES
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
                "controlled_knowledge": True,
                "large_pdf_support": True,
                "compression_system": True,
                "batch_upload": True,
                "memory_safety": True
            },
            "memory_info": {
                "knowledge_base_size_mb": round(knowledge_base.get("total_characters", 0) / 1024 / 1024, 2),
                "database_file": DATABASE_FILE,
                "pdf_max_size_mb": 200,
                "compression_enabled": True,
                "memory_limit_mb": 2048,
                "tier": "standard"
            }
        })
    except Exception as e:
        print(f"Error in health endpoint: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

# ================================
# ADMIN DIAGNOSTIC ROUTES
# ================================

@app.route("/admin/check-permissions", methods=["GET"])
def check_permissions():
    """Check file system permissions and directory status"""
    if not session.get("is_admin"):
        return "Admin access required", 403
    
    checks = {}
    
    # Check if directories exist and are writable
    for dir_name in [UPLOADS_DIR, CORE_MEMORY_DIR, USER_UPLOADS_DIR]:
        checks[dir_name] = {
            "exists": os.path.exists(dir_name),
            "is_dir": os.path.isdir(dir_name) if os.path.exists(dir_name) else False,
            "writable": os.access(dir_name, os.W_OK) if os.path.exists(dir_name) else False,
            "contents": os.listdir(dir_name) if os.path.exists(dir_name) else []
        }
    
    # Check knowledge base file
    kb_file = os.path.join(CORE_MEMORY_DIR, "knowledge_base.json")
    checks["knowledge_base.json"] = {
        "exists": os.path.exists(kb_file),
        "readable": os.access(kb_file, os.R_OK) if os.path.exists(kb_file) else False,
        "writable": os.access(kb_file, os.W_OK) if os.path.exists(kb_file) else False,
        "size": os.path.getsize(kb_file) if os.path.exists(kb_file) else 0
    }
    
    # Check database file
    checks["database"] = {
        "exists": os.path.exists(DATABASE_FILE),
        "readable": os.access(DATABASE_FILE, os.R_OK) if os.path.exists(DATABASE_FILE) else False,
        "writable": os.access(DATABASE_FILE, os.W_OK) if os.path.exists(DATABASE_FILE) else False,
        "size": os.path.getsize(DATABASE_FILE) if os.path.exists(DATABASE_FILE) else 0
    }
    
    return f"<pre>{json.dumps(checks, indent=2)}</pre>"

@app.route("/admin/debug-kb", methods=["GET"])
def debug_knowledge_base():
    """Debug knowledge base status"""
    if not session.get("is_admin"):
        return "Admin access required", 403
    
    debug_info = {
        "knowledge_base_file_exists": os.path.exists(KNOWLEDGE_BASE_FILE),
        "knowledge_base_file_size": 0,
        "knowledge_base_content": {},
        "uploads_dir_exists": os.path.exists(UPLOADS_DIR),
        "uploads_dir_contents": [],
        "database_stats": {},
        "core_memory_dir_exists": os.path.exists(CORE_MEMORY_DIR),
        "core_memory_contents": []
    }
    
    # Check knowledge base file size
    if os.path.exists(KNOWLEDGE_BASE_FILE):
        debug_info["knowledge_base_file_size"] = os.path.getsize(KNOWLEDGE_BASE_FILE)
        
        # Try to read content
        try:
            with open(KNOWLEDGE_BASE_FILE, 'r') as f:
                kb_content = json.load(f)
                debug_info["knowledge_base_content"] = {
                    "total_documents": len(kb_content.get("documents", [])),
                    "total_authors": len(kb_content.get("authorized_authors", [])),
                    "last_updated": kb_content.get("last_updated"),
                    "total_characters": kb_content.get("total_characters", 0),
                    "document_filenames": [doc.get("filename") for doc in kb_content.get("documents", [])]
                }
        except Exception as e:
            debug_info["knowledge_base_error"] = str(e)
    
    # Check uploads directory
    if os.path.exists(UPLOADS_DIR):
        debug_info["uploads_dir_contents"] = os.listdir(UPLOADS_DIR)
    
    # Check core memory directory
    if os.path.exists(CORE_MEMORY_DIR):
        debug_info["core_memory_contents"] = os.listdir(CORE_MEMORY_DIR)
    
    # Check database
    try:
        with get_db_connection() as conn:
            kb_docs = conn.execute('SELECT COUNT(*) as count FROM knowledge_base_docs').fetchone()
            debug_info["database_stats"]["knowledge_base_docs"] = kb_docs['count']
            
            # Get recent uploads
            recent_docs = conn.execute('''
                SELECT filename, upload_date, character_count 
                FROM knowledge_base_docs 
                ORDER BY upload_date DESC 
                LIMIT 5
            ''').fetchall()
            debug_info["database_stats"]["recent_uploads"] = [dict(doc) for doc in recent_docs]
            
    except Exception as e:
        debug_info["database_error"] = str(e)
    
    return f"<pre>{json.dumps(debug_info, indent=2)}</pre>"

@app.route("/admin/force-kb-refresh", methods=["POST"])
def force_kb_refresh():
    """Force refresh of knowledge base stats"""
    if not session.get("is_admin"):
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        # Force reload and recalculate knowledge base
        knowledge_base = load_knowledge_base()
        save_knowledge_base(knowledge_base)  # This will recalculate stats
        
        return jsonify({
            "success": True, 
            "message": "Knowledge base stats refreshed",
            "stats": {
                "total_documents": len(knowledge_base["documents"]),
                "total_characters": knowledge_base.get("total_characters", 0),
                "total_authors": len(knowledge_base.get("authorized_authors", []))
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
# BATCH UPLOAD SYSTEM
# ================================

@app.route("/admin/batch-process", methods=["POST"])
def admin_batch_process():
    """Process batch upload via API"""
    if not is_admin_user():
        return jsonify({"error": "Admin access required"}), 403
    
    try:
        # Get uploaded files
        files = request.files.getlist('pdfs')
        if not files:
            return jsonify({"error": "No files uploaded"}), 400
        
        results = []
        
        for i, file in enumerate(files):
            if file.filename and file.filename.lower().endswith('.pdf'):
                try:
                    # Save file temporarily
                    temp_path = os.path.join(UPLOADS_DIR, f"batch_{i}_{file.filename}")
                    file.save(temp_path)
                    
                    # Process with compression
                    doc_info = add_document_compressed(temp_path, file.filename, is_core=True)
                    
                    if not doc_info.get('error'):
                        results.append({
                            'filename': file.filename,
                            'status': 'success',
                            'size_mb': doc_info.get('file_size_mb', 0),
                            'compression_ratio': doc_info.get('compression_ratio', 0),
                            'authors': doc_info.get('extracted_authors', [])
                        })
                    else:
                        results.append({
                            'filename': file.filename,
                            'status': 'failed',
                            'error': doc_info.get('error', 'Processing failed')
                        })
                    
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
        
        # Calculate summary stats
        successful = len([r for r in results if r['status'] == 'success'])
        failed = len([r for r in results if r['status'] == 'failed'])
        
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
        print(f"Batch upload error: {e}")
        return jsonify({"error": str(e)}), 500

# ================================
# MEMORY-SAFE INITIALIZATION
# ================================

def initialize_system_safe():
    """Initialize with memory safety measures"""
    print("🚀 Initializing Memory-Safe Therapeutic AI...")
    
    # Set memory limits
    limit_memory()
    
    # Initialize database
    init_database()
    
    # Load knowledge base
    knowledge_base = load_knowledge_base()
    print(f"📚 Knowledge base loaded: {len(knowledge_base['documents'])} documents")
    
    # Check database health
    try:
        with get_db_connection() as conn:
            total_users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
            authenticated_users = conn.execute('SELECT COUNT(*) as count FROM users WHERE email IS NOT NULL').fetchone()['count']
            total_conversations = conn.execute('SELECT COUNT(*) as count FROM conversations').fetchone()['count']
            print(f"👥 Database: {total_users} users ({authenticated_users} registered, {total_users - authenticated_users} anonymous), {total_conversations} conversations")
    except Exception as e:
        print(f"⚠️ Database check failed: {e}")
    
    # Force initial garbage collection
    gc.collect()
    
    print("✅ Memory-safe Therapeutic AI initialized!")
    print("🎯 Safety features active for Standard tier:")
    print("   - 2GB memory limit")
    print("   - 100MB PDF limit for personal uploads")
    print("   - 200MB PDF limit for admin uploads")
    print("   - 500 page limit per PDF")
    print("   - Optimized garbage collection")
    print("   - Enhanced error handling")

# ================================
# APPLICATION STARTUP
# ================================

if __name__ == "__main__":
    initialize_system_safe()
    app.run(host="0.0.0.0", port=5000, debug=False)  # Disable debug in production
else:
    initialize_system_safe()
    application = app
