# ================================
# CRITICAL MEMORY FIXES - Add these to your main.py
# ================================

# 1. ADD THIS AT THE TOP after imports
import resource
import signal
import sys

# Memory limit enforcement (512MB limit)
def limit_memory():
    try:
        # Set memory limit to 512MB (Render free tier safe limit)
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        print("✅ Memory limit set to 512MB")
    except Exception as e:
        print(f"⚠️ Could not set memory limit: {e}")

# 2. REPLACE your extract_text_from_pdf_efficient function with this MEMORY-SAFE version:
def extract_text_from_pdf_efficient(file_path, max_size_mb=50):  # REDUCED from 100MB
    """Memory-efficient PDF extraction with aggressive limits"""
    try:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"Processing PDF: {file_path} ({file_size:.1f}MB)")
        
        # STRICTER size limits for stability
        if file_size > max_size_mb:
            return f"PDF too large ({file_size:.1f}MB). Maximum size: {max_size_mb}MB for stability"
        
        text_content = ""
        page_count = 0
        
        # Force garbage collection before starting
        gc.collect()
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"PDF has {total_pages} pages")
                
                # LIMIT pages for memory safety
                max_pages = min(total_pages, 200)  # Max 200 pages
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
                        
                        # AGGRESSIVE memory management
                        if page_num % 10 == 0:  # Every 10 pages instead of 25
                            gc.collect()
                            
                        # Progress indication
                        if page_num % 25 == 0 and page_num > 0:
                            print(f"Processed {page_num}/{max_pages} pages...")
                            
                        # Memory safety check
                        if len(text_content) > 5 * 1024 * 1024:  # 5MB text limit
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

# 3. ADD this MEMORY-SAFE initialization function:
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
    print("🎯 Safety features active:")
    print("   - 512MB memory limit")
    print("   - 50MB PDF limit")
    print("   - 200 page limit per PDF")
    print("   - Aggressive garbage collection")
    print("   - Enhanced error handling")

# 4. REPLACE your main initialization code at the bottom with:
if __name__ == "__main__":
    initialize_system_safe()  # Use safe version
    app.run(host="0.0.0.0", port=5000, debug=False)  # Disable debug in production
else:
    initialize_system_safe()  # Use safe version
    application = app

# 5. ADD these Flask configuration settings (add after app = Flask(__name__)):
# Memory and upload safety settings
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max request
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
app.config['JSON_AS_ASCII'] = False

# 6. ADD this error handler for memory issues:
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

# 7. UPDATE your gunicorn.conf.py file with this COMPLETE configuration:
"""
# MEMORY-SAFE Gunicorn configuration - Replace your entire gunicorn.conf.py with this:
import multiprocessing
import os
import gc

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', 5000)}"
backlog = 512

# CRITICAL: Single worker for memory safety
workers = 1
worker_class = "sync"
worker_connections = 5  # Very low to prevent memory issues
max_requests = 25  # Restart worker frequently
max_requests_jitter = 5
timeout = 180  # Longer timeout for large files
keepalive = 2

# Memory management - CRITICAL
preload_app = True
worker_tmp_dir = "/dev/shm"

# Request limits - CRITICAL
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Process naming
proc_name = "therapeutic-ai-safe"

# Graceful shutdowns
graceful_timeout = 30

# Memory optimization callbacks
def when_ready(server):
    print("🚀 Memory-optimized Therapeutic AI ready")

def worker_int(worker):
    print(f"Worker {worker.pid} shutting down - cleaning memory")
    gc.collect()

def pre_fork(server, worker):
    print(f"Forking worker {worker.pid}")

def post_fork(server, worker):
    print(f"Worker {worker.pid} started - memory optimized")
    gc.collect()

def worker_abort(worker):
    print(f"Worker {worker.pid} aborted - force cleanup")
    gc.collect()

# Environment variables
raw_env = [
    'PYTHONOPTIMIZE=1',
    'PYTHONDONTWRITEBYTECODE=1',
    'PYTHONUNBUFFERED=1',
]
"""
