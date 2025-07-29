# Memory-optimized Gunicorn configuration
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', 5000)}"
backlog = 2048

# Worker processes - limit for memory efficiency
workers = 1  # Single worker to minimize memory usage
worker_class = "sync"
worker_connections = 50  # Limit concurrent connections
max_requests = 100  # Restart workers after 100 requests to prevent memory leaks
max_requests_jitter = 10
timeout = 60
keepalive = 2

# Memory management
preload_app = True  # Share memory between workers
worker_tmp_dir = "/dev/shm"  # Use memory for temporary files

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "therapeutic-ai"

# Graceful shutdowns
graceful_timeout = 30

# Memory limits and optimizations
def when_ready(server):
    print("🚀 Therapeutic AI server ready (memory optimized)")

def worker_int(worker):
    print(f"Worker {worker.pid} received INT or QUIT signal")

def pre_fork(server, worker):
    print(f"Worker spawned (pid: {worker.pid})")

def post_fork(server, worker):
    print(f"Worker {worker.pid} booted")
    # Force garbage collection on worker start
    import gc
    gc.collect()

def worker_abort(worker):
    print(f"Worker {worker.pid} received SIGABRT signal")

# Environment variables for memory optimization
raw_env = [
    'PYTHONOPTIMIZE=1',  # Enable Python optimizations
    'PYTHONDONTWRITEBYTECODE=1',  # Don't write .pyc files
]
