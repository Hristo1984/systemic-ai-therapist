# Gunicorn configuration for Render Standard tier
import multiprocessing
import os
import gc

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', 5000)}"
backlog = 2048

# Workers - optimized for Standard tier
workers = min(multiprocessing.cpu_count(), 3)
worker_class = "sync"
worker_connections = 50
max_requests = 100
max_requests_jitter = 10
timeout = 300
keepalive = 2

# Memory management
preload_app = True
worker_tmp_dir = "/dev/shm"

# Request limits
limit_request_line = 8192
limit_request_fields = 200
limit_request_field_size = 16384

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Process naming
proc_name = "therapeutic-ai-standard"

# Graceful shutdowns
graceful_timeout = 60

# Worker lifecycle callbacks
def when_ready(server):
    print("🚀 Standard Tier Therapeutic AI ready")

def worker_int(worker):
    print(f"Worker {worker.pid} shutting down")
    gc.collect()

def pre_fork(server, worker):
    print(f"Forking worker {worker.pid}")

def post_fork(server, worker):
    print(f"Worker {worker.pid} started")
    gc.collect()

def worker_abort(worker):
    print(f"Worker {worker.pid} aborted")
    gc.collect()

# Environment variables
raw_env = [
    'PYTHONOPTIMIZE=1',
    'PYTHONDONTWRITEBYTECODE=1',
    'PYTHONUNBUFFERED=1'
]
