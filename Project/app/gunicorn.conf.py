# Gunicorn configuration for Render free tier (512MB RAM)
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
backlog = 2048

# Worker processes
workers = 1  # Minimize memory usage on free tier
worker_class = 'sync'
worker_connections = 100
timeout = 120
keepalive = 5

# Memory optimization
max_requests = 100  # Restart workers after 100 requests to prevent memory leaks
max_requests_jitter = 10
worker_tmp_dir = '/dev/shm'  # Use RAM disk for worker temp files

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

