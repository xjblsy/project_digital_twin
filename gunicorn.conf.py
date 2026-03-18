# Gunicorn configuration file
bind = "0.0.0.0:8080"
workers = 2
threads = 2
worker_class = "gthread"
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
preload_app = True
accesslog = "-"
errorlog = "-"
loglevel = "info"
