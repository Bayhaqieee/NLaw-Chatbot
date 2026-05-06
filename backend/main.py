from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import chat, upload, documents, evaluation

import sys
import os

app = FastAPI(title="NusantaraLaw Chatbot API")

# ── LoggerWriter: tee stdout to a file for /api/logs ─────────────────
# NOTE: We do NOT redirect sys.stderr — Uvicorn needs stderr intact
# for HTTP keepalive and connection management.
class LoggerWriter:
    def __init__(self, filename="app.log"):
        self._terminal = sys.__stdout__      # always the real stdout
        self._log = open(filename, "a", encoding="utf-8", buffering=1)

    def write(self, message):
        self._terminal.write(message)
        try:
            self._log.write(message)
        except Exception:
            pass

    def flush(self):
        self._terminal.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def fileno(self):
        return self._terminal.fileno()


sys.stdout = LoggerWriter("app.log")
# stderr intentionally NOT redirected — keeps Uvicorn HTTP stack healthy

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────
app.include_router(chat.router,       prefix="/api")
app.include_router(upload.router,     prefix="/api")
app.include_router(documents.router,  prefix="/api")
app.include_router(evaluation.router)

# ── Health check ──────────────────────────────────────────────────────
@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# ── Log viewer ────────────────────────────────────────────────────────
@app.get("/api/logs")
def get_logs(lines: int = 150):
    log_path = "app.log"
    if not os.path.exists(log_path):
        return {"logs": "Log file not found."}
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()
    return {"logs": "".join(all_lines[-lines:])}
