from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import chat, upload, documents

app = FastAPI(title="NusantaraLaw Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(documents.router, prefix="/api")

@app.get("/api/health")
def health_check():
    return {"status": "ok"}
