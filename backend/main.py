from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.get("/")
async def root():
    return {"message": "Translation API is running", "version": "2.1", "status": "OK"}

@app.post("/translate")
async def translate(req: TranslateRequest):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": f"Translate from {req.source_lang} to {req.target_lang}: {req.text}"}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        result = response.json()
        translation = result["choices"][0]["message"]["content"].strip()
    
    return {"translation": translation}

@app.post("/translate-pdf")
async def translate_pdf(
    file: UploadFile = File(...),
    source_lang: str = "auto", 
    target_lang: str = "en"
):
    """Test PDF endpoint"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_content = await file.read()
    
    if not file_content.startswith(b'%PDF-'):
        raise HTTPException(status_code=400, detail="Invalid PDF file")
    
    return {
        "success": True,
        "message": "PDF endpoint working!",
        "filename": file.filename,
        "size": len(file_content),
        "source_lang": source_lang,
        "target_lang": target_lang
    } 