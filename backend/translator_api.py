import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import hashlib

load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

app = FastAPI()

# Create a persistent HTTP client for connection pooling
client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0),
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
)

# Simple in-memory cache for translations
translation_cache = {}
CACHE_SIZE = 1000

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, use ["https://translay.onrender.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

def get_cache_key(text: str, source_lang: str, target_lang: str) -> str:
    """Generate a cache key for the translation request."""
    content = f"{text}|{source_lang}|{target_lang}"
    return hashlib.md5(content.encode()).hexdigest()

@app.post("/translate")
async def translate(req: TranslationRequest):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="No text provided.")
    
    # Check cache first
    cache_key = get_cache_key(req.text, req.source_lang, req.target_lang)
    if cache_key in translation_cache:
        return {"translation": translation_cache[cache_key]}
    
    try:
        # Use simplified prompt for faster translation
        prompt = f"Translate the following text from {req.source_lang} to {req.target_lang}. Output only the translation:\n\n{req.text}"
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4o-mini",  # Much faster than gpt-4
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1024
        }
        
        response = await client.post(url, headers=headers, json=data)
        result = response.json()
        translation = result["choices"][0]["message"]["content"].strip()
        
        # Cache the result (with size limit)
        if len(translation_cache) >= CACHE_SIZE:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(translation_cache))
            del translation_cache[oldest_key]
        
        translation_cache[cache_key] = translation
        
        return {"translation": translation}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Warm up the connection to reduce cold start latency."""
    try:
        # Make a simple request to warm up connections
        await client.get("https://api.openai.com/v1/models", 
                        headers={"Authorization": f"Bearer {API_KEY}"})
    except:
        pass  # Ignore errors during warmup

@app.post("/translate-pdf")
async def translate_pdf(
    file: UploadFile = File(...),
    source_lang: str = "auto", 
    target_lang: str = "en"
):
    """PDF translation endpoint - currently returns test response"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Read file content
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    # Basic PDF validation
    if not file_content.startswith(b'%PDF-'):
        raise HTTPException(status_code=400, detail="Invalid PDF file format")
    
    # Return success response (placeholder for now)
    return {
        "success": True,
        "message": "PDF endpoint is working! (v3.2)",
        "filename": file.filename,
        "file_size": len(file_content),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "status": "processed"
    }

@app.get("/")
async def root():
    return {
        "message": "Translation API is running", 
        "version": "3.2", 
        "status": "OK",
        "endpoints": ["/translate", "/translate-pdf"]
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources."""
    await client.aclose() 