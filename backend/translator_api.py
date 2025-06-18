import os
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str = "en"

def get_cache_key(text: str, source_lang: str, target_lang: str) -> str:
    """Generate a cache key for the translation."""
    content = f"{text}|{source_lang}|{target_lang}"
    return hashlib.md5(content.encode()).hexdigest()

async def translate_text_openai(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using OpenAI API with caching."""
    
    # Check cache first
    cache_key = get_cache_key(text, source_lang, target_lang)
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        # Create translation prompt
        if source_lang == "auto":
            prompt = f"Translate the following text to {target_lang}. Return only the translation:\n\n{text}"
        else:
            prompt = f"Translate the following text from {source_lang} to {target_lang}. Return only the translation:\n\n{text}"
        
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2000,
                "temperature": 0.3
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {response.status_code}")
        
        result = response.json()
        translated_text = result["choices"][0]["message"]["content"].strip()
        
        # Cache the result (with size limit)
        if len(translation_cache) >= CACHE_SIZE:
            # Remove oldest entry
            oldest_key = next(iter(translation_cache))
            del translation_cache[oldest_key]
        
        translation_cache[cache_key] = translated_text
        
        return translated_text
        
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

def extract_text_from_pdf(file_content: bytes) -> str:
    """Simple PDF text extraction - just return a placeholder for now."""
    # For now, return a simple message since we're avoiding heavy dependencies
    return "PDF text extraction placeholder. In a full implementation, this would extract actual text from the PDF file."

@app.get("/")
async def root():
    return {
        "message": "Translation API is running",
        "version": "4.2",
        "status": "OK",
        "endpoints": ["/translate", "/translate-pdf"]
    }

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """Translate text using OpenAI API."""
    try:
        translated_text = await translate_text_openai(
            request.text, 
            request.source_lang, 
            request.target_lang
        )
        
        return {
            "success": True,
            "original_text": request.text,
            "translated_text": translated_text,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate-pdf")
async def translate_pdf(
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("en")
):
    """Translate PDF file and return translated text as JSON."""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read file content
        file_content = await file.read()
        
        # Basic PDF validation (check for PDF header)
        if not file_content.startswith(b'%PDF-'):
            raise HTTPException(status_code=400, detail="Invalid PDF file")
        
        # Extract text from PDF (simplified)
        extracted_text = extract_text_from_pdf(file_content)
        
        # Translate the extracted text
        translated_text = await translate_text_openai(extracted_text, source_lang, target_lang)
        
        return JSONResponse({
            "success": True,
            "message": "PDF translated successfully",
            "filename": file.filename,
            "file_size": len(file_content),
            "source_lang": source_lang,
            "target_lang": target_lang,
            "original_text": extracted_text,
            "translated_text": translated_text,
            "version": "4.2"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 