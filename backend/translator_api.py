import os
import tempfile
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import io

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

def extract_text_from_pdf_simple(pdf_content: bytes) -> str:
    """Simple PDF text extraction without heavy libraries."""
    try:
        # Convert bytes to string and look for text between stream objects
        pdf_text = pdf_content.decode('latin-1', errors='ignore')
        
        # Simple text extraction - look for text between BT and ET markers
        text_blocks = []
        lines = pdf_text.split('\n')
        
        in_text_block = False
        current_text = ""
        
        for line in lines:
            if 'BT' in line:  # Begin text
                in_text_block = True
                continue
            elif 'ET' in line:  # End text
                if in_text_block and current_text:
                    text_blocks.append(current_text.strip())
                in_text_block = False
                current_text = ""
                continue
            
            if in_text_block:
                # Look for text in parentheses or brackets
                if '(' in line and ')' in line:
                    start = line.find('(')
                    end = line.rfind(')')
                    if start < end:
                        text = line[start+1:end]
                        # Clean up the text
                        text = text.replace('\\n', '\n').replace('\\t', '\t')
                        current_text += text + " "
        
        # If no text blocks found, try a different approach
        if not text_blocks:
            # Look for any readable text in the PDF
            readable_text = ""
            for char in pdf_text:
                if char.isprintable() and char not in '<>[]{}()':
                    readable_text += char
                elif char in '\n\r\t ':
                    readable_text += char
            
            # Extract words that look like actual text
            words = readable_text.split()
            text_words = [word for word in words if len(word) > 1 and any(c.isalpha() for c in word)]
            
            if text_words:
                text_blocks = [' '.join(text_words)]
        
        return '\n\n'.join(text_blocks) if text_blocks else "No text could be extracted from this PDF."
        
    except Exception as e:
        return f"Error extracting text: {str(e)}"

async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using OpenAI API."""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server.")
    
    # Check cache first
    cache_key = get_cache_key(text, source_lang, target_lang)
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    
    try:
        prompt = f"Translate the following text from {source_lang} to {target_lang}. Output only the translation:\n\n{text}"
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2048
        }
        
        response = await client.post(url, headers=headers, json=data)
        result = response.json()
        translation = result["choices"][0]["message"]["content"].strip()
        
        # Cache the result
        if len(translation_cache) >= CACHE_SIZE:
            oldest_key = next(iter(translation_cache))
            del translation_cache[oldest_key]
        
        translation_cache[cache_key] = translation
        return translation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

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
    """PDF translation endpoint - extracts text and returns translated text file"""
    
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
    
    try:
        # Extract text from PDF
        extracted_text = extract_text_from_pdf_simple(file_content)
        
        if not extracted_text or extracted_text.strip() == "No text could be extracted from this PDF.":
            return {
                "success": True,
                "message": "PDF processed but no extractable text found",
                "filename": file.filename,
                "extracted_text": extracted_text,
                "translation": "No text to translate"
            }
        
        # Translate the extracted text
        translation = await translate_text(extracted_text, source_lang, target_lang)
        
        # Return translated text as downloadable file
        output_filename = f"translated_{file.filename.replace('.pdf', '.txt')}"
        
        return Response(
            content=f"Original PDF: {file.filename}\nSource Language: {source_lang}\nTarget Language: {target_lang}\n\n--- EXTRACTED TEXT ---\n{extracted_text}\n\n--- TRANSLATION ---\n{translation}",
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={output_filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Translation API is running", 
        "version": "4.1", 
        "status": "OK",
        "endpoints": ["/translate", "/translate-pdf"],
        "features": ["Text translation", "PDF text extraction and translation"]
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources."""
    await client.aclose() 