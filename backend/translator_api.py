import os
import tempfile
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import PyPDF2
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

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

def extract_text_from_pdf(pdf_content: bytes) -> list:
    """Extract text from PDF content."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text_blocks = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():
                # Split into paragraphs
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                for para in paragraphs:
                    text_blocks.append({
                        'text': para,
                        'page': page_num,
                        'type': 'text'
                    })
        
        return text_blocks
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

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
            "max_tokens": 1024
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

def create_translated_pdf(text_blocks: list, translations: list, output_path: str):
    """Create a new PDF with translated text."""
    try:
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter
        
        current_page = 0
        y_position = height - 50  # Start from top
        
        for i, (block, translation) in enumerate(zip(text_blocks, translations)):
            # Check if we need a new page
            if block['page'] > current_page:
                c.showPage()
                current_page = block['page']
                y_position = height - 50
            
            # Word wrap the translation
            words = translation.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if c.stringWidth(test_line, "Helvetica", 12) < width - 100:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Draw the text lines
            for line in lines:
                if y_position < 50:  # Need new page
                    c.showPage()
                    y_position = height - 50
                
                c.drawString(50, y_position, line)
                y_position -= 20
            
            y_position -= 10  # Extra space between blocks
        
        c.save()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create PDF: {str(e)}")

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
    """PDF translation endpoint - processes PDF and returns translated version"""
    
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
        text_blocks = extract_text_from_pdf(file_content)
        
        if not text_blocks:
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        # Translate all text blocks
        translations = []
        for block in text_blocks:
            translation = await translate_text(block['text'], source_lang, target_lang)
            translations.append(translation)
        
        # Create a persistent temporary file that won't be cleaned up immediately
        output_filename = f"translated_{file.filename}"
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, output_filename)
        
        # Create translated PDF
        create_translated_pdf(text_blocks, translations, output_path)
        
        # Return the file
        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type="application/pdf",
            background=None  # Don't delete immediately
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Translation API is running", 
        "version": "4.0", 
        "status": "OK",
        "endpoints": ["/translate", "/translate-pdf"],
        "features": ["Text translation", "PDF translation with file download"]
    }

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources."""
    await client.aclose() 