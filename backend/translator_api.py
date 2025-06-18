import os
import tempfile
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
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
    return hashlib.md5(f"{text}:{source_lang}:{target_lang}".encode()).hexdigest()

def manage_cache_size():
    if len(translation_cache) > CACHE_SIZE:
        # Remove oldest 20% of entries
        keys_to_remove = list(translation_cache.keys())[:CACHE_SIZE // 5]
        for key in keys_to_remove:
            del translation_cache[key]

async def translate_text_openai(text: str, source_lang: str, target_lang: str) -> str:
    cache_key = get_cache_key(text, source_lang, target_lang)
    
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system", 
                        "content": f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. Return only the translation, no explanations."
                    },
                    {"role": "user", "content": text}
                ],
                "max_tokens": 2000,
                "temperature": 0.1
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"OpenAI API error: {response.text}")
        
        result = response.json()
        translated_text = result["choices"][0]["message"]["content"].strip()
        
        # Cache the result
        translation_cache[cache_key] = translated_text
        manage_cache_size()
        
        return translated_text
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

def extract_text_from_pdf(file_content: bytes) -> str:
    """Simple PDF text extraction - looks for readable text patterns"""
    try:
        # Convert to string and look for readable text
        text_content = file_content.decode('utf-8', errors='ignore')
        
        # Look for common PDF text patterns
        lines = text_content.split('\n')
        readable_lines = []
        
        for line in lines:
            # Skip binary/encoded lines
            if len(line) > 100 and any(ord(c) > 127 for c in line[:50]):
                continue
            # Skip lines with too many special characters
            if len([c for c in line if c.isalnum() or c.isspace()]) < len(line) * 0.3:
                continue
            # Keep lines with reasonable text
            if len(line.strip()) > 3 and any(c.isalpha() for c in line):
                readable_lines.append(line.strip())
        
        # If we found readable text, return it
        if readable_lines:
            return '\n'.join(readable_lines[:10])  # Limit to first 10 lines
        
        # Fallback: look for any text between common PDF markers
        import re
        text_matches = re.findall(r'\((.*?)\)', text_content)
        readable_text = []
        for match in text_matches:
            if len(match) > 3 and any(c.isalpha() for c in match):
                readable_text.append(match)
        
        if readable_text:
            return ' '.join(readable_text[:20])  # Limit to first 20 matches
            
        return "This appears to be a complex PDF with embedded content. Please try with a simpler text-based PDF."
        
    except Exception as e:
        return f"Could not extract text from PDF: {str(e)}"

def create_simple_pdf_with_text(text: str) -> bytes:
    """Create a simple PDF with the translated text"""
    try:
        # Create a simple PDF-like structure
        pdf_content = f"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
>>
endobj

4 0 obj
<<
/Length {len(text) + 100}
>>
stream
BT
/F1 12 Tf
50 750 Td
({text}) Tj
ET
endstream
endobj

5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000074 00000 n 
0000000120 00000 n 
0000000274 00000 n 
0000000373 00000 n 
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
456
%%EOF"""
        
        return pdf_content.encode('utf-8')
        
    except Exception as e:
        # Fallback: return text file as PDF
        return f"Translated Text:\n\n{text}".encode('utf-8')

@app.get("/")
async def root():
    return {
        "message": "Translation API is running",
        "version": "4.2",
        "status": "OK",
        "endpoints": ["/translate", "/translate-pdf"]
    }

@app.post("/translate")
async def translate(request: TranslationRequest):
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
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/translate-pdf")
async def translate_pdf(
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("en")
):
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read file content
        file_content = await file.read()
        
        # Validate PDF header
        if not file_content.startswith(b'%PDF-'):
            raise HTTPException(status_code=400, detail="Invalid PDF file")
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(file_content)
        
        if not extracted_text or len(extracted_text.strip()) < 3:
            return JSONResponse({
                "success": False,
                "message": "Could not extract readable text from PDF",
                "filename": file.filename,
                "file_size": len(file_content),
                "extracted_text": extracted_text[:200] if extracted_text else "No text found"
            })
        
        # Translate the extracted text
        translated_text = await translate_text_openai(extracted_text, source_lang, target_lang)
        
        # Create a simple PDF with translated text
        translated_pdf_content = create_simple_pdf_with_text(translated_text)
        
        # Create temporary file for response
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(translated_pdf_content)
            tmp_file_path = tmp_file.name
        
        # Return the translated PDF
        return FileResponse(
            tmp_file_path,
            media_type='application/pdf',
            filename=f"translated_{file.filename}",
            background=lambda: os.unlink(tmp_file_path)  # Clean up after sending
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": "PDF translation failed",
            "filename": file.filename if file else "unknown"
        })

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 