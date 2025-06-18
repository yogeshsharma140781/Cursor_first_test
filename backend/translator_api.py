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
import base64
from PyPDF2 import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

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
    """Extract text from PDF using PyPDF2"""
    try:
        # Create a PDF reader from bytes
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PdfReader(pdf_file)
        
        # Extract text from all pages
        text_content = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():
                text_content.append(text.strip())
        
        # Join all text
        full_text = '\n\n'.join(text_content)
        
        if not full_text.strip():
            return "No readable text found in PDF"
        
        # Limit text length to avoid API limits
        if len(full_text) > 4000:
            full_text = full_text[:4000] + "..."
        
        return full_text
        
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def create_pdf_with_text(text: str, filename: str = "translated.pdf") -> bytes:
    """Create a PDF with translated text using reportlab"""
    try:
        # Create a BytesIO buffer
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        
        # Get styles
        styles = getSampleStyleSheet()
        normal_style = styles['Normal']
        
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        story = []
        
        for para_text in paragraphs:
            if para_text.strip():
                # Clean up text for reportlab
                clean_text = para_text.replace('\n', ' ').strip()
                if clean_text:
                    para = Paragraph(clean_text, normal_style)
                    story.append(para)
                    story.append(Spacer(1, 0.2 * inch))
        
        # Build the PDF
        doc.build(story)
        
        # Get the PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content
        
    except Exception as e:
        # Fallback: create simple text-based PDF
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        
        # Add text to PDF
        c.setFont("Helvetica", 12)
        y_position = 750
        
        lines = text.split('\n')
        for line in lines:
            if y_position < 50:  # Start new page if needed
                c.showPage()
                y_position = 750
            
            # Wrap long lines
            if len(line) > 80:
                words = line.split(' ')
                current_line = ""
                for word in words:
                    if len(current_line + word) < 80:
                        current_line += word + " "
                    else:
                        c.drawString(50, y_position, current_line.strip())
                        y_position -= 15
                        current_line = word + " "
                        if y_position < 50:
                            c.showPage()
                            y_position = 750
                if current_line:
                    c.drawString(50, y_position, current_line.strip())
                    y_position -= 15
            else:
                c.drawString(50, y_position, line)
                y_position -= 15
        
        c.save()
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content

@app.get("/")
async def root():
    return {
        "message": "Translation API is running",
        "version": "5.0",
        "status": "OK",
        "endpoints": ["/translate", "/translate-pdf", "/translate-pdf-debug"],
        "improvements": [
            "Proper PDF text extraction with PyPDF2",
            "Professional PDF generation with ReportLab", 
            "Enhanced error handling and validation",
            "File size limits and better cleanup"
        ]
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
    temp_file_path = None
    try:
        # Validate file
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read file content
        file_content = await file.read()
        
        # Validate file size
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Validate PDF header
        if not file_content.startswith(b'%PDF-'):
            raise HTTPException(status_code=400, detail="Invalid PDF file - missing PDF header")
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(file_content)
        
        # Check if text extraction failed
        if not extracted_text or len(extracted_text.strip()) < 3:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": "Could not extract readable text from PDF",
                    "filename": file.filename,
                    "file_size": len(file_content),
                    "extracted_text": extracted_text[:200] if extracted_text else "No text found",
                    "suggestion": "Please try with a text-based PDF (not scanned images)"
                }
            )
        
        # Check if extracted text contains error message
        if "Error extracting text" in extracted_text or "No readable text found" in extracted_text:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": extracted_text,
                    "filename": file.filename,
                    "file_size": len(file_content),
                    "suggestion": "Please try with a different PDF or check if the PDF contains selectable text"
                }
            )
        
        # Translate the extracted text
        translated_text = await translate_text_openai(extracted_text, source_lang, target_lang)
        
        if not translated_text or len(translated_text.strip()) < 3:
            raise HTTPException(status_code=500, detail="Translation returned empty result")
        
        # Create PDF with translated text
        translated_pdf_content = create_pdf_with_text(translated_text, file.filename)
        
        if not translated_pdf_content or len(translated_pdf_content) < 100:
            raise HTTPException(status_code=500, detail="Failed to generate translated PDF")
        
        # Create temporary file for response
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as tmp_file:
            tmp_file.write(translated_pdf_content)
            temp_file_path = tmp_file.name
        
        # Verify the file was created successfully
        if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
            raise HTTPException(status_code=500, detail="Failed to create temporary PDF file")
        
        # Return the translated PDF
        return FileResponse(
            temp_file_path,
            media_type='application/pdf',
            filename=f"translated_{file.filename}",
            background=lambda: safe_cleanup(temp_file_path)
        )
        
    except HTTPException:
        # Clean up temp file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            safe_cleanup(temp_file_path)
        raise
    except Exception as e:
        # Clean up temp file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            safe_cleanup(temp_file_path)
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "PDF translation failed due to server error",
                "filename": file.filename if file and file.filename else "unknown"
            }
        )

def safe_cleanup(file_path: str):
    """Safely remove temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception:
        pass  # Ignore cleanup errors

@app.post("/translate-pdf-debug")
async def translate_pdf_debug(
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("en")
):
    """Debug version that returns JSON with base64-encoded PDF"""
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
        
        # Create PDF with translated text
        translated_pdf_content = create_pdf_with_text(translated_text, file.filename)
        
        # Return JSON with base64-encoded PDF
        pdf_base64 = base64.b64encode(translated_pdf_content).decode('utf-8')
        
        return JSONResponse({
            "success": True,
            "message": "PDF translated successfully",
            "filename": f"translated_{file.filename}",
            "original_text": extracted_text,
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "pdf_base64": pdf_base64,
            "pdf_size": len(translated_pdf_content),
            "version": "4.3-debug"
        })
        
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