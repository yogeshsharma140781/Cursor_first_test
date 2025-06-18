from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create FastAPI app
app = FastAPI(title="Translation API", version="3.1", description="PDF and Text Translation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Translation API is running", 
        "version": "3.1", 
        "status": "OK",
        "endpoints": ["/translate", "/translate-pdf"]
    }

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Text translation endpoint
@app.post("/translate")
async def translate(req: TranslateRequest):
    """Translate text using OpenAI API"""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
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
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            translation = result["choices"][0]["message"]["content"].strip()
        return {"translation": translation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# PDF translation endpoint
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
        "message": "PDF endpoint is working! (v3.1)",
        "filename": file.filename,
        "file_size": len(file_content),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "status": "processed"
    }

# Make sure the app is properly exported
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 