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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import simpleSplit
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import re
from typing import List, Dict, Any, Tuple

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

class SimpleTextBlock:
    """Simple text block representation inspired by test_layoutparser_simple.py"""
    def __init__(self, text: str, block_type: str = "text", font_size: float = 12, is_bold: bool = False, is_italic: bool = False):
        self.text = text.strip()
        self.type = block_type
        self.size = font_size
        self.bold = is_bold
        self.italic = is_italic
        self.confidence = 1.0

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
    """Extract text from PDF using PyPDF2 - robust fallback approach"""
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

def extract_text_blocks_from_pdf(file_content: bytes) -> List[SimpleTextBlock]:
    """Extract text blocks from PDF with smart paragraph detection (inspired by test_layoutparser_simple.py)"""
    try:
        # Create a PDF reader from bytes
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PdfReader(pdf_file)
        
        all_blocks = []
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            if not text.strip():
                continue
            
            # Split into paragraphs using smart detection (inspired by test_layoutparser_simple.py)
            paragraphs = smart_paragraph_detection(text)
            
            for para_text in paragraphs:
                if para_text.strip():
                    # Classify block type (adapted from test_layoutparser_simple.py)
                    block_type = classify_block_type(para_text)
                    
                    # Estimate font properties based on content
                    font_size, is_bold, is_italic = estimate_font_properties(para_text, block_type)
                    
                    block = SimpleTextBlock(para_text, block_type, font_size, is_bold, is_italic)
                    all_blocks.append(block)
        
        return all_blocks
        
    except Exception as e:
        # Fallback to simple text extraction
        simple_text = extract_text_from_pdf(file_content)
        if simple_text and not simple_text.startswith("Error"):
            # Create a single text block
            block = SimpleTextBlock(simple_text, "text", 12, False, False)
            return [block]
        raise HTTPException(status_code=500, detail=f"Error extracting text blocks from PDF: {str(e)}")

def smart_paragraph_detection(text: str) -> List[str]:
    """Smart paragraph detection adapted from test_layoutparser_simple.py concepts"""
    # Split by double newlines first
    potential_paragraphs = text.split('\n\n')
    
    # Further split by single newlines if they seem to be paragraph breaks
    final_paragraphs = []
    
    for para in potential_paragraphs:
        lines = para.split('\n')
        current_paragraph = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if this line should start a new paragraph
            if (current_paragraph and 
                (line[0].isupper() and len(line) > 20) or  # Starts with capital and is long
                any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '•', '-', '*']) or  # List item
                len(line) < 50 and not line.endswith('.') and not line.endswith(',') and not line.endswith(':')):  # Possible title/header
                
                # Finish current paragraph
                if current_paragraph:
                    final_paragraphs.append('\n'.join(current_paragraph))
                    current_paragraph = []
            
            current_paragraph.append(line)
        
        # Add remaining paragraph
        if current_paragraph:
            final_paragraphs.append('\n'.join(current_paragraph))
    
    return final_paragraphs

def classify_block_type(text: str) -> str:
    """Classify block type adapted from test_layoutparser_simple.py"""
    text_lower = text.lower().strip()
    
    # Check if it's a title (short, no ending punctuation)
    if len(text) < 80 and not text.endswith('.') and not text.endswith('!') and not text.endswith('?'):
        return "title"
    
    # Check for table-like content (contains numbers, currency, dates)
    if re.search(r'\d+[.,]\d+|€|\$|USD|EUR|\d{2}[-/]\d{2}[-/]\d{4}', text):
        return "table"
    
    # Check for list items
    if re.match(r'^\s*[-•*]\s+', text) or re.match(r'^\s*\d+\.\s+', text):
        return "list"
    
    # Check for headers (all caps, short)
    if len(text.split()) <= 5 and text.isupper():
        return "header"
    
    # Default to text
    return "text"

def estimate_font_properties(text: str, block_type: str) -> Tuple[float, bool, bool]:
    """Estimate font properties based on content and block type (adapted from test_layoutparser_simple.py)"""
    # Default values
    font_size = 12
    is_bold = False
    is_italic = False
    
    # Adjust based on block type
    if block_type == "title":
        font_size = 16
        is_bold = True
    elif block_type == "header":
        font_size = 14
        is_bold = True
    elif block_type == "table":
        font_size = 10
    elif block_type == "list":
        font_size = 11
    
    # Check for emphasis indicators
    if text.isupper() and len(text) < 100:
        is_bold = True
    
    return font_size, is_bold, is_italic

def postprocess_english(text: str, original: str) -> str:
    """Improve English punctuation and capitalization (adapted from test_layoutparser_simple.py)"""
    import re
    
    # Apply translation fixes (simplified version from test_layoutparser_simple.py)
    text = apply_translation_fixes(text, original)
    
    # Capitalize first letter of each sentence
    def capitalize_sentences(s):
        s = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), s)
        s = s[:1].upper() + s[1:] if s else s
        return s
    
    # Only add punctuation if it's a complete sentence and original had punctuation
    s = text.strip()
    if (s and s[-1] not in '.!?' and 
        original.strip().endswith('.') and 
        len(s.split()) > 3):  # Only for longer phrases
        s += '.'
        
    s = capitalize_sentences(s)
    return s

def apply_translation_fixes(text: str, original: str) -> str:
    """Apply translation fixes adapted from test_layoutparser_simple.py"""
    # Dictionary of common Dutch to English fixes
    fixes = {
        'INLOGGEGEVENS': 'LOGIN DETAILS',
        'AFKOOP EIGEN RISICO': 'DEDUCTIBLE BUYOUT',
        'Artikelnummer': 'Article Number',
        'Aantal': 'Quantity',
        'Stuksprijs': 'Unit Price',
        'Totaal': 'Total',
        'BTW-grondslag': 'VAT Base',
        'BTW-bedrag': 'VAT Amount',
        'Factuurnummer': 'Invoice Number',
        'Factuurdatum': 'Invoice Date',
        'Bestelnummer': 'Order Number',
        'Besteldatum': 'Order Date',
    }
    
    # Apply fixes
    for dutch, english in fixes.items():
        if dutch in original:
            text = re.sub(re.escape(dutch), english, text, flags=re.IGNORECASE)
    
    # Fix currency formatting
    if '€' in original:
        text = re.sub(r'EUR\s*(\d)', r'€\1', text)
    
    return text

def create_pdf_with_text(text: str, filename: str = "translated.pdf") -> bytes:
    """Create a well-formatted PDF with translated text using reportlab - ROBUST VERSION"""
    try:
        # Create a BytesIO buffer
        buffer = io.BytesIO()
        
        # Create the PDF document with better margins
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            rightMargin=72,  # 1 inch margins
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles and customize them
        styles = getSampleStyleSheet()
        
        # Create custom styles for better formatting
        title_style = styles['Title']
        title_style.spaceAfter = 20
        
        normal_style = styles['Normal']
        normal_style.fontSize = 11
        normal_style.leading = 14  # Line spacing
        normal_style.spaceAfter = 12
        normal_style.alignment = 0  # Left align
        
        # Create a custom style for better paragraph spacing
        custom_para_style = ParagraphStyle(
            'CustomNormal',
            parent=normal_style,
            fontSize=11,
            leading=16,
            spaceAfter=8,
            spaceBefore=4,
            alignment=0,
            leftIndent=0,
            rightIndent=0
        )
        
        story = []
        
        # Add title if the text seems to have one (first line is shorter and looks like a title)
        lines = text.strip().split('\n')
        first_line = lines[0].strip() if lines else ""
        
        # Check if first line looks like a title (short, no ending punctuation)
        if (len(first_line) < 80 and 
            len(lines) > 1 and 
            not first_line.endswith('.') and 
            not first_line.endswith('!') and 
            not first_line.endswith('?')):
            
            # Use first line as title
            title = Paragraph(first_line, title_style)
            story.append(title)
            story.append(Spacer(1, 0.3 * inch))
            
            # Process remaining lines
            remaining_text = '\n'.join(lines[1:])
        else:
            # No title, process all text
            remaining_text = text
        
        # Split text into paragraphs (double line breaks or single line breaks)
        paragraphs = []
        current_paragraph = []
        
        for line in remaining_text.split('\n'):
            line = line.strip()
            
            if not line:  # Empty line - end current paragraph
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(line)
        
        # Add final paragraph if exists
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Add paragraphs to story
        for para_text in paragraphs:
            if para_text.strip():
                # Handle special formatting
                clean_text = para_text.strip()
                
                # Check if it's a list item or bullet point
                if (clean_text.startswith('-') or 
                    clean_text.startswith('•') or 
                    clean_text.startswith('*') or
                    any(clean_text.startswith(f'{i}.') for i in range(1, 10))):
                    
                    # Format as list item with indentation
                    list_style = ParagraphStyle(
                        'ListItem',
                        parent=custom_para_style,
                        leftIndent=20,
                        bulletIndent=10,
                        spaceAfter=6
                    )
                    para = Paragraph(clean_text, list_style)
                else:
                    # Format as regular paragraph
                    para = Paragraph(clean_text, custom_para_style)
                
                story.append(para)
                story.append(Spacer(1, 0.1 * inch))
        
        # Build the PDF
        doc.build(story)
        
        # Get the PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content
        
    except Exception as e:
        # Enhanced fallback: create better formatted simple PDF
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Set up better formatting
        c.setFont("Helvetica", 11)
        margin = 72  # 1 inch margin
        y_position = height - margin
        line_height = 16
        
        # Split text into lines and handle wrapping
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:  # Empty line - add some space
                y_position -= line_height * 0.5
                continue
            
            # Check if we need a new page
            if y_position < margin + line_height:
                c.showPage()
                c.setFont("Helvetica", 11)
                y_position = height - margin
            
            # Handle long lines by wrapping them
            max_width = width - (2 * margin)
            if len(line) > 80:  # Approximate character limit
                words = line.split(' ')
                current_line = ""
                
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    
                    # Rough character-based wrapping (better than nothing)
                    if len(test_line) <= 80:
                        current_line = test_line
                    else:
                        if current_line:
                            c.drawString(margin, y_position, current_line)
                            y_position -= line_height
                            
                            # Check for new page
                            if y_position < margin + line_height:
                                c.showPage()
                                c.setFont("Helvetica", 11)
                                y_position = height - margin
                        
                        current_line = word
                
                # Draw the last line
                if current_line:
                    c.drawString(margin, y_position, current_line)
                    y_position -= line_height
            else:
                # Short line, draw directly
                c.drawString(margin, y_position, line)
                y_position -= line_height
        
        c.save()
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content

def create_pdf_with_advanced_formatting(blocks: List[SimpleTextBlock], filename: str = "translated.pdf") -> bytes:
    """Create a well-formatted PDF with advanced layout (adapted from test_layoutparser_simple.py concepts)"""
    try:
        # Create a BytesIO buffer
        buffer = io.BytesIO()
        
        # Create the PDF document with better margins
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            rightMargin=72,  # 1 inch margins
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles and customize them
        styles = getSampleStyleSheet()
        
        # Create custom styles for different block types
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=16,
            spaceAfter=20,
            spaceBefore=10,
            alignment=1,  # Center align
            textColor='black'
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading1'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=8,
            textColor='black'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            spaceAfter=8,
            spaceBefore=4,
            alignment=0,
            textColor='black'
        )
        
        table_style = ParagraphStyle(
            'CustomTable',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            spaceAfter=6,
            spaceBefore=4,
            leftIndent=10,
            textColor='black'
        )
        
        list_style = ParagraphStyle(
            'CustomList',
            parent=styles['Normal'],
            fontSize=11,
            leading=15,
            spaceAfter=4,
            spaceBefore=2,
            leftIndent=20,
            bulletIndent=10,
            textColor='black'
        )
        
        story = []
        
        # Process each block with appropriate styling
        for block in blocks:
            clean_text = block.text.strip()
            
            if not clean_text:
                continue
            
            # Select style based on block type
            if block.type == "title":
                para = Paragraph(clean_text, title_style)
                story.append(para)
                story.append(Spacer(1, 0.2 * inch))
            elif block.type == "header":
                para = Paragraph(clean_text, header_style)
                story.append(para)
                story.append(Spacer(1, 0.1 * inch))
            elif block.type == "table":
                para = Paragraph(clean_text, table_style)
                story.append(para)
                story.append(Spacer(1, 0.05 * inch))
            elif block.type == "list":
                para = Paragraph(clean_text, list_style)
                story.append(para)
                story.append(Spacer(1, 0.05 * inch))
            else:  # Regular text
                para = Paragraph(clean_text, normal_style)
                story.append(para)
                story.append(Spacer(1, 0.1 * inch))
        
        # Build the PDF
        doc.build(story)
        
        # Get the PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content
        
    except Exception as e:
        # Fallback to simple text approach
        combined_text = "\n\n".join([block.text for block in blocks])
        return create_pdf_with_text(combined_text, filename)

async def openai_translate_batch(texts: List[str], target_lang: str = 'en') -> List[str]:
    """Translate multiple texts using OpenAI API (adapted from test_layoutparser_simple.py)"""
    all_translations = []
    for text in texts:
        try:
            translated = await translate_text_openai(text, "auto", target_lang)
            all_translations.append(translated)
        except Exception as e:
            print(f"Translation error: {e}")
            all_translations.append(text)  # Fallback to original
    return all_translations

@app.get("/")
async def root():
    return {
        "message": "Translation API is running",
        "version": "6.2",
        "status": "OK",
        "endpoints": ["/translate", "/translate-pdf", "/translate-pdf-debug"],
        "improvements": [
            "Hybrid approach with robust fallbacks",
            "Smart paragraph detection (inspired by test_layoutparser_simple.py)", 
            "Block classification and font estimation",
            "Advanced formatting with reliable PDF generation",
            "Multiple fallback strategies for maximum reliability",
            "Enhanced error handling and recovery"
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
        
        # Try advanced approach first, with fallback to simple approach
        try:
            # Extract blocks using smart layout parser (inspired by test_layoutparser_simple.py)
            text_blocks = extract_text_blocks_from_pdf(file_content)
            
            if text_blocks and len(text_blocks) > 1:
                # Advanced approach worked - translate blocks
                texts = [block.text for block in text_blocks]
                translated_texts = await openai_translate_batch(texts, target_lang)
                
                if translated_texts and len(translated_texts) == len(texts):
                    # Apply post-processing to translations
                    processed_blocks = []
                    for i, block in enumerate(text_blocks):
                        translated_text = translated_texts[i] if i < len(translated_texts) else block.text
                        processed_text = postprocess_english(translated_text, block.text)
                        
                        # Create new block with processed text
                        processed_block = SimpleTextBlock(processed_text, block.type, block.size, block.bold, block.italic)
                        processed_blocks.append(processed_block)
                    
                    # Create PDF with advanced formatting
                    translated_pdf_content = create_pdf_with_advanced_formatting(processed_blocks, file.filename)
                else:
                    raise Exception("Translation batch failed")
            else:
                raise Exception("Advanced block extraction failed")
                
        except Exception as e:
            print(f"Advanced approach failed: {e}, falling back to simple approach")
            # Fallback to simple approach
            text_content = extract_text_from_pdf(file_content)
            
            if text_content.startswith("Error") or text_content.startswith("No readable"):
                return JSONResponse(
                    status_code=422,
                    content={
                        "success": False,
                        "message": "Could not extract readable text from PDF",
                        "filename": file.filename,
                        "file_size": len(file_content),
                        "suggestion": "Please try with a text-based PDF (not scanned images)"
                    }
                )
            
            # Translate the text
            translated_text = await translate_text_openai(text_content, source_lang, target_lang)
            
            # Create PDF with simple formatting
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
        
        # Try advanced approach first
        approach_used = "advanced"
        try:
            # Extract blocks using smart layout parser
            text_blocks = extract_text_blocks_from_pdf(file_content)
            
            if text_blocks and len(text_blocks) > 1:
                # Get sample text from blocks
                sample_text = "\n\n".join([f"[{block.type.upper()}] {block.text[:100]}..." if len(block.text) > 100 else f"[{block.type.upper()}] {block.text}" 
                                           for block in text_blocks[:3]])
                
                # Translate the extracted blocks
                texts = [block.text for block in text_blocks]
                translated_texts = await openai_translate_batch(texts, target_lang)
                
                # Apply post-processing to translations
                processed_blocks = []
                for i, block in enumerate(text_blocks):
                    translated_text = translated_texts[i] if i < len(translated_texts) else block.text
                    processed_text = postprocess_english(translated_text, block.text)
                    
                    processed_block = SimpleTextBlock(processed_text, block.type, block.size, block.bold, block.italic)
                    processed_blocks.append(processed_block)
                
                # Create PDF with advanced formatting
                translated_pdf_content = create_pdf_with_advanced_formatting(processed_blocks, file.filename)
                
                translated_sample = "\n\n".join([f"[{block.type.upper()}] {trans[:100]}..." if len(trans) > 100 else f"[{block.type.upper()}] {trans}" 
                                               for trans in translated_texts[:3]])
                
                block_types = [block.type for block in text_blocks]
                blocks_processed = len(text_blocks)
            else:
                raise Exception("Advanced block extraction failed")
                
        except Exception as e:
            print(f"Advanced approach failed: {e}, using simple approach")
            approach_used = "simple"
            
            # Fallback to simple approach
            text_content = extract_text_from_pdf(file_content)
            
            if text_content.startswith("Error") or text_content.startswith("No readable"):
                return JSONResponse({
                    "success": False,
                    "message": "Could not extract readable text from PDF",
                    "filename": file.filename,
                    "file_size": len(file_content),
                    "extracted_blocks": 0
                })
            
            sample_text = text_content[:300] + "..." if len(text_content) > 300 else text_content
            
            # Translate the text
            translated_text = await translate_text_openai(text_content, source_lang, target_lang)
            translated_sample = translated_text[:300] + "..." if len(translated_text) > 300 else translated_text
            
            # Create PDF with simple formatting
            translated_pdf_content = create_pdf_with_text(translated_text, file.filename)
            
            block_types = ["text"]
            blocks_processed = 1
        
        # Return JSON with base64-encoded PDF
        pdf_base64 = base64.b64encode(translated_pdf_content).decode('utf-8')
        
        return JSONResponse({
            "success": True,
            "message": f"PDF translated successfully using {approach_used} approach (inspired by test_layoutparser_simple.py)",
            "filename": f"translated_{file.filename}",
            "original_text": sample_text,
            "translated_text": translated_sample,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "pdf_base64": pdf_base64,
            "pdf_size": len(translated_pdf_content),
            "blocks_processed": blocks_processed,
            "block_types": block_types,
            "approach_used": approach_used,
            "version": "6.2-debug-hybrid"
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