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
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple
import re
import numpy as np
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import simpleSplit
from reportlab.platypus import Paragraph, Frame, SimpleDocTemplate, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from collections import defaultdict
from fuzzywuzzy import fuzz

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
    """Simple text block representation"""
    def __init__(self, text: str, bbox: Tuple[float, float, float, float], block_type: str = "text", font: str = None, size: float = None, bold: bool = False, italic: bool = False):
        self.text = text.strip()
        self.bbox = bbox  # (x0, y0, x1, y1)
        self.type = block_type
        self.confidence = 1.0
        self.structured_data = None  # For storing table data
        self.font = font
        self.size = size
        self.bold = bold
        self.italic = italic
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "text": self.text,
            "bbox": self.bbox,
            "type": self.type,
            "confidence": self.confidence,
            "font": self.font,
            "size": self.size,
            "bold": self.bold,
            "italic": self.italic
        }
        if self.structured_data:
            result["structured_data"] = self.structured_data
        return result

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
    """Extract text from PDF using PyPDF2 - keeping for simple text extraction"""
    try:
        # Create a PDF reader from bytes
        pdf_file = io.BytesIO(file_content)
        from PyPDF2 import PdfReader
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

def extract_blocks_from_pdf(pdf_content: bytes) -> List[Tuple[SimpleTextBlock, int]]:
    """Extract paragraph blocks from PDF using PyMuPDF with page numbers, storing font/style info."""
    # Create temporary file for PyMuPDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_file.write(pdf_content)
        tmp_file_path = tmp_file.name
    
    try:
        doc = fitz.open(tmp_file_path)
        blocks = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_dict = page.get_text("dict")
            
            if len(text_dict["blocks"]) == 0:
                continue
                
            line_items = []
            for block_idx, block in enumerate(text_dict["blocks"]):
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = "\n".join(span["text"] for span in line["spans"]).strip()
                        if not line_text:
                            continue
                        line_bbox = line["bbox"]
                        font = line["spans"][0].get("font", "")
                        size = line["spans"][0].get("size", 0)
                        flags = line["spans"][0].get("flags", 0)
                        bold = bool(flags & 1) or ("bold" in font.lower())
                        italic = bool(flags & 2) or ("italic" in font.lower() or "oblique" in font.lower())
                        x0 = line_bbox[0]
                        y0 = line_bbox[1]
                        x1 = line_bbox[2]
                        y1 = line_bbox[3]
                        line_items.append({
                            "text": line_text,
                            "bbox": line_bbox,
                            "font": font,
                            "size": size,
                            "bold": bold,
                            "italic": italic,
                            "x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1,
                            "block_idx": block_idx
                        })
            
            line_items.sort(key=lambda l: (l["y0"], l["x0"]))
            paragraphs = []
            current_para = []
            
            for i, line in enumerate(line_items):
                if not current_para:
                    current_para.append(line)
                    continue
                prev = current_para[-1]
                vertical_gap = line["y0"] - prev["y1"]
                left_diff = abs(line["x0"] - prev["x0"])
                same_font = (line["font"] == prev["font"]) and (abs(line["size"] - prev["size"]) < 0.5)
                
                if vertical_gap > prev["size"] * 1.0 or left_diff > 10 or not same_font:
                    if current_para:
                        para_text = "\n".join(l["text"] for l in current_para)
                        para_bbox = [
                            min(l["x0"] for l in current_para),
                            min(l["y0"] for l in current_para),
                            max(l["x1"] for l in current_para),
                            max(l["y1"] for l in current_para)
                        ]
                        block_type = classify_block_type(para_text, para_bbox, page.rect)
                        # Dominant font/size/style
                        fonts = [l["font"] for l in current_para]
                        sizes = [l["size"] for l in current_para]
                        bolds = [l["bold"] for l in current_para]
                        italics = [l["italic"] for l in current_para]
                        dominant_font = max(set(fonts), key=fonts.count)
                        dominant_size = max(set(sizes), key=sizes.count)
                        is_bold = any(bolds)
                        is_italic = any(italics)
                        blocks.append((SimpleTextBlock(para_text, para_bbox, block_type, dominant_font, dominant_size, is_bold, is_italic), page_num))
                    current_para = [line]
                else:
                    current_para.append(line)
            
            if current_para:
                para_text = "\n".join(l["text"] for l in current_para)
                para_bbox = [
                    min(l["x0"] for l in current_para),
                    min(l["y0"] for l in current_para),
                    max(l["x1"] for l in current_para),
                    max(l["y1"] for l in current_para)
                ]
                block_type = classify_block_type(para_text, para_bbox, page.rect)
                fonts = [l["font"] for l in current_para]
                sizes = [l["size"] for l in current_para]
                bolds = [l["bold"] for l in current_para]
                italics = [l["italic"] for l in current_para]
                dominant_font = max(set(fonts), key=fonts.count)
                dominant_size = max(set(sizes), key=sizes.count)
                is_bold = any(bolds)
                is_italic = any(italics)
                blocks.append((SimpleTextBlock(para_text, para_bbox, block_type, dominant_font, dominant_size, is_bold, is_italic), page_num))
        
        doc.close()
        return blocks
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def classify_block_type(text: str, bbox: Tuple[float, float, float, float], page_rect) -> str:
    """Simple rule-based block type classification"""
    text_lower = text.lower().strip()
    
    # Check if it's a title (large font, top of page, short text)
    if len(text) < 50 and bbox[1] < page_rect.height * 0.2:
        return "title"
    
    # Check if it's a header/footer (very top or bottom)
    if bbox[1] < page_rect.height * 0.1 or bbox[3] > page_rect.height * 0.9:
        return "header" if bbox[1] < page_rect.height * 0.1 else "footer"
    
    # Check for table-like content (contains numbers, currency, dates)
    if re.search(r'\d+[.,]\d+|€|\$|USD|EUR|\d{2}[-/]\d{2}[-/]\d{4}', text):
        return "table"
    
    # Check for list items
    if re.match(r'^\s*[-•*]\s+', text) or re.match(r'^\s*\d+\.\s+', text):
        return "list"
    
    # Default to text
    return "text"

def create_pdf_with_text(text: str, filename: str = "translated.pdf") -> bytes:
    """Create a well-formatted PDF with translated text using reportlab"""
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

def create_translated_pdf_with_layout(pdf_content: bytes, translated_blocks: List[Tuple[SimpleTextBlock, int, str]], filename: str = "translated.pdf") -> bytes:
    """Create a well-formatted PDF using the advanced layout parser approach"""
    
    # Create temporary file for original PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_file.write(pdf_content)
        original_pdf_path = tmp_file.name
    
    try:
        # Use the advanced PDF creation from the layout parser
        doc = fitz.open(original_pdf_path)
        
        # Group blocks by page
        pages_blocks = {}
        for block, page_num, translated_text in translated_blocks:
            if page_num not in pages_blocks:
                pages_blocks[page_num] = []
            pages_blocks[page_num].append((block, translated_text))

        # Build a mapping of original font sizes to common output font sizes
        font_size_map = {}
        for block, _, _ in translated_blocks:
            if block.size is not None:
                orig_size = round(block.size, 1)
                if orig_size not in font_size_map:
                    font_size_map[orig_size] = max(5, int(block.size))

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            if page_num in pages_blocks:
                # Redact original text
                for block, translated_text in pages_blocks[page_num]:
                    x0, y0, x1, y1 = block.bbox
                    padding = 3
                    redact_rect = fitz.Rect(x0 - padding, y0 - padding, x1 + padding, y1 + padding)
                    page.add_redact_annot(redact_rect, (1, 1, 1))
                page.apply_redactions()
                
                # Insert translated text with proper formatting
                for block, translated_text in pages_blocks[page_num]:
                    x0, y0, x1, y1 = block.bbox
                    
                    # Use proper font handling
                    font_name = "helv"  # Fallback to built-in font
                    
                    # Process text with proper formatting
                    processed_text = postprocess_english(translated_text, block.text)
                    max_width = x1 - x0 - 6
                    max_height = y1 - y0 - 6
                    
                    # Use consistent font sizing
                    orig_size = round(block.size, 1) if block.size is not None else None
                    font_size = font_size_map.get(orig_size, 12)
                    min_font_size = 5
                    if font_size < min_font_size:
                        font_size = min_font_size
                    
                    # Wrap text properly
                    wrapped_lines = wrap_text_to_width(processed_text, font_size, font_name, max_width, page)
                    line_height = font_size * 1.15
                    start_y = y0 + font_size + 2
                    current_y = start_y
                    
                    for line in wrapped_lines:
                        if current_y < y1 + 3:
                            try:
                                clean_line = clean_text_for_pdf(str(line))
                                page.insert_text((x0 + 3, current_y), clean_line, fontsize=font_size, color=(0, 0, 0), fontname=font_name)
                            except Exception as e:
                                print(f"Error inserting text: {line} – {e}")
                        current_y += line_height
        
        # Save to buffer
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as output_tmp:
            doc.save(output_tmp.name)
            doc.close()
            
            # Read the result
            with open(output_tmp.name, 'rb') as f:
                pdf_content = f.read()
            
            # Clean up
            os.unlink(output_tmp.name)
            
        return pdf_content
        
    finally:
        # Clean up original temp file
        if os.path.exists(original_pdf_path):
            os.unlink(original_pdf_path)

def wrap_text_to_width(text, font_size, font_name, max_width, page):
    """Wrap text to fit within specified width"""
    lines = []
    for para in text.split('\n'):
        words = para.split()
        if not words:
            lines.append('')
            continue
        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + ' ' + word
            # Use approximate character-based width calculation
            char_width = font_size * 0.6  # Approximate character width
            if len(test_line) * char_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
    return lines

def clean_text_for_pdf(text: str) -> str:
    """Clean text for PDF insertion to prevent encoding issues"""
    # Remove or replace problematic characters
    text = text.replace('\u2019', "'")  # Replace smart quotes
    text = text.replace('\u201c', '"')  # Replace smart quotes
    text = text.replace('\u201d', '"')  # Replace smart quotes
    text = text.replace('\u2013', '-')  # Replace en dash
    text = text.replace('\u2014', '--') # Replace em dash
    text = text.replace('\u00a0', ' ')  # Replace non-breaking space
    
    # Keep only printable ASCII and common extended characters
    cleaned = ''
    for char in text:
        if ord(char) < 127 or char in 'áàäâéèëêíìïîóòöôúùüûñçÁÀÄÂÉÈËÊÍÌÏÎÓÒÖÔÚÙÜÛÑÇ€':
            cleaned += char
        else:
            cleaned += ' '  # Replace with space
    
    return cleaned

def postprocess_english(text: str, original: str) -> str:
    """Improve English punctuation and capitalization"""
    import re
    
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

async def openai_translate_batch(texts, target_lang='en'):
    """Translate multiple texts using OpenAI API"""
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
        "version": "6.0",
        "status": "OK",
        "endpoints": ["/translate", "/translate-pdf", "/translate-pdf-debug"],
        "improvements": [
            "Advanced layout parser integration",
            "Proper paragraph and block detection", 
            "Font style preservation",
            "Enhanced formatting and spacing",
            "Professional PDF generation with layout preservation"
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
        
        # Extract blocks using advanced layout parser
        blocks_with_pages = extract_blocks_from_pdf(file_content)
        
        if not blocks_with_pages:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": "Could not extract readable text blocks from PDF",
                    "filename": file.filename,
                    "file_size": len(file_content),
                    "suggestion": "Please try with a text-based PDF (not scanned images)"
                }
            )
        
        # Translate the extracted blocks
        texts = [block.text for block, _ in blocks_with_pages]
        translated_texts = await openai_translate_batch(texts, target_lang)
        
        if not translated_texts or len(translated_texts) != len(texts):
            raise HTTPException(status_code=500, detail="Translation returned incomplete results")
        
        # Combine blocks with translations
        blocks_with_translations = []
        for i, (block, page_num) in enumerate(blocks_with_pages):
            translated_text = translated_texts[i] if i < len(translated_texts) else block.text
            blocks_with_translations.append((block, page_num, translated_text))
        
        # Create PDF with advanced layout preservation
        translated_pdf_content = create_translated_pdf_with_layout(file_content, blocks_with_translations, file.filename)
        
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
        
        # Extract blocks using advanced layout parser
        blocks_with_pages = extract_blocks_from_pdf(file_content)
        
        if not blocks_with_pages:
            return JSONResponse({
                "success": False,
                "message": "Could not extract readable text blocks from PDF",
                "filename": file.filename,
                "file_size": len(file_content),
                "extracted_blocks": 0
            })
        
        # Get sample text from blocks
        sample_text = "\n\n".join([block.text[:100] + "..." if len(block.text) > 100 else block.text 
                                   for block, _ in blocks_with_pages[:3]])
        
        # Translate the extracted blocks
        texts = [block.text for block, _ in blocks_with_pages]
        translated_texts = await openai_translate_batch(texts, target_lang)
        
        # Combine blocks with translations
        blocks_with_translations = []
        for i, (block, page_num) in enumerate(blocks_with_pages):
            translated_text = translated_texts[i] if i < len(translated_texts) else block.text
            blocks_with_translations.append((block, page_num, translated_text))
        
        # Create PDF with advanced layout preservation
        translated_pdf_content = create_translated_pdf_with_layout(file_content, blocks_with_translations, file.filename)
        
        # Return JSON with base64-encoded PDF
        pdf_base64 = base64.b64encode(translated_pdf_content).decode('utf-8')
        
        return JSONResponse({
            "success": True,
            "message": "PDF translated successfully with advanced layout parser",
            "filename": f"translated_{file.filename}",
            "original_text": sample_text,
            "translated_text": "\n\n".join([trans[:100] + "..." if len(trans) > 100 else trans 
                                           for trans in translated_texts[:3]]),
            "source_lang": source_lang,
            "target_lang": target_lang,
            "pdf_base64": pdf_base64,
            "pdf_size": len(translated_pdf_content),
            "blocks_processed": len(blocks_with_pages),
            "version": "6.0-debug-advanced-layout"
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