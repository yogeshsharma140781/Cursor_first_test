#!/usr/bin/env python3
"""
PDF Translation API v8.8 - Using Latest Google Noto Fonts for Perfect Unicode Support
Incorporates the successfully tested Google Noto fonts for proper Devanagari rendering
"""

import os
import sys
import tempfile
import shutil
import requests
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse

import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from PIL import Image
import io

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not found")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

app = FastAPI(title="PDF Translation API v8.8 - Google Noto Edition")

def ensure_google_noto_fonts():
    """Ensure Google Noto fonts are downloaded and registered"""
    
    fonts_dir = Path.home() / 'Library' / 'Fonts'
    fonts_dir.mkdir(parents=True, exist_ok=True)
    
    # Google Fonts GitHub repository (official source)
    github_fonts = {
        'NotoSansDevanagari-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf',
        'NotoSansDevanagari-Bold.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Bold.ttf',
        'NotoSans-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf',
        'NotoSans-Bold.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf',
        'NotoSansArabic-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansArabic/NotoSansArabic-Regular.ttf',
        'NotoSansArabic-Bold.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansArabic/NotoSansArabic-Bold.ttf',
    }
    
    # Download missing fonts
    for font_name, url in github_fonts.items():
        font_path = fonts_dir / font_name
        
        if not font_path.exists():
            try:
                print(f"Downloading {font_name}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(font_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Downloaded: {font_path}")
                
            except Exception as e:
                print(f"❌ Failed to download {font_name}: {e}")
    
    # Register fonts with ReportLab
    font_registry = {
        'NotoSansDevanagari-Regular': 'NotoSansDevanagari-Regular.ttf',
        'NotoSansDevanagari-Bold': 'NotoSansDevanagari-Bold.ttf',
        'NotoSans-Regular': 'NotoSans-Regular.ttf', 
        'NotoSans-Bold': 'NotoSans-Bold.ttf',
        'NotoSansArabic-Regular': 'NotoSansArabic-Regular.ttf',
        'NotoSansArabic-Bold': 'NotoSansArabic-Bold.ttf',
    }
    
    registered_fonts = []
    
    for font_name, filename in font_registry.items():
        font_path = fonts_dir / filename
        
        if font_path.exists() and font_name not in pdfmetrics.getRegisteredFontNames():
            try:
                pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                registered_fonts.append(font_name)
                print(f"✅ Registered Unicode font: {font_name} -> {font_path}")
                
            except Exception as e:
                print(f"❌ Failed to register {font_name}: {e}")
    
    return registered_fonts

def detect_script_language(text: str) -> str:
    """Detect the primary script/language of text"""
    
    if not text.strip():
        return 'en'
    
    # Count characters by Unicode ranges
    script_counts = {
        'hi': 0,  # Devanagari (Hindi)
        'ar': 0,  # Arabic
        'zh': 0,  # Chinese
        'ja': 0,  # Japanese
        'ko': 0,  # Korean
        'en': 0   # Latin/English
    }
    
    for char in text:
        code = ord(char)
        
        if 0x0900 <= code <= 0x097F:  # Devanagari
            script_counts['hi'] += 1
        elif 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:  # Arabic
            script_counts['ar'] += 1
        elif 0x4E00 <= code <= 0x9FFF:  # CJK Unified Ideographs
            script_counts['zh'] += 1
        elif 0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF:  # Hiragana/Katakana
            script_counts['ja'] += 1
        elif 0xAC00 <= code <= 0xD7AF:  # Hangul
            script_counts['ko'] += 1
        elif (0x0020 <= code <= 0x007F) or (0x00A0 <= code <= 0x00FF):  # Latin
            script_counts['en'] += 1
    
    # Return the script with highest count
    detected_script = max(script_counts, key=script_counts.get)
    return detected_script

def get_best_font_for_language(language: str, bold: bool = False) -> str:
    """Get the best Google Noto font for a given language"""
    
    weight = 'Bold' if bold else 'Regular'
    
    font_mapping = {
        'hi': f'NotoSansDevanagari-{weight}',  # Hindi/Devanagari
        'ar': f'NotoSansArabic-{weight}',      # Arabic
        'zh': f'NotoSans-{weight}',            # Chinese (fallback to NotoSans)
        'ja': f'NotoSans-{weight}',            # Japanese (fallback to NotoSans)  
        'ko': f'NotoSans-{weight}',            # Korean (fallback to NotoSans)
        'en': f'NotoSans-{weight}',            # English/Latin
    }
    
    selected_font = font_mapping.get(language, f'NotoSans-{weight}')
    
    # Verify font is registered
    if selected_font in pdfmetrics.getRegisteredFontNames():
        return selected_font
    else:
        # Fallback to Helvetica if Noto font not available
        return 'Helvetica-Bold' if bold else 'Helvetica'

def extract_visual_elements(pdf_path: str) -> List[Dict]:
    """Extract logos, images, and visual elements from PDF"""
    
    doc = fitz.open(pdf_path)
    visual_elements = []
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        
        # Extract images
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                    bbox = page.get_image_bbox(img)
                    
                    visual_elements.append({
                        'type': 'image',
                        'page': page_num,
                        'bbox': bbox,
                        'data': img_data,
                        'size': (pix.width, pix.height)
                    })
                
                pix = None
                
            except Exception as e:
                print(f"Error extracting image {img_index}: {e}")
        
        # Extract vector drawings (logos, shapes)
        drawings = page.get_drawings()
        for drawing in drawings:
            rect = drawing.get('rect')
            
            if rect:
                width = rect.width
                height = rect.height
                
                # Skip very small elements (likely decorative lines)
                if width > 20 and height > 20:
                    # Determine if it's likely a logo based on size
                    is_logo = (width > 30 and height > 30) or (width * height > 1000)
                    
                    if is_logo:
                        # Render drawing as image
                        try:
                            mat = fitz.Matrix(2, 2)  # 2x scale for quality
                            pix = page.get_pixmap(matrix=mat, clip=rect)
                            img_data = pix.tobytes("png")
                            
                            visual_elements.append({
                                'type': 'logo',
                                'page': page_num, 
                                'bbox': rect,
                                'data': img_data,
                                'size': (width, height)
                            })
                            
                            pix = None
                            
                        except Exception as e:
                            print(f"Error rendering drawing: {e}")
    
    doc.close()
    return visual_elements

def extract_text_blocks(pdf_path: str) -> List[Dict]:
    """Extract text blocks with positions from PDF"""
    
    doc = fitz.open(pdf_path)
    text_blocks = []
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" in block:
                block_text = ""
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"]
                    block_text += "\n"
                
                block_text = block_text.strip()
                
                if block_text:
                    text_blocks.append({
                        'text': block_text,
                        'page': page_num,
                        'bbox': fitz.Rect(block["bbox"])
                    })
    
    doc.close()
    return text_blocks

def translate_text(text: str, target_language: str = "English") -> str:
    """Translate text using OpenAI GPT-4o-mini"""
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {
            "role": "system", 
            "content": f"You are a professional translator. Translate the given text to {target_language}. Maintain proper formatting, capitalization, and professional tone. Return only the translation without any explanations."
        },
        {
            "role": "user", 
            "content": text
        }
    ]
    
    data = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(OPENAI_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        translated_text = result["choices"][0]["message"]["content"].strip()
        
        return translated_text
        
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text on error

def create_translated_pdf(text_blocks: List[Dict], visual_elements: List[Dict], target_language: str = "English") -> str:
    """Create translated PDF with Google Noto fonts"""
    
    output_path = tempfile.mktemp(suffix='.pdf')
    c = canvas.Canvas(output_path, pagesize=letter)
    
    # Group elements by page
    pages_data = {}
    max_page = 0
    
    for element in visual_elements + text_blocks:
        page_num = element['page']
        max_page = max(max_page, page_num)
        
        if page_num not in pages_data:
            pages_data[page_num] = {'visual': [], 'text': []}
        
        if 'data' in element:  # Visual element
            pages_data[page_num]['visual'].append(element)
        else:  # Text element
            pages_data[page_num]['text'].append(element)
    
    # Process each page
    for page_num in range(max_page + 1):
        if page_num > 0:
            c.showPage()
        
        page_data = pages_data.get(page_num, {'visual': [], 'text': []})
        
        # Render visual elements first (backgrounds)
        print(f"Rendering visual elements for page {page_num}...")
        for element in page_data['visual']:
            try:
                bbox = element['bbox']
                img_data = element['data']
                
                # Convert PDF coordinates to ReportLab coordinates
                pdf_height = 792  # Letter page height in points
                rl_x = bbox.x0
                rl_y = pdf_height - bbox.y1
                width = bbox.width
                height = bbox.height
                
                # Handle positioning edge cases
                if rl_y < 0:
                    print(f"Logo extends above page boundary (Y: {rl_y}), positioning at page top")
                    height += rl_y  # Adjust height
                    rl_y = 0
                
                if rl_y + height > pdf_height:
                    height = pdf_height - rl_y
                
                print(f"Element positioning: PDF coords Y({bbox.y0}, {bbox.y1}) -> ReportLab Y({rl_y}, {rl_y + height})")
                
                # Create image reader from bytes
                img_reader = ImageReader(io.BytesIO(img_data))
                c.drawImage(img_reader, rl_x, rl_y, width=width, height=height)
                
                print(f"Rendered {element['type']} at ({rl_x}, {rl_y}, {width}x{height})")
                
            except Exception as e:
                print(f"Error rendering visual element: {e}")
        
        # Render translated text blocks
        print(f"Rendering text blocks for page {page_num}...")
        for block in page_data['text']:
            try:
                # Check if text overlaps with visual elements
                text_bbox = block['bbox']
                overlaps = False
                
                for visual in page_data['visual']:
                    visual_bbox = visual['bbox']
                    if (text_bbox.intersects(visual_bbox) and 
                        visual_bbox.width * visual_bbox.height > 10000):  # Large visual element
                        overlaps = True
                        break
                
                if overlaps:
                    print("Text block overlaps with logo, skipping text rendering")
                    continue
                
                # Translate the text
                original_text = block['text']
                translated_text = translate_text(original_text, target_language)
                
                print(f"Block {len([b for b in page_data['text'] if page_data['text'].index(b) <= page_data['text'].index(block)])}:")
                print(f"  Original: {original_text[:50]}{'...' if len(original_text) > 50 else ''}")
                print(f"  Translated: {translated_text[:50]}{'...' if len(translated_text) > 50 else ''}")
                
                # Detect language and select appropriate font
                detected_lang = detect_script_language(translated_text)
                
                # Try bold first for headers/titles
                is_bold = (len(original_text) < 50 or 
                          any(keyword in original_text.lower() for keyword in ['naam', 'adres', 'datum', 'nummer']))
                
                selected_font = get_best_font_for_language(detected_lang, bold=is_bold)
                
                print(f"Selected font '{selected_font}' for language '{detected_lang}' and text: '{translated_text[:50]}...'")
                
                # Position text
                pdf_height = 792
                x = text_bbox.x0
                y = pdf_height - text_bbox.y1
                
                # Set font and render
                font_size = 10
                c.setFont(selected_font, font_size)
                
                # Handle multi-line text
                lines = translated_text.split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        line_y = y - (i * (font_size + 2))
                        if line_y > 0:  # Only render if within page bounds
                            c.drawString(x, line_y, line.strip())
                
            except Exception as e:
                print(f"Error rendering text block: {e}")
    
    c.save()
    return output_path

@app.on_event("startup")
async def startup_event():
    """Initialize fonts on startup"""
    print("🚀 Initializing PDF Translation API v8.8 - Google Noto Edition")
    registered_fonts = ensure_google_noto_fonts()
    print(f"✅ Ready with {len(registered_fonts)} Google Noto fonts registered")

@app.post("/translate-pdf")
async def translate_pdf_endpoint(file: UploadFile = File(...), target_lang: str = Form("en")):
    """Translate PDF with perfect Unicode support using Google Noto fonts"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    temp_input = None
    temp_output = None
    
    try:
        print(f"Processing PDF: {file.filename}")
        
        # Map target_lang to full language name
        target_language_map = {
            'en': 'English',
            'hi': 'Hindi', 
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ar': 'Arabic',
            'nl': 'Dutch',
            'pl': 'Polish',
            'tr': 'Turkish',
            'uk': 'Ukrainian',
            'vi': 'Vietnamese'
        }
        target_language = target_language_map.get(target_lang, 'English')
        print(f"Target language: {target_language} (from code: {target_lang})")
        
        # Save uploaded file temporarily
        temp_input = tempfile.mktemp(suffix='.pdf')
        with open(temp_input, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        # Extract visual elements (logos, images)
        print("Extracting visual elements...")
        visual_elements = extract_visual_elements(temp_input)
        print(f"Found {len(visual_elements)} visual elements:")
        for i, elem in enumerate(visual_elements):
            print(f"  {i+1}. {elem['type']} at {elem['bbox']} - {elem['size'][0]}x{elem['size'][1]}")
        
        # Extract text blocks
        text_blocks = extract_text_blocks(temp_input)
        print(f"Extracted {len(text_blocks)} text blocks")
        
        # Create translated PDF
        temp_output = create_translated_pdf(text_blocks, visual_elements, target_language)
        
        print(f"✅ Translation completed successfully!")
        
        # Return the translated PDF
        return FileResponse(
            temp_output,
            media_type='application/pdf',
            filename=f"translated_{file.filename}"
        )
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
        
    finally:
        # Cleanup
        if temp_input and os.path.exists(temp_input):
            os.remove(temp_input)
            print(f"Cleaned up temporary file: {temp_input}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    registered_fonts = [font for font in pdfmetrics.getRegisteredFontNames() if 'Noto' in font]
    
    return {
        "status": "healthy",
        "version": "8.8 - Google Noto Edition", 
        "google_noto_fonts": len(registered_fonts),
        "fonts_available": registered_fonts
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002) 