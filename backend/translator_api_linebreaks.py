#!/usr/bin/env python3
"""
PDF Translation API v8.8 - Line Break Preserving Version
Translates to English while maintaining original line breaks and text structure
"""

import os
import sys
import tempfile
import shutil
import requests
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from PIL import Image
import io
import textwrap
from reportlab.pdfbase.pdfmetrics import stringWidth

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not found")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

app = FastAPI(title="PDF Translation API v8.8 - Line Break Preserving Edition")

def is_openai_error_message(text: str) -> bool:
    """Check if text is an OpenAI error message"""
    error_patterns = [
        "I'm sorry, but it seems that the text you provided",
        "I'm sorry, but there is no text provided",
        "I'm sorry, but it seems there is no text provided",
        "I'm sorry, but",
        "Could you please provide",
        "no text provided for translation",
        "text you provided is not clear or complete",
        "I'm sorry, but it seems",
        "I cannot translate",
        "Unable to translate"
    ]
    
    text_lower = text.lower()
    return any(pattern.lower() in text_lower for pattern in error_patterns)

def clean_translation_text(text: str) -> str:
    """Clean translation text by removing error messages"""
    if is_openai_error_message(text):
        return ""  # Return empty string for error messages
    return text.strip()

def preserve_line_breaks(original_text: str, translated_text: str) -> str:
    """Preserve the original line break structure in the translated text"""
    
    # Split original text into lines
    original_lines = original_text.split('\n')
    
    # Split translated text into lines
    translated_lines = translated_text.split('\n')
    
    # If we have the same number of lines, preserve the structure
    if len(original_lines) == len(translated_lines):
        return translated_text
    
    # If original has more lines, try to match structure
    if len(original_lines) > len(translated_lines):
        # Find where line breaks should be in translated text
        result_lines = []
        translated_words = translated_text.split()
        word_index = 0
        
        for original_line in original_lines:
            if not original_line.strip():  # Empty line
                result_lines.append("")
                continue
            
            # Count words in original line
            original_words = original_line.split()
            line_words = []
            
            # Take the same number of words for this line
            for _ in range(len(original_words)):
                if word_index < len(translated_words):
                    line_words.append(translated_words[word_index])
                    word_index += 1
                else:
                    break
            
            result_lines.append(" ".join(line_words))
        
        return "\n".join(result_lines)
    
    # If translated has more lines, use it as is
    return translated_text

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
                    visual_elements.append({
                        'type': 'drawing',
                        'page': page_num,
                        'bbox': rect,
                        'size': (width, height)
                    })
    
    doc.close()
    return visual_elements

def merge_close_blocks(text_blocks: List[Dict], vertical_threshold: float = 100.0) -> List[Dict]:
    """Merge text blocks that are close to each other vertically"""
    if not text_blocks:
        return []
    
    print(f"🔍 Analyzing {len(text_blocks)} blocks for merging (threshold: {vertical_threshold}pt)...")
    
    # Sort blocks by vertical position (top to bottom)
    sorted_blocks = sorted(text_blocks, key=lambda x: x['bbox'].y0)
    
    # Print all vertical distances
    print("📏 All vertical distances between consecutive blocks:")
    for i in range(len(sorted_blocks) - 1):
        current_block = sorted_blocks[i + 1]
        last_block = sorted_blocks[i]
        vertical_distance = abs(current_block['bbox'].y0 - last_block['bbox'].y1)
        font_size_diff = abs(current_block.get('font_size', 12) - last_block.get('font_size', 12))
        print(f"  Block {i+1} → {i+2}: {vertical_distance:.1f}pt (font diff: {font_size_diff:.1f}pt)")
    
    merged_blocks = []
    current_group = [sorted_blocks[0]]
    
    for i in range(1, len(sorted_blocks)):
        current_block = sorted_blocks[i]
        last_block = sorted_blocks[i-1]
        
        # Check if blocks are close vertically
        vertical_distance = abs(current_block['bbox'].y0 - last_block['bbox'].y1)
        
        # Check if blocks have similar formatting (font size within 2 points)
        font_size_diff = abs(current_block.get('font_size', 12) - last_block.get('font_size', 12))
        
        # Merge if close vertically and similar formatting
        if vertical_distance <= vertical_threshold and font_size_diff <= 2:
            current_group.append(current_block)
            print(f"  📦 Merging block {i+1} (distance: {vertical_distance:.1f}pt)")
        else:
            # Create merged block from current group
            if len(current_group) > 1:
                merged_block = create_merged_block(current_group)
                merged_blocks.append(merged_block)
                print(f"  ✅ Created merged block from {len(current_group)} blocks")
            else:
                merged_blocks.append(current_group[0])
            current_group = [current_block]
    
    # Handle the last group
    if len(current_group) > 1:
        merged_block = create_merged_block(current_group)
        merged_blocks.append(merged_block)
        print(f"  ✅ Created final merged block from {len(current_group)} blocks")
    else:
        merged_blocks.append(current_group[0])
    
    print(f"📦 Merged {len(text_blocks)} blocks into {len(merged_blocks)} blocks")
    return merged_blocks

def create_merged_block(blocks: List[Dict]) -> Dict:
    """Create a single block from multiple close blocks"""
    # Combine text with line breaks
    combined_text = "\n".join([block['text'] for block in blocks])
    
    # Combine original lines
    combined_original_lines = []
    for block in blocks:
        combined_original_lines.extend(block.get('original_lines', []))
    
    # Combine line styles
    combined_line_styles = []
    for block in blocks:
        combined_line_styles.extend(block.get('line_styles', []))
    
    # Calculate combined bounding box
    min_x = min([block['bbox'].x0 for block in blocks])
    min_y = min([block['bbox'].y0 for block in blocks])
    max_x = max([block['bbox'].x1 for block in blocks])
    max_y = max([block['bbox'].y1 for block in blocks])
    
    # Use the most common font size
    font_sizes = [block.get('font_size', 12) for block in blocks]
    most_common_font_size = max(set(font_sizes), key=font_sizes.count)
    
    return {
        'text': combined_text,
        'original_lines': combined_original_lines,
        'line_styles': combined_line_styles,
        'page': blocks[0]['page'],
        'bbox': fitz.Rect(min_x, min_y, max_x, max_y),
        'font_size': most_common_font_size,
        'merged_from': len(blocks)
    }

def extract_text_blocks_with_structure(pdf_path: str) -> List[Dict]:
    """Extract text blocks with positioning, line structure, and font style from PDF"""
    doc = fitz.open(pdf_path)
    text_blocks = []
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                lines_content = []
                lines_style = []
                for line in block["lines"]:
                    line_text = ""
                    line_spans = []
                    for span in line["spans"]:
                        span_text = span["text"]
                        line_text += span_text
                        font = span.get("font", "")
                        size = span.get("size", 12)
                        # Heuristic for bold/italic
                        is_bold = bool(re.search(r"Bold|bold|Bd|BD", font))
                        is_italic = bool(re.search(r"Italic|italic|It|IT", font))
                        line_spans.append({
                            "text": span_text,
                            "font": font,
                            "size": size,
                            "is_bold": is_bold,
                            "is_italic": is_italic
                        })
                    lines_content.append(line_text.strip())
                    lines_style.append(line_spans)
                text_content = "\n".join(lines_content)
                if text_content.strip():
                    text_blocks.append({
                        'text': text_content,
                        'original_lines': lines_content,
                        'line_styles': lines_style,  # List of list of span dicts
                        'page': page_num,
                        'bbox': fitz.Rect(block["bbox"]),
                        'font_size': block["lines"][0]["spans"][0]["size"] if block["lines"] and block["lines"][0]["spans"] else 12
                    })
    doc.close()
    
    # Merge close blocks to prevent overlapping
    merged_blocks = merge_close_blocks(text_blocks)
    return merged_blocks

def translate_text_with_structure(original_text: str, target_language: str = "English") -> str:
    """Translate text to English while preserving line structure"""
    
    # Skip translation for very short or non-meaningful text
    if len(original_text.strip()) < 3 or original_text.strip() in ['~', '.', '..', '...', 'lt', ';;;', 'ê', '"', "'"]:
        return ""
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create a more structured prompt for line break preservation
    system_prompt = f"""You are a professional translator. Translate the given text to {target_language}. 
IMPORTANT: Preserve the exact line break structure of the original text. If the original has line breaks, 
maintain them in the translation. Return only the translation without any explanations or apologies. 
If the text is unclear or incomplete, return an empty string."""
    
    messages = [
        {
            "role": "system", 
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": f"Original text with line breaks:\n{original_text}"
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
        
        # Clean the translation to remove error messages
        cleaned_text = clean_translation_text(translated_text)
        
        if cleaned_text:
            # Preserve line breaks from original structure
            preserved_text = preserve_line_breaks(original_text, cleaned_text)
            return preserved_text
        
        return ""
        
    except Exception as e:
        print(f"Translation error: {e}")
        return ""  # Return empty string on error

def wrap_text_to_width(text, font_name, font_size, max_width):
    """Wrap a single line of text to fit within max_width using the given font."""
    words = text.split()
    if not words:
        return ['']
    lines = []
    current_line = words[0]
    for word in words[1:]:
        test_line = current_line + ' ' + word
        if stringWidth(test_line, font_name, font_size) <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines

def fit_text_to_box(c, text, font_name, font_size, box_width, box_height, min_font_size=6):
    """Word wrap and dynamically reduce font size to fit text within the bounding box."""
    lines = text.split('\n')
    fitted_font_size = font_size
    while True:
        wrapped_lines = []
        for line in lines:
            wrapped_lines.extend(wrap_text_to_width(line, font_name, fitted_font_size, box_width))
        line_height = fitted_font_size + 2
        total_height = len(wrapped_lines) * line_height
        if total_height <= box_height and all(stringWidth(l, font_name, fitted_font_size) <= box_width for l in wrapped_lines):
            break
        fitted_font_size -= 1
        if fitted_font_size < min_font_size:
            break
    # Truncate lines if still too many
    max_lines = int(box_height // line_height)
    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[:max_lines]
        if wrapped_lines:
            last = wrapped_lines[-1]
            while stringWidth(last + '...', font_name, fitted_font_size) > box_width and len(last) > 3:
                last = last[:-1]
            wrapped_lines[-1] = last + '...'
    return wrapped_lines, fitted_font_size, line_height

def create_translated_pdf_with_structure(text_blocks: List[Dict], visual_elements: List[Dict], target_language: str = "English") -> str:
    """Create translated PDF with line break and style preservation, dynamic fitting, and no overlap."""
    output_path = tempfile.mktemp(suffix='.pdf')
    c = canvas.Canvas(output_path, pagesize=letter)
    pages_data = {}
    max_page = 0
    for element in visual_elements + text_blocks:
        page_num = element['page']
        max_page = max(max_page, page_num)
        if page_num not in pages_data:
            pages_data[page_num] = {'visual': [], 'text': []}
        if 'data' in element:
            pages_data[page_num]['visual'].append(element)
        else:
            pages_data[page_num]['text'].append(element)
    for page_num in range(max_page + 1):
        if page_num > 0:
            c.showPage()
        page_data = pages_data.get(page_num, {'visual': [], 'text': []})
        print(f"Rendering visual elements for page {page_num}...")
        for element in page_data['visual']:
            try:
                bbox = element['bbox']
                img_data = element['data']
                pdf_height = 792
                rl_x = bbox.x0
                rl_y = pdf_height - bbox.y1
                width = bbox.width
                height = bbox.height
                if rl_y < 0:
                    height += rl_y
                    rl_y = 0
                if rl_y + height > pdf_height:
                    height = pdf_height - rl_y
                img_reader = ImageReader(io.BytesIO(img_data))
                c.drawImage(img_reader, rl_x, rl_y, width=width, height=height)
            except Exception as e:
                print(f"Error rendering visual element: {e}")
        print(f"Rendering text blocks for page {page_num}...")
        for block in page_data['text']:
            try:
                text_bbox = block['bbox']
                overlaps = False
                for visual in page_data['visual']:
                    visual_bbox = visual['bbox']
                    if (text_bbox.intersects(visual_bbox) and 
                        visual_bbox.width * visual_bbox.height > 10000):
                        overlaps = True
                        break
                if overlaps:
                    print("Text block overlaps with logo, skipping text rendering")
                    continue
                original_text = block['text']
                translated_text = translate_text_with_structure(original_text, "English")
                if not translated_text:
                    block_num = len([b for b in page_data['text'] if page_data['text'].index(b) <= page_data['text'].index(block)]) + 1
                    print(f"Block {block_num}: Skipped (empty translation)")
                    continue
                block_num = len([b for b in page_data['text'] if page_data['text'].index(b) <= page_data['text'].index(block)]) + 1
                print(f"Block {block_num}:")
                if block.get('merged_from', 1) > 1:
                    print(f"  Merged from {block['merged_from']} blocks")
                print(f"  Original: {original_text[:50]}{'...' if len(original_text) > 50 else ''}")
                print(f"  Translated: {translated_text[:50]}{'...' if len(translated_text) > 50 else ''}")
                # Use the style of the first line/span as a base
                line_styles = block.get('line_styles', [])
                base_font = 'NotoSans-Regular'
                base_size = block.get('font_size', 10)
                is_bold = False
                is_italic = False
                if line_styles and line_styles[0]:
                    first_span = line_styles[0][0]
                    is_bold = first_span.get('is_bold', False)
                    is_italic = first_span.get('is_italic', False)
                    # Map font to Noto/Helvetica
                    if is_bold:
                        base_font = 'NotoSans-Bold' if 'NotoSans-Bold' in pdfmetrics.getRegisteredFontNames() else 'Helvetica-Bold'
                    else:
                        base_font = 'NotoSans-Regular' if 'NotoSans-Regular' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'
                # Fit text to box
                box_width = text_bbox.width
                box_height = text_bbox.height
                fitted_lines, fitted_font_size, line_height = fit_text_to_box(
                    c, translated_text, base_font, base_size, box_width, box_height)
                pdf_height = 792
                x = text_bbox.x0
                y = pdf_height - text_bbox.y1
                c.setFont(base_font, fitted_font_size)
                for i, line in enumerate(fitted_lines):
                    if line.strip():
                        line_y = y - (i * line_height)
                        if line_y > 0 and line_y > y - box_height:
                            c.drawString(x, line_y, line.strip())
            except Exception as e:
                print(f"Error rendering text block: {e}")
    c.save()
    return output_path

@app.on_event("startup")
async def startup_event():
    """Initialize fonts on startup"""
    print("🚀 Initializing PDF Translation API v8.8 - Line Break Preserving Edition")
    registered_fonts = ensure_google_noto_fonts()
    print(f"✅ Ready with {len(registered_fonts)} Google Noto fonts registered")

@app.post("/translate-pdf")
async def translate_pdf_endpoint(file: UploadFile = File(...), target_language: str = "English"):
    """Translate PDF to English with line break preservation"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    temp_input = None
    temp_output = None
    
    try:
        print(f"Processing PDF: {file.filename}")
        
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
        
        # Extract text blocks with structure
        text_blocks = extract_text_blocks_with_structure(temp_input)
        print(f"Extracted {len(text_blocks)} text blocks with line structure")
        
        # Create translated PDF with line break preservation
        temp_output = create_translated_pdf_with_structure(text_blocks, visual_elements, target_language)
        
        print(f"✅ Line break preserving translation completed successfully!")
        
        # Return the translated PDF
        return FileResponse(
            temp_output,
            media_type='application/pdf',
            filename=f"linebreak_translated_{file.filename}"
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
    return {"status": "healthy", "version": "8.8-linebreak"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004) 