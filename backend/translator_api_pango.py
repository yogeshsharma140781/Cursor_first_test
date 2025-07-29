#!/usr/bin/env python3

"""
Advanced Translation API with Pango/Cairo Text Rendering
Superior complex script support compared to ReportLab version
"""

import os
import tempfile
import shutil
import fitz  # PyMuPDF
import json
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageDraw
import io
from collections import defaultdict
from fuzzywuzzy import fuzz
import qrcode
import platform
import unicodedata

# Pango/Cairo imports
try:
    import gi
    gi.require_version('Pango', '1.0')
    gi.require_version('PangoCairo', '1.0')
    # Cairo is part of the main PyGObject installation, not a separate namespace
    from gi.repository import Pango, PangoCairo, GLib
    import cairo  # Use the direct pycairo import for Cairo
    PANGO_AVAILABLE = True
    print("✅ Pango/Cairo successfully imported")
except ImportError as e:
    print(f"❌ Warning: Pango/Cairo not available: {e}")
    PANGO_AVAILABLE = False

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import hashlib
from contextlib import asynccontextmanager

load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

# Unicode font configuration for Pango
PANGO_FONT_FAMILIES = {
    'default': 'Noto Sans',
    'devanagari': 'Noto Sans Devanagari',
    'arabic': 'Noto Sans Arabic',
    'cjk': 'Noto Sans CJK',
    'thai': 'Noto Sans Thai',
    'hebrew': 'Noto Sans Hebrew',
    'cyrillic': 'Noto Sans',
}

# Language to script mapping
LANGUAGE_SCRIPT_MAP = {
    'hi': 'devanagari', 'ne': 'devanagari', 'mr': 'devanagari',
    'ar': 'arabic', 'fa': 'arabic', 'ur': 'arabic',
    'he': 'hebrew', 'yi': 'hebrew',
    'th': 'thai', 'lo': 'thai',
    'km': 'khmer', 'my': 'myanmar',
    'ka': 'georgian', 'am': 'ethiopic',
    'ru': 'cyrillic', 'uk': 'cyrillic', 'bg': 'cyrillic', 'mk': 'cyrillic', 'sr': 'cyrillic'
}

def detect_script_from_text(text: str) -> str:
    """Detect script from text characters"""
    if not text:
        return 'default'
    
    # Count characters from different scripts
    script_counts = defaultdict(int)
    
    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            script_counts['devanagari'] += 1
        elif 0x0600 <= code <= 0x06FF:
            script_counts['arabic'] += 1
        elif 0x0590 <= code <= 0x05FF:
            script_counts['hebrew'] += 1
        elif 0x0E00 <= code <= 0x0E7F:
            script_counts['thai'] += 1
        else:
            script_counts['default'] += 1
    
    # Return the script with the highest count
    if script_counts:
        return max(script_counts.items(), key=lambda x: x[1])[0]
    return 'default'

def get_pango_font_for_language(target_lang: str, text: str = "", is_bold: bool = False) -> str:
    """Get appropriate Pango font family for language"""
    
    # Detect script from text if language mapping is not available
    script = LANGUAGE_SCRIPT_MAP.get(target_lang, detect_script_from_text(text))
    
    # Get base font family
    font_family = PANGO_FONT_FAMILIES.get(script, PANGO_FONT_FAMILIES['default'])
    
    # Add weight if bold
    if is_bold:
        return f"{font_family} Bold"
    
    return font_family

def is_qr_code_text(text: str) -> bool:
    """Detect if text looks like QR code content"""
    # QR codes often contain patterns like █▀▀▀▀▀█ ▀▀▀█▄█▀ █ ▄ █ ▀ █▀▀▀▀▀█
    qr_patterns = [
        r'[█▀▄▌▐░▒▓]',  # Block characters
        r'[▀▁▂▃▄▅▆▇█]',  # Block characters
        r'[▌▐░▒▓]',      # More block characters
    ]
    
    # Check if text contains many block characters
    block_char_count = sum(1 for char in text if any(re.search(pattern, char) for pattern in qr_patterns))
    return block_char_count > len(text) * 0.3  # More than 30% block characters

class SimpleTextBlock:
    """Simplified text block for translation"""
    def __init__(self, text: str, bbox: Tuple[float, float, float, float], block_type: str = "text", font: str = None, size: float = None, bold: bool = False, italic: bool = False):
        self.text = text
        self.bbox = bbox
        self.type = block_type
        self.font = font
        self.size = size or 12
        self.bold = bold
        self.italic = italic
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'bbox': self.bbox,
            'type': self.type,
            'font': self.font,
            'size': self.size,
            'bold': self.bold,
            'italic': self.italic
        }

class PangoAdvancedPDFLayoutParser:
    """Advanced PDF layout parser with Pango text rendering"""
    
    def __init__(self, require_api_key=True):
        if require_api_key and not API_KEY:
            raise ValueError("OpenAI API key is required")
        self.cache = {}
    
    async def translate_text_openai(self, text: str, target_lang: str = 'en') -> str:
        """Translate text using OpenAI API"""
        if not text.strip():
            return text
        
        # Check cache first
        cache_key = hashlib.md5(f"{text}_{target_lang}".encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Handle PRESERVE_ORIGINAL texts
        if text.startswith("PRESERVE_ORIGINAL:"):
            return text[18:]  # Remove prefix and return original
        
        # Preprocess text for translation
        processed_text = self._preprocess_for_translation(text)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4",
                        "messages": [
                            {
                                "role": "system", 
                                "content": f"You are a professional translator. Translate the following text to {target_lang}. Preserve formatting, capitalization patterns, and any technical terms. Do not add explanations or notes."
                            },
                            {
                                "role": "user", 
                                "content": processed_text
                            }
                        ],
                        "max_tokens": 1000,
                        "temperature": 0.1
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    translated = result['choices'][0]['message']['content'].strip()
                    
                    # Post-process translation
                    translated = self._convert_placeholders(translated)
                    translated = self._fix_translation_issues(translated, text)
                    
                    # Cache the result
                    self.cache[cache_key] = translated
                    return translated
                else:
                    print(f"❌ OpenAI API error: {response.status_code}")
                    return text
                    
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return text
    
    def _should_preserve_original(self, text: str) -> bool:
        """Determine if text should be preserved as original"""
        # URLs and web addresses
        if re.search(r'https?://[^\s]+', text):
            return True
        
        # Email addresses
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            return True
        
        # Phone numbers
        if re.search(r'[\+]?[0-9\s\-\(\)]{7,}', text):
            return True
        
        # Postal codes (various formats)
        if re.search(r'\b[A-Z]{1,2}[0-9][A-Z0-9]?\s*[0-9][A-Z]{2}\b', text):  # UK format
            return True
        if re.search(r'\b[0-9]{5}(?:-[0-9]{4})?\b', text):  # US format
            return True
        if re.search(r'\b[0-9]{4}\s*[A-Z]{2}\b', text):  # Dutch format
            return True
        
        # Reference numbers, case numbers, etc.
        if re.search(r'\b[A-Z][0-9]-[0-9]{9,}\b', text):  # Case numbers like Z1-186720992110
            return True
        
        # File extensions
        if re.search(r'\.[a-zA-Z]{2,4}\b', text):
            return True
        
        # Special preservation markers
        if text.startswith("PRESERVE_ORIGINAL:"):
            return True
        
        return False
    
    def _preprocess_for_translation(self, text: str) -> str:
        """Preprocess text for translation"""
        if self._should_preserve_original(text):
            return f"PRESERVE_ORIGINAL:{text}"
        return text
    
    def _convert_placeholders(self, text: str) -> str:
        """Convert placeholders back to original text"""
        if text.startswith("PRESERVE_ORIGINAL:"):
            return text[18:]
        return text
    
    def _fix_translation_issues(self, translated: str, original: str) -> str:
        """Fix common translation issues"""
        # Preserve original if translation seems wrong
        if len(translated) < len(original) * 0.3:  # Too short
            return original
        
        # Preserve original if it contains special patterns
        if self._should_preserve_original(original):
            return original
        
        return translated
    
    def extract_blocks_from_pdf(self, pdf_content: bytes) -> List[Tuple[SimpleTextBlock, int]]:
        """Extract text blocks from PDF using PyMuPDF"""
        blocks = []
        
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_dict = page.get_text("dict")
            
            for block in text_dict["blocks"]:
                if "lines" in block:  # Text block
                    block_text = ""
                    bbox = block["bbox"]
                    
                    # Extract font information
                    font_name = None
                    font_size = 12
                    is_bold = False
                    is_italic = False
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]
                            
                            # Get font info from first span
                            if font_name is None:
                                font_name = span.get("font", "")
                                font_size = span.get("size", 12)
                                is_bold = "bold" in font_name.lower()
                                is_italic = "italic" in font_name.lower()
                        
                        block_text += " "
                    
                    block_text = block_text.strip()
                    
                    # Skip QR code text blocks - they will be handled as visual elements
                    if is_qr_code_text(block_text):
                        print(f"🔍 Detected QR code text: {block_text[:30]}...")
                        continue
                    
                    if block_text:
                        text_block = SimpleTextBlock(
                            text=block_text,
                            bbox=bbox,
                            font=font_name,
                            size=font_size,
                            bold=is_bold,
                            italic=is_italic
                        )
                        blocks.append((text_block, page_num))
        
        doc.close()
        return blocks
    
    def extract_visual_elements(self, pdf_content: bytes, page_num: int = 0) -> List[Dict]:
        """Extract visual elements from PDF including QR codes"""
        visual_elements = []
        
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        if page_num < len(doc):
            page = doc.load_page(page_num)
            
            # Extract images
            images = page.get_images()
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        
                        # Get image rectangle
                        img_rects = page.get_image_rects(img)
                        if img_rects:
                            bbox = img_rects[0]
                            visual_elements.append({
                                "type": "image",
                                "subtype": "raster",
                                "bbox": tuple(bbox),
                                "data": img_data,
                                "page": page_num
                            })
                    
                    pix = None
                except:
                    continue
            
            # Extract QR codes from text blocks
            text_dict = page.get_text("dict")
            for block in text_dict["blocks"]:
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"]
                        block_text += " "
                    
                    block_text = block_text.strip()
                    
                    # If this looks like QR code text, create a visual element
                    if is_qr_code_text(block_text):
                        bbox = block["bbox"]
                        # Create a simple QR code visual representation
                        visual_elements.append({
                            "type": "qr_code",
                            "subtype": "qr_code",
                            "bbox": tuple(bbox),
                            "data": block_text,
                            "page": page_num
                        })
        
        doc.close()
        return visual_elements

def calculate_font_size_for_fit(layout: Pango.Layout, text: str, max_width: float, max_height: float, initial_size: int = 12) -> int:
    """Calculate appropriate font size to fit text within bounds"""
    font_size = initial_size
    
    # Try decreasing font size until text fits
    while font_size > 6:  # Minimum font size
        # Create font description with size
        font_desc = Pango.FontDescription.from_string(f"Noto Sans {font_size}")
        layout.set_font_description(font_desc)
        layout.set_text(text, -1)
        
        # Get text dimensions
        width, height = layout.get_size()
        width_px = width / Pango.SCALE
        height_px = height / Pango.SCALE
        
        if width_px <= max_width and height_px <= max_height:
            break
        
        font_size -= 1
    
    return font_size

def create_pango_pdf(pdf_content: bytes, blocks_with_translations: list, visual_elements: list) -> bytes:
    """Create PDF using Pango/Cairo for text rendering"""
    
    if not PANGO_AVAILABLE:
        raise ImportError("Pango/Cairo not available. Install with: pip install pycairo PyGObject")
    
    # Get original PDF dimensions
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    page_sizes = [(page.rect.width, page.rect.height) for page in doc]
    doc.close()
    
    # Create output PDF
    output_buffer = io.BytesIO()
    
    # Create Cairo surface for PDF
    if page_sizes:
        page_width, page_height = page_sizes[0]
    else:
        page_width, page_height = 595, 842  # A4 default
        
    surface = cairo.PDFSurface(output_buffer, page_width, page_height)
    ctx = cairo.Context(surface)
    
    # Create Pango layout
    layout = PangoCairo.create_layout(ctx)
    
    for page_num, (current_page_width, current_page_height) in enumerate(page_sizes):
        if page_num > 0:
            surface.set_size(current_page_width, current_page_height)
            surface.show_page()
        
        # Set white background
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()
        
        print(f"🔤 Rendering page {page_num + 1} with Pango/Cairo...")
        
        # Render visual elements first
        page_visual_elements = [elem for elem in visual_elements if elem.get("page", 0) == page_num]
        for elem in page_visual_elements:
            render_visual_element_cairo(ctx, elem, current_page_height)
        
        # Render text blocks
        text_blocks_on_page = [(block, blk_page_num, translated_text) 
                              for block, blk_page_num, translated_text in blocks_with_translations 
                              if blk_page_num == page_num]
        
        for block, blk_page_num, translated_text in text_blocks_on_page:
            try:
                # Convert PDF coordinates to Cairo coordinates
                cairo_x = block.bbox[0]
                cairo_y = current_page_height - block.bbox[3]  # Flip Y coordinate
                block_width = block.bbox[2] - block.bbox[0]
                block_height = block.bbox[3] - block.bbox[1]
                
                # Determine font based on text content and original styling
                font_name = "Noto Sans"
                font_size = 12
                bold = False
                
                # Check if text contains Devanagari characters
                if any('\u0900' <= char <= '\u097F' for char in translated_text):
                    font_name = "Noto Sans Devanagari"
                    print(f"✅ Pango rendered '{font_name}': '{translated_text[:30]}...'")
                else:
                    print(f"✅ Pango rendered '{font_name}': '{translated_text[:30]}...'")
                
                # Check if original text was bold
                if hasattr(block, 'font') and 'bold' in block.font.lower():
                    bold = True
                    font_name += " Bold"
                
                # Render text with Pango
                success = render_text_with_pango(
                    ctx, translated_text, cairo_x, cairo_y, 
                    block_width, block_height, font_name, font_size, bold
                )
                
                if not success:
                    print(f"❌ Failed to render text: {translated_text[:50]}...")
                    
            except Exception as e:
                print(f"❌ Error rendering text block: {e}")
                continue
    
    surface.finish()
    output_buffer.seek(0)
    return output_buffer.getvalue()

def render_visual_element_cairo(ctx: cairo.Context, element: Dict, page_height: float):
    """Render visual element using Cairo"""
    try:
        bbox = element["bbox"]
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        
        if element["subtype"] == "raster":
            # Load image data
            image_bytes = element["data"]
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGBA for consistency
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            
            # Convert PIL image to numpy array
            arr = np.array(image)
            height_img, width_img = arr.shape[:2]
            
            # Create Cairo surface from numpy array
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width_img, height_img)
            buf = surface.get_data()
            
            # Convert RGBA to BGRA and copy to Cairo surface
            arr_bgra = arr.copy()
            if arr.shape[2] == 4:  # RGBA
                arr_bgra[:, :, [0, 2]] = arr[:, :, [2, 0]]  # Swap R and B
                buf[:] = arr_bgra.tobytes()
            
            surface.mark_dirty()
            
            # Draw image
            ctx.save()
            ctx.translate(x0, y0)
            ctx.scale(width / width_img, height / height_img)
            ctx.set_source_surface(surface, 0, 0)
            ctx.paint()
            ctx.restore()
            
            print(f"✅ Cairo rendered image at ({x0:.1f}, {y0:.1f}, {width:.1f}x{height:.1f})")
        
        elif element["subtype"] == "qr_code":
            # Render QR code as a visual element
            qr_text = element["data"]
            
            # Create a simple QR code representation
            ctx.save()
            ctx.set_source_rgb(0, 0, 0)  # Black
            ctx.rectangle(x0, y0, width, height)
            ctx.set_line_width(1)
            ctx.stroke()
            
            # Add QR code label
            ctx.set_font_size(8)
            ctx.move_to(x0 + 2, y0 + height - 2)
            ctx.show_text("QR Code")
            ctx.restore()
            
            print(f"✅ Cairo rendered QR code at ({x0:.1f}, {y0:.1f}, {width:.1f}x{height:.1f})")
        
    except Exception as e:
        print(f"❌ Error rendering visual element with Cairo: {e}")
        # Draw placeholder rectangle
        ctx.save()
        ctx.set_source_rgb(0.9, 0.9, 0.9)
        ctx.rectangle(x0, y0, x1-x0, y1-y0)
        ctx.fill()
        ctx.restore()

def render_text_with_pango(ctx, text, x, y, width, height, font_name="Noto Sans", font_size=12, bold=False):
    """Render text using Pango with proper font size setting"""
    try:
        # Save context state
        ctx.save()
        
        # Create Pango layout
        layout = PangoCairo.create_layout(ctx)
        
        # Set font description with size
        font_desc = Pango.FontDescription.from_string(f"{font_name} {font_size}")
        if bold:
            font_desc.set_weight(Pango.Weight.BOLD)
        layout.set_font_description(font_desc)
        
        # Set text
        layout.set_text(text, -1)
        
        # Set width constraint if provided
        if width > 0:
            layout.set_width(int(width * Pango.SCALE))
        
        # Get text dimensions
        text_width, text_height = layout.get_size()
        text_width = text_width / Pango.SCALE
        text_height = text_height / Pango.SCALE
        
        # Check if text fits in the available space
        if text_width > width and width > 0:
            # Calculate new font size to fit
            scale_factor = width / text_width
            new_font_size = max(6, font_size * scale_factor)
            print(f"🔧 Adjusted font size from {font_size} to {new_font_size:.2f} to fit text")
            
            # Update font description with new size
            font_desc = Pango.FontDescription.from_string(f"{font_name} {new_font_size}")
            if bold:
                font_desc.set_weight(Pango.Weight.BOLD)
            layout.set_font_description(font_desc)
        
        # CRITICAL: Set text color to black BEFORE rendering
        ctx.set_source_rgb(0, 0, 0)  # Pure black color
        ctx.set_operator(cairo.OPERATOR_OVER)  # Ensure normal blending
        
        # Move to position and render
        ctx.move_to(x, y)
        PangoCairo.show_layout(ctx, layout)
        
        # Restore context state
        ctx.restore()
        
        return True
        
    except Exception as e:
        print(f"❌ Pango rendering error: {e}")
        
        # Restore context if we saved it
        try:
            ctx.restore()
        except:
            pass
        
        # Fallback to simple Cairo text rendering
        try:
            print(f"🔄 Attempting Cairo fallback rendering...")
            
            # Save context for fallback
            ctx.save()
            
            # Set font size for Cairo fallback
            fallback_font_size = font_size
            if width > 0 and len(text) > 0:
                # Estimate if text needs smaller font
                estimated_width = len(text) * fallback_font_size * 0.6  # Rough estimate
                if estimated_width > width:
                    fallback_font_size = max(6, width / (len(text) * 0.6))
            
            # Set font for Cairo
            ctx.set_font_size(fallback_font_size)
            ctx.set_source_rgb(0, 0, 0)  # Black color
            ctx.set_operator(cairo.OPERATOR_OVER)  # Ensure normal blending
            
            # Simple text rendering
            ctx.move_to(x, y + fallback_font_size)  # Adjust Y position for baseline
            ctx.show_text(text[:100])  # Limit text length to avoid overflow
            
            # Restore context
            ctx.restore()
            
            print(f"✅ Cairo fallback successful with font size {fallback_font_size}")
            return True
            
        except Exception as fallback_error:
            print(f"❌ Even Cairo fallback failed: {fallback_error}")
            # Try to restore context
            try:
                ctx.restore()
            except:
                pass
            return False

# FastAPI setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Pango-based Translation API...")
    if PANGO_AVAILABLE:
        print("✅ Pango/Cairo available for superior text rendering")
    else:
        print("⚠️  Pango/Cairo not available. Install with: brew install pygobject3 gtk+3 cairo && pip install pycairo PyGObject")
    yield
    # Shutdown
    print("🛑 Shutting down Pango Translation API...")

app = FastAPI(
    title="Pango Translation API",
    description="Advanced PDF translation with Pango text rendering",
    version="1.0.0-pango",
    lifespan=lifespan
)

# Enable CORS
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

@app.get("/")
async def root():
    return {
        "message": "Pango-based Translation API",
        "version": "1.0.0-pango",
        "features": [
            "Superior complex script rendering with Pango",
            "Better text shaping for Devanagari, Arabic, CJK",
            "Cairo-based PDF generation",
            "Font fallback handling",
            "Visual elements preservation",
            "QR code detection and handling",
            "Dynamic font size adjustment",
            "Automatic language detection",
            "Smart text preservation for addresses/URLs"
        ],
        "pango_available": PANGO_AVAILABLE,
        "status": "ready"
    }

@app.post("/translate")
async def translate(request: TranslationRequest):
    """Translate simple text"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    parser = PangoAdvancedPDFLayoutParser()
    translated = await parser.translate_text_openai(request.text, request.target_lang)
    
    return {
        "original": request.text,
        "translated": translated,
        "source_lang": request.source_lang,
        "target_lang": request.target_lang
    }

@app.post("/translate-pdf")
async def translate_pdf_pango(
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("en")
):
    """Translate PDF using Pango for text rendering"""
    
    if not PANGO_AVAILABLE:
        raise HTTPException(status_code=500, detail="Pango/Cairo not available. Install with: brew install pygobject3 gtk+3 cairo && pip install pycairo PyGObject")
    
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        # Read PDF content
        pdf_content = await file.read()
        
        # Initialize parser
        parser = PangoAdvancedPDFLayoutParser()
        
        # Extract text blocks
        print("📄 Extracting text blocks...")
        blocks = parser.extract_blocks_from_pdf(pdf_content)
        print(f"📄 Found {len(blocks)} text blocks")
        
        # Extract visual elements
        print("🖼️  Extracting visual elements...")
        visual_elements = parser.extract_visual_elements(pdf_content, 0)
        print(f"🖼️  Found {len(visual_elements)} visual elements")
        
        # Translate text blocks
        print("🌐 Translating text blocks...")
        blocks_with_translations = []
        
        for i, (block, page_num) in enumerate(blocks):
            print(f"Block {i+1}: Original: {block.text[:50]}...")
            
            translated = await parser.translate_text_openai(block.text, target_lang)
            blocks_with_translations.append((block, page_num, translated))
            
            print(f"  Translated: {translated[:50]}...")
        
        # Create PDF with Pango rendering
        print("🔤 Creating PDF with Pango rendering...")
        output_pdf = create_pango_pdf(pdf_content, blocks_with_translations, visual_elements)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(output_pdf)
            tmp_path = tmp_file.name
        
        print(f"✅ Pango PDF translation complete!")
        
        # Return the PDF file as streaming response to avoid background task errors
        def generate_pdf():
            try:
                with open(tmp_path, 'rb') as f:
                    while chunk := f.read(8192):
                        yield chunk
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        return StreamingResponse(
            generate_pdf(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=translated_{target_lang}.pdf"}
        )
        
    except Exception as e:
        print(f"❌ Error in Pango PDF translation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 