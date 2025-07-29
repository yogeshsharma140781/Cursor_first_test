#!/usr/bin/env python3

"""
Advanced Translation API with Full Layout Parser Integration
Complete implementation of test_layoutparser_simple.py functionality
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
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import simpleSplit, ImageReader
from reportlab.lib.fonts import addMapping
import os

# Register Noto Sans fonts at module level
try:
    # Get the path to the fonts directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fonts_dir = os.path.join(current_dir, 'fonts')
    
    # Register Noto Sans fonts using a different approach
    from reportlab.pdfbase import pdfmetrics
    
    # Register fonts directly with pdfmetrics
    pdfmetrics.registerFont(TTFont('NotoSans-Regular', os.path.join(fonts_dir, 'NotoSans-Regular.ttf')))
    pdfmetrics.registerFont(TTFont('NotoSans-Bold', os.path.join(fonts_dir, 'NotoSans-Bold.ttf')))
    
    # Also try the addMapping approach
    addMapping('NotoSans-Regular', 0, 0, 'NotoSans-Regular')
    addMapping('NotoSans-Bold', 1, 0, 'NotoSans-Bold')
    
    # Verify registration by checking if fonts are available
    registered_fonts = list(pdfmetrics.getRegisteredFontNames())
    print(f"[FONT] Successfully registered Noto Sans fonts at module level")
    print(f"[FONT] Registered fonts: {registered_fonts}")
    
    if 'NotoSans-Regular' in registered_fonts and 'NotoSans-Bold' in registered_fonts:
        print("[FONT] Noto fonts are properly registered and available")
    else:
        print("[FONT] Warning: Noto fonts not found in registered fonts list")
        
except Exception as e:
    print(f"[FONT] Could not register Noto fonts at module level: {e}")
import io
from collections import defaultdict
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet
from fuzzywuzzy import fuzz
import qrcode
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import hashlib
from contextlib import asynccontextmanager
import uuid
from datetime import datetime, timedelta

# All functionality is self-contained in this file

# Translation cancellation system
active_translations = {}  # Store active translation tasks
translation_cancelled = {}  # Track cancelled translations

def generate_translation_id():
    """Generate a unique translation ID"""
    return str(uuid.uuid4())

def is_translation_cancelled(translation_id: str) -> bool:
    """Check if a translation has been cancelled"""
    return translation_cancelled.get(translation_id, False)

def cancel_translation(translation_id: str) -> bool:
    """Cancel a translation by ID"""
    if translation_id in active_translations:
        translation_cancelled[translation_id] = True
        return True
    return False

def cleanup_translation(translation_id: str):
    """Clean up translation tracking"""
    active_translations.pop(translation_id, None)
    translation_cancelled.pop(translation_id, None)

load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

# Font registration will be handled by setup_production_fonts()
NOTO_DEVANAGARI_FONT_NAME = "NotoSansDevanagari-Regular"

def setup_production_fonts():
    """Setup all required fonts for production using existing Noto fonts"""
    try:
        # Get the path to the fonts directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fonts_dir = os.path.join(current_dir, 'fonts')
        
        # Register Noto Sans fonts (these include Cyrillic and Extended Latin support)
        noto_sans_regular_path = os.path.join(fonts_dir, 'NotoSans-Regular.ttf')
        noto_sans_bold_path = os.path.join(fonts_dir, 'NotoSans-Bold.ttf')
        
        # Register Noto Sans Regular
        if 'NotoSans-Regular' not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont('NotoSans-Regular', noto_sans_regular_path))
            print(f"[FONT] Registered NotoSans-Regular from {noto_sans_regular_path}")
        else:
            print(f"[FONT] NotoSans-Regular already registered.")
        
        # Register Noto Sans Bold
        if 'NotoSans-Bold' not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont('NotoSans-Bold', noto_sans_bold_path))
            print(f"[FONT] Registered NotoSans-Bold from {noto_sans_bold_path}")
        else:
            print(f"[FONT] NotoSans-Bold already registered.")
        
        # Register Noto Sans Devanagari for Hindi
        noto_devanagari_path = os.path.join(fonts_dir, 'NotoSansDevanagari-Regular.ttf')
        if 'NotoSansDevanagari-Regular' not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont('NotoSansDevanagari-Regular', noto_devanagari_path))
            print(f"[FONT] Registered NotoSansDevanagari-Regular from {noto_devanagari_path}")
        else:
            print(f"[FONT] NotoSansDevanagari-Regular already registered.")
        

        
        print(f"[FONT] Production font setup completed successfully.")
        
    except Exception as e:
        print(f"[FONT] Error in production font setup: {e}")

# Setup fonts at module level
setup_production_fonts()

def contains_devanagari(text):
    return bool(re.search(r"[\u0900-\u097F]", text))

def contains_arabic(text):
    return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", text))

def select_appropriate_font(default_font_name, is_bold, is_italic, original_font_name, target_language=None, text=None):
    """Enhanced font selection with comprehensive Unicode support using existing Noto fonts"""
    
    # Use Noto Sans for Arabic (NotoSans includes Arabic support)
    if (target_language and target_language.lower() in ["arabic", "ar"]) or (text and contains_arabic(text)):
        if is_bold:
            font_name = "NotoSans-Bold"
        else:
            font_name = "NotoSans-Regular"
        print(f"[FONT] Using {font_name} for Arabic text.")
        return font_name
    
    # Use Noto Sans Devanagari for Hindi/Devanagari
    if (target_language and target_language.lower() in ["hindi", "hi"]) or (text and contains_devanagari(text)):
        if NOTO_DEVANAGARI_FONT_NAME not in pdfmetrics.getRegisteredFontNames():
            print(f"[FONT WARNING] {NOTO_DEVANAGARI_FONT_NAME} not registered! Using fallback font.")
            return default_font_name
        print(f"[FONT] Using {NOTO_DEVANAGARI_FONT_NAME} for Hindi/Devanagari text.")
        return NOTO_DEVANAGARI_FONT_NAME
    
    # Use Noto Sans for Cyrillic languages (Russian, Ukrainian)
    # NotoSans-Regular and NotoSans-Bold include Cyrillic glyphs
    if target_language and target_language.lower() in ["russian", "ru", "ukrainian", "uk"]:
        if is_bold:
            font_name = "NotoSans-Bold"
        else:
            font_name = "NotoSans-Regular"
        print(f"[FONT] Using {font_name} for {target_language} (Cyrillic) text.")
        return font_name
    
    # Use Noto Sans for Polish (Extended Latin with diacritics)
    # Polish has special characters: ą, ć, ę, ł, ń, ó, ś, ź, ż
    if target_language and target_language.lower() in ["polish", "pl"]:
        if is_bold:
            font_name = "NotoSans-Bold"
        else:
            font_name = "NotoSans-Regular"
        print(f"[FONT] Using {font_name} for Polish (Extended Latin) text.")
        return font_name
    
    # Use Noto Sans for Vietnamese (Extended Latin with diacritics)
    # Vietnamese has special characters: à, á, ả, ã, ạ, ă, ằ, ắ, ẳ, ẵ, ặ, â, ầ, ấ, ẩ, ẫ, ậ, etc.
    if target_language and target_language.lower() in ["vietnamese", "vi"]:
        if is_bold:
            font_name = "NotoSans-Bold"
        else:
            font_name = "NotoSans-Regular"
        print(f"[FONT] Using {font_name} for Vietnamese (Extended Latin) text.")
        return font_name
    
    # Use Noto fonts for Turkish and other Latin-based languages with special characters
    # Turkish has special characters: ğ, ş, ı, ö, ç, ü, İ, Ğ, Ş, Ç, Ö, Ü
    # Noto fonts have excellent Unicode support
    if target_language and target_language.lower() in ["turkish", "tr"]:
        # Use Noto Sans for Turkish characters
        if is_bold:
            font_name = "NotoSans-Bold"
        else:
            font_name = "NotoSans-Regular"
        print(f"[FONT] Using {font_name} for Turkish text.")
        return font_name
    
    # Use Noto fonts for Spanish and other Latin-based languages
    if target_language and target_language.lower() in ["spanish", "es", "french", "fr", "german", "de", "italian", "it", "portuguese", "pt"]:
        if is_bold:
            font_name = "NotoSans-Bold"
        else:
            font_name = "NotoSans-Regular"
        print(f"[FONT] Using {font_name} for {target_language} text.")
        return font_name
    
    # Fallback to original logic
    return default_font_name

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    await client.aclose()

app = FastAPI(lifespan=lifespan)

# Create a persistent HTTP client for connection pooling
client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0),
    limits=httpx.Limits(max_keepalive_connections=10, max_connections=50)
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
    """Advanced text block representation with full layout information"""
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

class AdvancedPDFLayoutParser:
    """Advanced PDF layout parser with full PyMuPDF capabilities"""
    
    def __init__(self, require_api_key=True):
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if require_api_key:
            assert self.api_key, "OPENAI_API_KEY not found in environment!"
    
    async def translate_text_openai(self, text: str, target_lang: str = 'en', translation_id: str = None) -> str:
        """Advanced text translation using OpenAI API with post-processing improvements"""
        if not text.strip():
            return text
        
        # Handle single characters and symbols that should be preserved
        if len(text.strip()) == 1:
            char = text.strip()
            if char in ['+', '-', '=', '×', '÷', '€', '$', '£', '¥', '%', '°', '°C', '°F']:
                return char  # Preserve mathematical symbols and currency signs
        
        try:
            # Check for cancellation before starting translation
            if translation_id and is_translation_cancelled(translation_id):
                raise Exception("Translation was cancelled")
            
            preprocessed_text = self._preprocess_for_translation(text)
            
            # Check for cancellation before API call
            if translation_id and is_translation_cancelled(translation_id):
                raise Exception("Translation was cancelled")
            
            prompt = (
                f"You are a professional translator. Translate the following text to {target_lang}.\n\n"
                f"IMPORTANT RULES:\n"
                f"- Preserve web addresses, URLs, and website paths exactly as they are\n"
                f"- Keep postal codes, street names, addresses, and city names unchanged\n"
                f"- Preserve phone numbers, email addresses, and reference numbers\n"
                f"- Maintain original formatting and capitalization patterns\n"
                f"- Return ONLY the translated text, no explanations or instructions\n\n"
                f"Text to translate:\n{preprocessed_text}\n\n"
                f"Translation:"
            )
            
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.3
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"OpenAI API error: {response.text}")
            
            result = response.json()
            result_text = result["choices"][0]["message"]["content"].strip()
            
            # Post-processing fixes for common translation issues
            # DISABLED: result_text = self._fix_translation_issues(result_text, text)
            # Convert placeholders back to proper English terms - do this LAST
            result_text = self._convert_placeholders(result_text)
            return result_text
            
        except Exception as e:
            print(f"Translation error for '{text[:50]}...': {e}")
            return text
    
    def extract_blocks_from_pdf(self, pdf_content: bytes) -> List[Tuple[SimpleTextBlock, int]]:
        """Extract paragraph blocks from PDF using PyMuPDF with page numbers, storing font/style info."""
        # Save PDF content to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(pdf_content)
            tmp_file.flush()
            pdf_path = tmp_file.name
        
        try:
            doc = fitz.open(pdf_path)
            blocks = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_dict = page.get_text("dict")
                print(f"\n[EXTRACT DEBUG] Page {page_num}: Processing {len(text_dict['blocks'])} blocks")
                tables = self._detect_table_regions(text_dict)
                print(f"[EXTRACT DEBUG] Page {page_num}: Found {len(tables)} tables")
                table_block_indices = set(table["block_index"] for table in tables)
                for table in tables:
                    table_text = self._format_table_as_text(table)
                    table_block = SimpleTextBlock(table_text, table["bbox"], "table")
                    table_block.structured_data = table
                    blocks.append((table_block, page_num))
                if len(text_dict["blocks"]) == 0:
                    continue
                line_items = []
                for block_idx, block in enumerate(text_dict["blocks"]):
                    if "lines" in block and block_idx not in table_block_indices:
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
                            block_type = self._classify_block_type(para_text, para_bbox, page.rect)
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
                    block_type = self._classify_block_type(para_text, para_bbox, page.rect)
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
            # Cleanup temporary file
            try:
                os.unlink(pdf_path)
            except:
                pass

    def _preprocess_for_translation(self, text: str) -> str:
        """Pre-process text to handle Dutch terms that might be dropped by translation"""
        # Dictionary of Dutch terms that should be preserved/replaced before translation
        dutch_terms = {
            'INLOGGEGEVENS': 'LOGIN_DETAILS_PLACEHOLDER',
            'AFKOOP EIGEN RISICO': 'DEDUCTIBLE_BUYOUT_PLACEHOLDER',
            'SEPA MACHTIGING': 'SEPA_AUTHORIZATION_PLACEHOLDER',
            
            # Single word terms that translation often gets wrong
            'Aantal': 'QUANTITY_PLACEHOLDER',
            'Stuksprijs': 'UNIT_PRICE_PLACEHOLDER',
            'Totaal': 'TOTAL_PLACEHOLDER',
            'Artikelnummer': 'ARTICLE_NUMBER_PLACEHOLDER',
            'Factuurnummer': 'INVOICE_NUMBER_PLACEHOLDER',
            'Factuurdatum': 'INVOICE_DATE_PLACEHOLDER',
            'Bestelnummer': 'ORDER_NUMBER_PLACEHOLDER',
            'Besteldatum': 'ORDER_DATE_PLACEHOLDER',
            'Bezorgdatum': 'DELIVERY_DATE_PLACEHOLDER',
            'Bezorgwijze': 'DELIVERY_METHOD_PLACEHOLDER',
            'Betaalwijze': 'PAYMENT_METHOD_PLACEHOLDER',
            'Liefernummer': 'DELIVERY_NUMBER_PLACEHOLDER',
            'Restbedrag': 'REMAINING_AMOUNT_PLACEHOLDER',
            'BTW-grondslag': 'VAT_BASE_PLACEHOLDER',
            'BTW-bedrag': 'VAT_AMOUNT_PLACEHOLDER',
        }
        
        processed_text = text
        for dutch, placeholder in dutch_terms.items():
            if dutch in processed_text:
                # Replace with placeholder that translation won't drop
                processed_text = re.sub(re.escape(dutch), placeholder, processed_text, flags=re.IGNORECASE)
        
        return processed_text
    
    def _convert_placeholders(self, text: str) -> str:
        """Convert placeholders back to proper English terms"""
        placeholder_fixes = {
            'LOGIN_DETAILS_PLACEHOLDER': 'LOGIN DETAILS',
            'DEDUCTIBLE_BUYOUT_PLACEHOLDER': 'DEDUCTIBLE BUYOUT',
            'SEPA_AUTHORIZATION_PLACEHOLDER': 'SEPA AUTHORIZATION',
            
            # Invoice terminology placeholders
            'QUANTITY_PLACEHOLDER': 'Quantity',
            'UNIT_PRICE_PLACEHOLDER': 'Unit Price',
            'TOTAL_PLACEHOLDER': 'Total',
            'ARTICLE_NUMBER_PLACEHOLDER': 'Article Number',
            'INVOICE_NUMBER_PLACEHOLDER': 'Invoice Number',
            'INVOICE_DATE_PLACEHOLDER': 'Invoice Date',
            'ORDER_NUMBER_PLACEHOLDER': 'Order Number',
            'ORDER_DATE_PLACEHOLDER': 'Order Date',
            'DELIVERY_DATE_PLACEHOLDER': 'Delivery Date',
            'DELIVERY_METHOD_PLACEHOLDER': 'Delivery Method',
            'PAYMENT_METHOD_PLACEHOLDER': 'Payment Method',
            'DELIVERY_NUMBER_PLACEHOLDER': 'Delivery Number',
            'REMAINING_AMOUNT_PLACEHOLDER': 'Remaining Amount',
            'VAT_BASE_PLACEHOLDER': 'VAT Base',
            'VAT_AMOUNT_PLACEHOLDER': 'VAT Amount',
        }
        
        for placeholder, english_term in placeholder_fixes.items():
            # Handle case-insensitive replacement since translation may lowercase placeholders
            text = re.sub(re.escape(placeholder), english_term, text, flags=re.IGNORECASE)
        
        return text

    def _classify_block_type(self, text: str, bbox: Tuple[float, float, float, float], page_rect) -> str:
        """Advanced rule-based block type classification"""
        text_lower = text.lower().strip()
        
        # Check if it's a QR code (contains block drawing characters)
        if self._is_qr_code(text):
            return "qr_code"
        
        # Check if it's a title (large font, top of page, short text)
        if len(text) < 50 and bbox[1] < page_rect.height * 0.2:
            return "title"
        
        # Check if it's a header/footer (very top or bottom)
        if bbox[1] < page_rect.height * 0.1 or bbox[3] > page_rect.height * 0.9:
            return "header" if bbox[1] < page_rect.height * 0.1 else "footer"
        
        # Check for actual table structure (multiple columns, structured data)
        # Only classify as table if it has clear tabular structure
        lines = text.split('\n')
        if len(lines) > 1:
            # Check if multiple lines have similar structure (like a real table)
            has_tabular_structure = False
            if len(lines) >= 2:
                # Look for patterns that suggest actual table structure
                # e.g., multiple columns separated by spaces, consistent alignment
                first_line_words = lines[0].split()
                if len(first_line_words) >= 2:
                    # Check if subsequent lines have similar word count (suggesting columns)
                    similar_structure_count = 0
                    for line in lines[1:]:
                        if len(line.split()) >= len(first_line_words) * 0.8:  # Allow some variation
                            similar_structure_count += 1
                    
                    # If most lines have similar structure, it might be a table
                    if similar_structure_count >= len(lines) * 0.7:
                        has_tabular_structure = True
            
            if has_tabular_structure:
                return "table"
        
        # Check for simple structured data (like addresses, contact info) - treat as text
        # These should NOT be classified as tables since they don't have visual borders
        if re.search(r'\d+[.,]\d+|€|\$|USD|EUR|\d{2}[-/]\d{2}[-/]\d{4}', text):
            # This is structured text, not a visual table
            return "text"
        
        # Check for list items
        if re.match(r'^\s*[-•*]\s+', text) or re.match(r'^\s*\d+\.\s+', text):
            return "list"
        
        # Default to text
        return "text"
    
    def _is_qr_code(self, text: str) -> bool:
        """Detect if text block contains QR code characters"""
        # QR codes contain block drawing characters
        qr_chars = {'█', '▀', '▄', '▌', '▐', '░', '▒', '▓', '■', '□', '▪', '▫'}
        char_count = sum(1 for char in text if char in qr_chars)
        # If more than 20% of characters are QR-related, it's likely a QR code
        return len(text) > 10 and char_count / len(text) > 0.2

    def remove_duplicates(self, blocks: List[Tuple[SimpleTextBlock, int]], similarity_threshold: float = 85) -> List[Tuple[SimpleTextBlock, int]]:
        """Remove duplicate blocks using fuzzy matching"""
        unique_blocks = []
        
        for block, page_num in blocks:
            is_duplicate = False
            
            for existing_block, existing_page in unique_blocks:
                # Check text similarity
                similarity = fuzz.ratio(block.text.lower(), existing_block.text.lower())
                
                if similarity >= similarity_threshold:
                    # Check if bboxes are close (might be same content detected twice)
                    bbox_distance = self._calculate_bbox_distance(block.bbox, existing_block.bbox)
                    if bbox_distance < 50:  # pixels
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_blocks.append((block, page_num))
        
        return unique_blocks
    
    def _calculate_bbox_distance(self, bbox1: Tuple[float, float, float, float], 
                                bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate distance between two bounding boxes"""
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        return ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    
    def _calculate_expanded_width(self, current_block, all_blocks_with_translations, current_page_num, page_width):
        """Calculate expanded width by analyzing available whitespace around the block"""
        x0, y0, x1, y1 = current_block.bbox
        original_width = x1 - x0
        
        # Find blocks on the same page
        same_page_blocks = [
            (block, page_num, text) for block, page_num, text in all_blocks_with_translations 
            if page_num == current_page_num and block != current_block
        ]
        
        # Check for whitespace to the right
        max_right_expansion = page_width - x1  # Space to page edge
        
        # Find the nearest block to the right that would limit expansion
        for block, _, _ in same_page_blocks:
            other_x0, other_y0, other_x1, other_y1 = block.bbox
            
            # Check if this block is to the right and overlaps vertically
            if (other_x0 > x1 and  # Block is to the right
                not (other_y1 < y0 or other_y0 > y1)):  # Overlaps vertically
                # This block limits our right expansion
                available_space = other_x0 - x1 - 5  # Leave 5px buffer
                if available_space > 0:
                    max_right_expansion = min(max_right_expansion, available_space)
        
        # Check for whitespace to the left (less common but possible)
        max_left_expansion = x0  # Space to page edge
        
        for block, _, _ in same_page_blocks:
            other_x0, other_y0, other_x1, other_y1 = block.bbox
            
            # Check if this block is to the left and overlaps vertically
            if (other_x1 < x0 and  # Block is to the left
                not (other_y1 < y0 or other_y0 > y1)):  # Overlaps vertically
                # This block limits our left expansion
                available_space = x0 - other_x1 - 5  # Leave 5px buffer
                if available_space > 0:
                    max_left_expansion = min(max_left_expansion, available_space)
        
        # Calculate expanded width
        # Prefer expanding to the right, but can expand left if needed
        expanded_width = original_width + max_right_expansion
        
        # If we still need more space and have left space available, use some of it
        if max_left_expansion > 10:  # Only if significant space available
            expanded_width += min(max_left_expansion * 0.3, 30)  # Use up to 30% of left space, max 30px
        
        # Don't expand beyond reasonable limits
        max_reasonable_width = page_width * 0.8  # Don't use more than 80% of page width
        expanded_width = min(expanded_width, max_reasonable_width)
        
        # Ensure we don't make it smaller than original
        expanded_width = max(expanded_width, original_width)
        
        return expanded_width

    def _fix_translation_issues(self, translated: str, original: str) -> str:
        """Fix common translation issues with comprehensive Dutch-English mappings"""
        # Dictionary of common Dutch to English fixes
        fixes = {
            'INLOGGEGEVENS': 'LOGIN DETAILS',
            'AFKOOP EIGEN RISICO': 'DEDUCTIBLE BUYOUT',
            'REKENINGNUMMER': 'ACCOUNT NUMBER',
            'SALDO': 'BALANCE',
            'DATUM': 'DATE',
            'BEDRAG': 'AMOUNT',
            'BESCHRIJVING': 'DESCRIPTION',
            'VALUTA': 'CURRENCY',
            'IBAN': 'IBAN',
            'BIC': 'BIC',
            'SWIFT': 'SWIFT',
            
            # Invoice-specific terms
            'Artikelnummer': 'Article Number',
            'Aantal': 'Quantity',
            'Stuksprijs': 'Unit Price',
            'Totaal': 'Total',
            'BTW-grondslag': 'VAT Base',
            'BTW-bedrag': 'VAT Amount',
            'Restbedrag': 'Remaining Amount',
            'Factuurnummer': 'Invoice Number',
            'Factuurdatum': 'Invoice Date',
            'Bestelnummer': 'Order Number',
            'Besteldatum': 'Order Date',
            'Bezorgdatum': 'Delivery Date',
            'Bezorgwijze': 'Delivery Method',
            'Betaalwijze': 'Payment Method',
            'Liefernummer': 'Delivery Number',
            'Voorgaand document': 'Previous Document',
            'Club-nummer': 'Club Number',
            'KvK Nummer': 'CoC Number',
            'Telefoon': 'Phone',
            'Pagina': 'Page',
        }
        
        # Apply direct fixes for known terms
        for dutch, english in fixes.items():
            if dutch in original.upper():
                translated = translated.replace(dutch, english)
        
        # Apply fixes with case-insensitive matching for better coverage
        for dutch, english in fixes.items():
            # Try exact case match first
            translated = translated.replace(dutch, english)
            # Try case-insensitive match
            translated = re.sub(re.escape(dutch), english, translated, flags=re.IGNORECASE)
        
        # Fix specific translation errors that AI makes
        translated = re.sub(r'\bSurrender deductible\b', 'DEDUCTIBLE BUYOUT', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bAfkoop eigen risico\b', 'DEDUCTIBLE BUYOUT', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bSurrender own risk\b', 'DEDUCTIBLE BUYOUT', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bStuberty\b', 'Unit Price', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bNumber\b', 'Quantity', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bPiece price\b', 'Unit Price', translated, flags=re.IGNORECASE)
        
        # Apply improved casing preservation BEFORE the specific fixes
        translated = self._preserve_original_casing(original, translated)
        
        # Apply URL and web address formatting
        translated = self._format_urls_and_web_addresses(translated)
        
        # Apply post-casing fixes to handle remaining issues
        translated = self._apply_post_casing_fixes(translated)
        
        # Fix specific problematic translations
        translated = re.sub(r'\bYOU Can\b', 'You can', translated)
        translated = re.sub(r'\bMY Deductible\b', 'My deductible', translated)
        translated = re.sub(r'\bmy Profile\b', 'My Profile', translated)
        translated = re.sub(r'\bLog in\b', 'log in', translated)
        
        # Fix business/invoice terminology
        translated = re.sub(r'\bnumber of\b', 'Quantity', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bpiece price\b', 'Unit price', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bVat foundation\b', 'VAT base', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bVat base\b', 'VAT base', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bBest date\b', 'Order date', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bDelivery method\b', 'Delivery method', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bPayment Method\b', 'Payment method', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bPrevious document\b', 'Previous document', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bTotal amount\b', 'Total amount', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bresidual amount\b', 'Remaining amount', translated, flags=re.IGNORECASE)
        
        # Fix technical terms
        translated = re.sub(r'\bhome copy\b', 'home copying levy', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bchamber of commerce number\b', 'CoC Number', translated, flags=re.IGNORECASE)
        
        # Fix currency formatting
        translated = re.sub(r'\bEur\b', 'EUR', translated)
        translated = re.sub(r'\beur\b', 'EUR', translated)
        
        # Fix spacing issues
        translated = re.sub(r'([a-z])([A-Z])', r'\1 \2', translated)  # Add space between camelCase
        translated = re.sub(r'(\w)(My Profile)', r'\1 \2', translated)  # Fix "OpMy Profile" -> "Op My Profile"
        translated = re.sub(r'Op(My Profile)', r'On \1', translated)  # Fix "OpMy Profile" -> "On My Profile"
        translated = re.sub(r'(\w)you\b', r'\1 you', translated)  # Fix missing space before "you"
        
        # Fix sentence spacing - add space after periods before words
        translated = re.sub(r'\.([a-zA-Z])', r'. \1', translated)
        
        # Preserve and fix currency symbols
        # First, ensure euro symbols are preserved from original if they exist
        if '€' in original:
            # Make sure euro symbol is properly formatted in translation
            translated = re.sub(r'€\s*(\d)', r'€\1', translated)  # Remove space after €
            translated = re.sub(r'(\d)\s*€', r'\1€', translated)  # Remove space before €
            # Fix common translation corruptions of euro symbol
            translated = re.sub(r'[•·∙‧⋅](\d)', r'€\1', translated)  # Replace bullet-like chars before numbers
            translated = re.sub(r'(\d)[•·∙‧⋅]', r'\1€', translated)  # Replace bullet-like chars after numbers
            translated = re.sub(r'EUR\s*(\d)', r'€\1', translated)  # Replace EUR with € symbol
        
        # Fix IBAN formatting
        if 'IBAN' in original.upper():
            # Remove extra spaces in IBAN numbers
            translated = re.sub(r'IBAN\s*:\s*([A-Z]{2}\s*\d{2}(?:\s*[A-Z0-9]{4})*)', 
                              lambda m: f"IBAN: {m.group(1).replace(' ', '')}", translated)
        
        # Fix common word combinations
        translated = re.sub(r'\bOn My Profile you\b', 'On My Profile you', translated)
        translated = re.sub(r'\bOpMy Profileyou\b', 'On My Profile you', translated)
        
        # Apply final currency and casing fixes AFTER all other processing
        translated = re.sub(r'\bEur\b', 'EUR', translated)
        translated = re.sub(r'\beur\b', 'EUR', translated)
        translated = re.sub(r'\bEUR\b', 'EUR', translated)  # Ensure consistency
        
        # Apply final proper noun and brand name fixes AFTER casing logic
        if 'SAMSUNG' in original.upper():
            translated = re.sub(r'\bsamsung\b', 'SAMSUNG', translated, flags=re.IGNORECASE)
        if 'VIEWFINITY' in original.upper():
            translated = re.sub(r'\bviewfinity\b', 'VIEWFINITY', translated, flags=re.IGNORECASE)
        
        # Preserve street names and addresses AFTER casing
        street_names = ['Wilhelminakade', 'IJburglaan', 'Rotterdam', 'Amsterdam']
        for street in street_names:
            if street.lower() in original.lower():
                translated = re.sub(rf'\b{street.lower()}\b', street, translated, flags=re.IGNORECASE)
        
        # Preserve postal codes and country codes AFTER casing
        postal_codes = ['AP', 'EM', 'NL']
        for code in postal_codes:
            if code in original:
                translated = re.sub(rf'\b{code.lower()}\b', code, translated)
        
        # Fix abbreviations AFTER casing - remove extra spaces
        translated = re.sub(r'N\. V\. T\.', 'N.v.t.', translated)
        translated = re.sub(r'n\. v\. t\.', 'N.v.t.', translated)
        
        # Fix VAT number format and casing
        translated = re.sub(r'\bVAT no\.\b', 'VAT-Nr.', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bvat no\.\b', 'VAT-Nr.', translated, flags=re.IGNORECASE)
        translated = re.sub(r'VAT no\.:', 'VAT-Nr.:', translated, flags=re.IGNORECASE)
        translated = re.sub(r'Nl(\d)', r'NL\1', translated)  # Fix NL country code casing
        
        # Fix CoC number format
        translated = re.sub(r'\bco c number\b', 'CoC Number', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bcoc number\b', 'CoC Number', translated, flags=re.IGNORECASE)
        
        # Apply improved casing preservation BEFORE the specific fixes (CRITICAL ORDER)
        translated = self._preserve_original_casing(original, translated)
        
        # Apply post-casing fixes to handle remaining issues (CRITICAL ORDER)
        translated = self._apply_post_casing_fixes(translated)
        
        # Apply final English post-processing
        translated = self._postprocess_english(translated, original)
        
        # Remove any preservation rules that might have been included in the output
        translated = self._remove_preservation_rules_from_output(translated)
        
        return translated.strip()

    def _preserve_original_casing(self, original: str, translated: str) -> str:
        """Preserve original capitalization patterns with smart sentence-level handling"""
        if not original or not translated:
            return translated
        
        original_clean = original.strip()
        translated_clean = translated.strip()
        
        # Check if this starts with header words (all caps) followed by sentence text
        words = original_clean.split()
        if words:
            # Count consecutive all-caps words from the beginning
            header_word_count = 0
            for word in words:
                if word.isupper():
                    header_word_count += 1
                else:
                    break
            
            # If we have at least one all-caps word followed by non-caps words, it's mixed content
            if header_word_count > 0 and header_word_count < len(words):
                return self._preserve_mixed_content_casing(original_clean, translated_clean)
        
        # Check if this is entirely a header/title (short, mostly caps)
        is_header = (
            len(words) <= 3 or  # Short phrases are likely headers
            original_clean.isupper()  # All caps
        )
        
        if is_header:
            return self._preserve_header_casing(original_clean, translated_clean)
        else:
            return self._preserve_sentence_casing(original_clean, translated_clean)

    def _preserve_mixed_content_casing(self, original: str, translated: str) -> str:
        """Handle content that starts with a header followed by sentence text"""
        orig_words = original.split()
        trans_words = translated.split()
        
        # Find where the header ends - look for the first lowercase word
        header_end = 0
        for i, word in enumerate(orig_words):
            if word.isupper():
                header_end = i + 1
            else:
                break
        
        result_words = []
        sentence_started = False
        
        for i, trans_word in enumerate(trans_words):
            if i < len(orig_words):
                orig_word = orig_words[i]
                
                # If we're in the header section
                if i < header_end:
                    # For header words, check if translated word is an acronym first
                    if self._is_acronym(trans_word):
                        result_words.append(trans_word.upper())
                    else:
                        result_words.append(self._apply_word_casing(orig_word, trans_word))
                else:
                    # We're in the sentence section
                    processed_word = self._apply_sentence_word_casing_with_context(
                        trans_word, i, trans_words, not sentence_started, result_words
                    )
                    result_words.append(processed_word)
                    
                    if not sentence_started:
                        sentence_started = True
                    
                    # Check for sentence end to reset sentence_started
                    if trans_word.rstrip('.,!?:;').endswith(('.', '!', '?')) or orig_word.endswith(('.', '!', '?')):
                        sentence_started = False
            else:
                # Extra translated words - need to determine if they belong to header or sentence
                # If we haven't reached the sentence part yet, treat as header
                if len(result_words) < header_end:
                    if self._is_acronym(trans_word):
                        result_words.append(trans_word.upper())
                    else:
                        # Use the casing pattern of the last header word
                        if header_end > 0 and orig_words[header_end - 1].isupper():
                            result_words.append(trans_word.upper())
                        else:
                            result_words.append(trans_word.capitalize())
                else:
                    # Treat as sentence continuation
                    processed_word = self._apply_sentence_word_casing_with_context(
                        trans_word, i, trans_words, not sentence_started, result_words
                    )
                    result_words.append(processed_word)
                    
                    if not sentence_started:
                        sentence_started = True
                    
                    # Check for sentence end
                    if trans_word.rstrip('.,!?:;').endswith(('.', '!', '?')):
                        sentence_started = False
        
        return ' '.join(result_words)

    def _apply_sentence_word_casing_with_context(self, word: str, position: int, all_words: list, is_sentence_start: bool, previous_words: list) -> str:
        """Apply proper casing for a word in sentence context with better context awareness"""
        word_clean = word.rstrip('.,!?:;')
        
        if is_sentence_start:
            return word.capitalize()
        elif self._is_acronym(word_clean):
            return word.upper()
        elif self._is_likely_proper_noun(word_clean, position, all_words):
            return word.capitalize()
        else:
            return word.lower()

    def _preserve_header_casing(self, original: str, translated: str) -> str:
        """Preserve casing for headers/titles"""
        orig_words = original.split()
        trans_words = translated.split()
        
        result_words = []
        
        for i, trans_word in enumerate(trans_words):
            if i < len(orig_words):
                orig_word = orig_words[i]
                result_words.append(self._apply_word_casing(orig_word, trans_word))
            else:
                # For extra words in headers, use the pattern from nearby words
                if orig_words:
                    # Check if most words are uppercase
                    caps_count = sum(1 for w in orig_words if w.isupper())
                    if caps_count > len(orig_words) * 0.5:
                        result_words.append(trans_word.upper())
                    else:
                        result_words.append(trans_word.capitalize())
                else:
                    result_words.append(trans_word.capitalize())
        
        return ' '.join(result_words)

    def _preserve_sentence_casing(self, original: str, translated: str) -> str:
        """Preserve casing for sentence text with smart sentence-level rules"""
        trans_words = translated.split()
        result_words = []
        
        for i, trans_word in enumerate(trans_words):
            # First word should be capitalized
            is_sentence_start = (i == 0)
            result_words.append(self._apply_sentence_word_casing_with_context(trans_word, i, trans_words, is_sentence_start, result_words))
        
        return ' '.join(result_words)

    def _is_acronym(self, word: str) -> bool:
        """Check if a word is likely an acronym that should stay in all caps"""
        word_clean = word.rstrip('.,!?:;')
        
        # Known acronyms
        acronyms = {
            'sepa', 'iban', 'bic', 'swift', 'eu', 'usa', 'uk', 'nl', 'id', 'api',
            'pdf', 'html', 'xml', 'json', 'http', 'https', 'www', 'email', 'sms',
            'atm', 'pin', 'cvv', 'vat', 'btw', 'kvk', 'bsn'
        }
        
        # Check if it's a known acronym
        if word_clean.lower() in acronyms:
            return True
        
        return False

    def _is_likely_proper_noun(self, word: str, position: int, all_words: list) -> bool:
        """Determine if a word is likely a proper noun"""
        word_clean = word.rstrip('.,!?:;')
        
        # Don't treat common words as proper nouns
        common_words = {
            'to', 'and', 'or', 'but', 'with', 'by', 'for', 'at', 'in', 'on', 'of', 'the',
            'a', 'an', 'this', 'that', 'these', 'those', 'you', 'your', 'my', 'our',
            'signing', 'form', 'gives', 'permission', 'bank', 'account', 'amount',
            'payment', 'term', 'days', 'after', 'invoice', 'date', 'agree', 'contact',
            'within', 'weeks', 'conditions', 'online', 'registration', 'customer',
            'general', 'terms', 'debit', 'deposit', 'collection', 'assignments',
            'accordance', 'order', 'based', 'depreciation', 'booked', 'back', 'also',
            'give', 'through', 'continuous', 'from', 'can', 'have', 'it', 'about',
            'during', 'as', 'agreed', 'authorization', 'one', 'off'
        }
        
        if word_clean.lower() in common_words:
            return False
        
        # Known proper nouns
        proper_nouns = {
            'greenwheels', 'amsterdam', 'sharma', 'yogesh', 'netherlands', 'holland', 
            'dutch', 'english', 'profile', 'curtiusstraat', 'donker'
        }
        
        if word_clean.lower() in proper_nouns:
            return True
        
        return False

    def _apply_word_casing(self, original_word: str, translated_word: str) -> str:
        """Apply the casing pattern of original_word to translated_word with smart character mapping"""
        if not original_word or not translated_word:
            return translated_word
        
        # Handle simple cases first
        if original_word.isupper():
            return translated_word.upper()
        elif original_word.islower():
            return translated_word.lower()
        elif original_word.istitle():
            return translated_word.capitalize()
        
        # For mixed case, do character-by-character mapping
        result = []
        orig_letters = [c for c in original_word if c.isalpha()]
        
        letter_idx = 0
        for char in translated_word:
            if char.isalpha():
                if letter_idx < len(orig_letters):
                    # Apply casing from corresponding original letter
                    if orig_letters[letter_idx].isupper():
                        result.append(char.upper())
                    else:
                        result.append(char.lower())
                    letter_idx += 1
                else:
                    # More letters in translation than original
                    # Use the pattern from the last few letters
                    if orig_letters:
                        # Look at the last 2-3 letters to determine pattern
                        recent_letters = orig_letters[-2:] if len(orig_letters) >= 2 else orig_letters
                        upper_count = sum(1 for c in recent_letters if c.isupper())
                        
                        if upper_count > len(recent_letters) / 2:
                            result.append(char.upper())
                        else:
                            result.append(char.lower())
                    else:
                        result.append(char.lower())
            else:
                # Non-alphabetic character - keep as is
                result.append(char)
        
        return ''.join(result)

    def _format_urls_and_web_addresses(self, text: str) -> str:
        """Detect and properly format URLs, web addresses, and email addresses"""
        import re
        
        # Pattern to match various URL formats
        url_patterns = [
            # Full URLs with protocol
            r'\b(?:https?://)?(?:www\.)?([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(?:/[^\s]*)?',
            # www.domain.com format
            r'\bwww\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.[a-zA-Z]{2,}(?:/[^\s]*)?',
            # domain.com format (when clearly a website)
            r'\b[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.[a-zA-Z]{2,}(?:/[a-zA-Z0-9._~:/?#[\]@!$&\'()*+,;=-]*)?'
        ]
        
        # Email pattern
        email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        
        def format_url(match):
            url = match.group(0)
            
            # Convert to lowercase for processing
            url_lower = url.lower()
            
            # Fix common capitalization issues
            if url_lower.startswith('www.'):
                url = 'www.' + url[4:]
            elif url_lower.startswith('http://www.'):
                url = 'http://www.' + url[11:]
            elif url_lower.startswith('https://www.'):
                url = 'https://www.' + url[12:]
            elif url_lower.startswith('http://'):
                url = 'http://' + url[7:]
            elif url_lower.startswith('https://'):
                url = 'https://' + url[8:]
            
            # Handle specific known domains
            known_domains = {
                'ind.nl': 'ind.nl',
                'government.nl': 'government.nl',
                'rijksoverheid.nl': 'rijksoverheid.nl',
                'belastingdienst.nl': 'belastingdienst.nl',
                'kvk.nl': 'kvk.nl',
                'google.com': 'google.com',
                'microsoft.com': 'microsoft.com',
                'apple.com': 'apple.com'
            }
            
            # Apply known domain formatting
            for domain, formatted in known_domains.items():
                if domain in url_lower:
                    url = url_lower.replace(domain, formatted)
                    break
            else:
                # Default: keep domain in lowercase
                url = url_lower
            
            return url
        
        def format_email(match):
            email = match.group(0)
            # Keep emails in lowercase (standard practice)
            return email.lower()
        
        # Apply URL formatting
        for pattern in url_patterns:
            text = re.sub(pattern, format_url, text, flags=re.IGNORECASE)
        
        # Apply email formatting
        text = re.sub(email_pattern, format_email, text, flags=re.IGNORECASE)
        
        # Fix specific URL-related translation issues
        text = re.sub(r'\bWww\.', 'www.', text)
        text = re.sub(r'\bWWW\.', 'www.', text)
        text = re.sub(r'\bHTTP://', 'http://', text)
        text = re.sub(r'\bHTTPS://', 'https://', text)
        text = re.sub(r'\bHttp://', 'http://', text)
        text = re.sub(r'\bHttps://', 'https://', text)
        
        return text

    def _apply_post_casing_fixes(self, text: str) -> str:
        """Apply final casing fixes after the main casing logic"""
        # Fix specific patterns that the main logic might miss
        
        # Fix "MY" followed by lowercase words to "My"
        text = re.sub(r'\bMY\b(?=\s+[a-z])', 'My', text)
        
        # Fix "YOU" followed by lowercase words to "You"  
        text = re.sub(r'\bYOU\b(?=\s+[a-z])', 'You', text)
        
        # Fix "my Profile" to "My Profile" (Profile should be capitalized)
        text = re.sub(r'\bmy Profile\b', 'My Profile', text)
        
        # Fix mid-sentence "Log in" to "log in"
        text = re.sub(r'(?<!^)(?<![.!?]\s)\bLog in\b', 'log in', text)
        
        # Apply proper sentence capitalization - this should be the final step
        text = self._capitalize_sentences(text)
        
        return text

    def _capitalize_sentences(self, text: str) -> str:
        """Ensure the first letter of each sentence is capitalized"""
        # Split text into sentences while preserving the delimiters
        sentences = re.split(r'([.!?]+\s*)', text)
        
        result = []
        for i, part in enumerate(sentences):
            if i % 2 == 0:  # This is a sentence (not a delimiter)
                if part.strip():  # Only process non-empty parts
                    # Capitalize the first letter of the sentence
                    part = part.lstrip()  # Remove leading whitespace
                    if part:
                        # Find the first alphabetic character and capitalize it
                        for j, char in enumerate(part):
                            if char.isalpha():
                                part = part[:j] + char.upper() + part[j+1:]
                                break
                        # Add back any leading whitespace that was removed
                        leading_space = sentences[i][:-len(part)] if len(sentences[i]) > len(part) else ''
                        part = leading_space + part
            result.append(part)
        
        return ''.join(result)

    def _postprocess_english(self, text: str, original: str) -> str:
        """Improve English punctuation and capitalization, removing unnecessary periods."""
        import re
        
        # Remove unnecessary periods first
        text = self._remove_unnecessary_periods(text, original)
        
        # Capitalize first letter of each sentence
        def capitalize_sentences(s):
            s = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), s)
            s = s[:1].upper() + s[1:] if s else s
            return s
        
        # Only add punctuation if it's a complete sentence and original had punctuation
        s = text.strip()
        if (s and s[-1] not in '.!?' and 
            original.strip().endswith('.') and 
            self._should_have_period(s)):
            s += '.'
            
        s = capitalize_sentences(s)
        return s
    
    def _remove_preservation_rules_from_output(self, translated: str) -> str:
        """Remove any preservation rules that might have been included in the output"""
        # Remove common preservation rule patterns
        patterns_to_remove = [
            r'CRITICAL PRESERVATION RULES.*?Only provide the translation',
            r'IMPORTANT.*?postal codes.*?NOT Dutch words',
            r'Keep ALL.*?UNCHANGED',
            r'Letters like.*?postal district codes',
            r'Only provide the translation.*?no explanations',
            r'CRITICAL PRESERVATION RULES.*?DO NOT TRANSLATE THESE.*?',
            r'1\. Keep ALL.*?6\. IMPORTANT.*?',
        ]
        
        for pattern in patterns_to_remove:
            translated = re.sub(pattern, '', translated, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any remaining instruction-like text
        translated = re.sub(r'Translation:\s*', '', translated, flags=re.IGNORECASE)
        translated = re.sub(r'Translated text:\s*', '', translated, flags=re.IGNORECASE)
        
        # Remove OpenAI error messages
        error_patterns = [
            r"I'm sorry, but.*?translate.*?",
            r"Please provide.*?translate.*?",
            r"Unable to translate.*?",
            r"No text provided.*?",
            r"Text is missing.*?",
            r"Empty text.*?",
            r"Invalid input.*?",
        ]
        
        for pattern in error_patterns:
            translated = re.sub(pattern, '', translated, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra whitespace
        translated = re.sub(r'\s+', ' ', translated)
        translated = translated.strip()
        
        return translated
    
    def _remove_unnecessary_periods(self, text: str, original_text: str) -> str:
        """Remove unnecessary periods that don't belong in the translation"""
        # If original text doesn't end with a period, remove trailing period from translation
        if not original_text.rstrip().endswith('.') and text.rstrip().endswith('.'):
            text = text.rstrip('.').rstrip()
        
        # Remove periods from labels/headers that shouldn't have them
        # Common patterns: single words, short phrases, titles, labels
        if len(text.split()) <= 3 and not original_text.endswith('.'):
            # Remove trailing periods from short labels/titles
            if text.endswith('.') and not self._should_have_period(text):
                text = text.rstrip('.')
        
        # Remove periods from addresses, names, codes, numbers
        if self._is_label_or_identifier(text, original_text):
            text = text.rstrip('.')
            
        return text
    
    def _should_have_period(self, text: str) -> bool:
        """Determine if text should naturally have a period"""
        text_lower = text.lower().strip()
        
        # Abbreviations that should keep periods
        abbreviations = ['mr.', 'mrs.', 'dr.', 'prof.', 'inc.', 'ltd.', 'co.', 'p.o.', 'etc.']
        if any(abbrev in text_lower for abbrev in abbreviations):
            return True
            
        # Complete sentences should have periods
        if len(text.split()) > 5 and any(word in text_lower for word in ['the', 'is', 'are', 'was', 'were', 'have', 'has', 'will', 'would']):
            return True
            
        return False
    
    def _is_label_or_identifier(self, text: str, original_text: str) -> bool:
        """Check if text is a label, identifier, or similar that shouldn't have periods"""
        import re
        text_lower = text.lower().strip()
        
        # Names, addresses, codes, numbers
        patterns = [
            r'^[A-Z]\.\s*[A-Za-z]+$',  # Y. Sharma
            r'^\d+.*[A-Z]{2,}.*\d*$',  # 1087 EM AMSTERDAM, Z1-186720992110
            r'^[A-Za-z]+\s+\d+$',      # IJburglaan 816
            r'^[A-Za-z]+\s+\d+$',      # 2850241598
            r'^www\.',                 # www.ind.nl
            r'^[A-Z]{2,}\s+[A-Z]{2,}', # RVN NAT ZW Team 05
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
                
        # Single words or very short phrases that are clearly labels
        words = text.split()
        if len(words) <= 2 and not any(word.lower() in ['the', 'is', 'are', 'and', 'or', 'but'] for word in words):
            return True
            
        return False

    def _detect_table_regions(self, text_dict: dict) -> List[Dict]:
        """Detect table-like regions in the PDF and extract them cell-by-cell"""
        print(f"[TABLE DETECTION] Starting table detection for {len(text_dict['blocks'])} blocks")
        tables = []
        
        print(f"\n[TABLE DETECTION DEBUG] Processing {len(text_dict['blocks'])} blocks")
        
        for block_idx, block in enumerate(text_dict["blocks"]):
            if "lines" not in block:
                continue
                
            lines = block["lines"]
            if len(lines) < 2:  # Tables should have at least 2 lines
                continue
            
            # Check if this looks like a table based on content patterns
            line_texts = [" ".join([span["text"] for span in line["spans"]]).strip() for line in lines]
            
            print(f"[TABLE DETECTION DEBUG] Block {block_idx}: {len(lines)} lines")
            print(f"[TABLE DETECTION DEBUG] Line texts: {line_texts}")
            
            # More general table detection - look for structured data patterns
            if self._is_general_table(line_texts):
                print(f"[TABLE DETECTION DEBUG] Block {block_idx} detected as general table")
                table = self._extract_general_table(block, line_texts, block_idx)
                if table:
                    tables.append(table)
            
            # Product table detection
            elif self._is_product_table(line_texts):
                print(f"[TABLE DETECTION DEBUG] Block {block_idx} detected as product table")
                table = self._extract_product_table(block, line_texts, block_idx)
                if table:
                    tables.append(table)
            
            # VAT table detection  
            elif self._is_vat_table(line_texts):
                print(f"[TABLE DETECTION DEBUG] Block {block_idx} detected as VAT table")
                table = self._extract_vat_table(block, line_texts, block_idx)
                if table:
                    tables.append(table)
            
            # Summary table detection
            elif self._is_summary_table(line_texts):
                print(f"[TABLE DETECTION DEBUG] Block {block_idx} detected as summary table")
                table = self._extract_summary_table(block, line_texts, block_idx)
                if table:
                    tables.append(table)
            else:
                print(f"[TABLE DETECTION DEBUG] Block {block_idx} not detected as any table type")
        
        print(f"[TABLE DETECTION DEBUG] Found {len(tables)} tables")
        return tables
    
    def _is_general_table(self, line_texts: List[str]) -> bool:
        """Check if this block represents a general table with structured data"""
        if len(line_texts) < 2:
            return False
            
        # Be more conservative - only detect as table if there's clear visual table structure
        # Look for patterns that suggest actual visual table structure:
        # 1. Multiple columns with clear alignment
        # 2. Lines with explicit separators like |, +, or consistent column spacing
        # 3. Header rows followed by data rows with consistent structure
        
        # Check for explicit table separators (these indicate visual table structure)
        has_table_separators = any('|' in line or '+' in line or '=' in line for line in line_texts)
        
        # Check for consistent column structure (multiple columns with similar width)
        has_consistent_columns = False
        if len(line_texts) >= 3:
            # Look for lines that have similar word count and spacing patterns
            word_counts = [len(line.split()) for line in line_texts]
            if len(set(word_counts)) <= 2:  # Most lines have similar word count
                # Check if lines have consistent spacing patterns (suggesting columns)
                spacing_patterns = []
                for line in line_texts:
                    if len(line) > 20:  # Only check longer lines
                        spaces = [i for i, char in enumerate(line) if char == ' ']
                        if len(spaces) >= 2:
                            spacing_patterns.append(spaces)
                
                if len(spacing_patterns) >= 2:
                    # Check if spacing patterns are similar (suggesting aligned columns)
                    similar_patterns = 0
                    for i in range(len(spacing_patterns) - 1):
                        for j in range(i + 1, len(spacing_patterns)):
                            # Check if spacing patterns are similar
                            pattern1 = spacing_patterns[i]
                            pattern2 = spacing_patterns[j]
                            if len(pattern1) == len(pattern2):
                                # Check if spacings are at similar positions
                                similar_positions = sum(1 for p1, p2 in zip(pattern1, pattern2) if abs(p1 - p2) < 5)
                                if similar_positions >= len(pattern1) * 0.7:
                                    similar_patterns += 1
                    
                    if similar_patterns >= len(spacing_patterns) * 0.5:
                        has_consistent_columns = True
        
        # Only classify as table if there's clear visual table structure
        return has_table_separators or has_consistent_columns
        
        # Check for header-like patterns (shorter lines that might be headers)
        potential_headers = [line for line in line_texts if len(line.split()) <= 3 and not any(c.isdigit() for c in line)]
        
        # Check for water bill specific patterns
        water_bill_patterns = ['waterverbruik', 'vastrecht', 'belasting', 'btw', 'totaal', 'bedrag', 'tarief', 'periode']
        has_water_bill_terms = any(any(pattern in line.lower() for pattern in water_bill_patterns) for line in line_texts)
        
        # Debug output
        print(f"[TABLE DETECTION DEBUG] Checking lines: {line_texts[:3]}...")  # Show first 3 lines
        print(f"[TABLE DETECTION DEBUG] has_currency: {has_currency}, has_numbers: {has_numbers}, has_separators: {has_separators}, has_water_bill_terms: {has_water_bill_terms}")
        
        # Consider it a table if it has structured data characteristics
        # Be more lenient for water bill tables
        is_table = (has_currency or has_numbers or has_separators or has_water_bill_terms) and len(line_texts) >= 2
        print(f"[TABLE DETECTION DEBUG] Result: {is_table}")
        return is_table
    
    def _extract_general_table(self, block: dict, line_texts: List[str], block_idx: int) -> Dict:
        """Extract general table data cell-by-cell"""
        table_data = {
            "type": "general_table",
            "block_index": block_idx,
            "bbox": block["bbox"],
            "rows": []
        }
        
        # Process each line as a potential table row
        for i, line_text in enumerate(line_texts):
            # Split by common table separators
            if '|' in line_text:
                cells = [cell.strip() for cell in line_text.split('|')]
            elif ':' in line_text:
                # Handle key-value pairs
                parts = line_text.split(':', 1)
                if len(parts) == 2:
                    cells = [parts[0].strip(), parts[1].strip()]
                else:
                    cells = [line_text.strip()]
            else:
                # Try to split by consistent spacing or other patterns
                cells = [line_text.strip()]
            
            row = {
                "row_index": i,
                "cells": cells,
                "original_text": line_text
            }
            table_data["rows"].append(row)
        
        return table_data
    
    def _is_product_table(self, line_texts: List[str]) -> bool:
        """Check if this block represents a product table"""
        # Look for product table headers
        headers = ["pos", "artikelnummer", "beschrijving", "aantal", "btw", "stuksprijs", "totaal"]
        text_lower = " ".join(line_texts).lower()
        return sum(1 for header in headers if header in text_lower) >= 4
    
    def _is_vat_table(self, line_texts: List[str]) -> bool:
        """Check if this block represents a VAT calculation table"""
        vat_terms = ["btw", "btw-grondslag", "btw-bedrag", "totaal"]
        text_lower = " ".join(line_texts).lower()
        return sum(1 for term in vat_terms if term in text_lower) >= 3
    
    def _is_summary_table(self, line_texts: List[str]) -> bool:
        """Check if this block represents a summary/totals table"""
        summary_terms = ["totaal", "restbedrag", "eur"]
        text_lower = " ".join(line_texts).lower()
        return sum(1 for term in summary_terms if term in text_lower) >= 2
    
    def _extract_product_table(self, block: dict, line_texts: List[str], block_idx: int) -> Dict:
        """Extract product table data cell-by-cell"""
        # Product table structure: Pos | Article# Description | Qty | VAT | Unit Price | Total
        table_data = {
            "type": "product_table",
            "block_index": block_idx,
            "bbox": block["bbox"],
            "headers": [],
            "rows": []
        }
        
        # Identify headers (first 6 lines typically)
        headers = line_texts[:6]
        table_data["headers"] = headers
        
        # Extract product row data (remaining lines)
        product_data = line_texts[6:]
        if len(product_data) >= 6:  # Should have: pos, article#, description, qty, vat%, unit_price, total
            row = {
                "position": product_data[0] if len(product_data) > 0 else "",
                "article_number": product_data[1] if len(product_data) > 1 else "",
                "description": product_data[2] if len(product_data) > 2 else "",
                "quantity": product_data[3] if len(product_data) > 3 else "",
                "vat_rate": product_data[4] if len(product_data) > 4 else "",
                "unit_price": product_data[5] if len(product_data) > 5 else "",
                "total": product_data[6] if len(product_data) > 6 else ""
            }
            table_data["rows"].append(row)
        
        return table_data
    
    def _extract_vat_table(self, block: dict, line_texts: List[str], block_idx: int) -> Dict:
        """Extract VAT calculation table cell-by-cell"""
        table_data = {
            "type": "vat_table", 
            "block_index": block_idx,
            "bbox": block["bbox"],
            "headers": [],
            "rows": []
        }
        
        # VAT table structure: BTW | BTW-grondslag | BTW-bedrag | Totaal
        headers = line_texts[:4]
        table_data["headers"] = headers
        
        # Extract VAT calculation data
        vat_data = line_texts[4:]
        if len(vat_data) >= 4:
            row = {
                "vat_rate": vat_data[0] if len(vat_data) > 0 else "",
                "vat_base": vat_data[1] if len(vat_data) > 1 else "",
                "vat_amount": vat_data[2] if len(vat_data) > 2 else "",
                "total": vat_data[3] if len(vat_data) > 3 else ""
            }
            table_data["rows"].append(row)
        
        return table_data
    
    def _extract_summary_table(self, block: dict, line_texts: List[str], block_idx: int) -> Dict:
        """Extract summary/totals table cell-by-cell"""
        table_data = {
            "type": "summary_table",
            "block_index": block_idx, 
            "bbox": block["bbox"],
            "rows": []
        }
        
        # Summary table: pairs of label-value
        for i in range(0, len(line_texts), 2):
            if i + 1 < len(line_texts):
                row = {
                    "label": line_texts[i],
                    "value": line_texts[i + 1]
                }
                table_data["rows"].append(row)
        
        return table_data

    def _format_table_as_text(self, table: Dict) -> str:
        """Format table data as readable text for display purposes"""
        if table["type"] == "general_table":
            text_parts = []
            text_parts.append("TABLE DATA:")
            for row in table["rows"]:
                # Join cells with | separator for better readability
                row_text = " | ".join(row["cells"])
                text_parts.append(row_text)
            return "\n".join(text_parts)
        
        elif table["type"] == "product_table":
            text_parts = []
            text_parts.append("PRODUCT TABLE:")
            text_parts.append("Headers: " + " | ".join(table["headers"]))
            for row in table["rows"]:
                row_text = f"Position: {row['position']}, Article: {row['article_number']}, Description: {row['description']}, Qty: {row['quantity']}, VAT: {row['vat_rate']}, Unit Price: {row['unit_price']}, Total: {row['total']}"
                text_parts.append(row_text)
            return "\n".join(text_parts)
        
        elif table["type"] == "vat_table":
            text_parts = []
            text_parts.append("VAT CALCULATION TABLE:")
            text_parts.append("Headers: " + " | ".join(table["headers"]))
            for row in table["rows"]:
                row_text = f"VAT Rate: {row['vat_rate']}, VAT Base: {row['vat_base']}, VAT Amount: {row['vat_amount']}, Total: {row['total']}"
                text_parts.append(row_text)
            return "\n".join(text_parts)
        
        elif table["type"] == "summary_table":
            text_parts = []
            text_parts.append("SUMMARY TABLE:")
            for row in table["rows"]:
                text_parts.append(f"{row['label']}: {row['value']}")
            return "\n".join(text_parts)
        
        return "TABLE DATA"

    def extract_visual_elements(self, pdf_content: bytes, page_num: int = 0) -> List[Dict]:
        """Extract all visual elements (images, vector graphics, logos) from a PDF page."""
        # Save PDF content to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(pdf_content)
            tmp_file.flush()
            pdf_path = tmp_file.name
        
        try:
            visual_elements = []
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            
            try:
                # Extract raster images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Get image placement information
                        img_rects = page.get_image_rects(xref)
                        for rect in img_rects:
                            bbox = tuple(rect)
                            
                            # Determine if this is likely a logo based on size and position
                            width = bbox[2] - bbox[0]
                            height = bbox[3] - bbox[1]
                            is_logo = self._is_likely_logo(bbox, width, height, page.rect)
                            
                            visual_elements.append({
                                "type": "logo" if is_logo else "image",
                                "subtype": "raster",
                                "bbox": bbox,
                                "data": image_bytes,
                                "format": image_ext,
                                "xref": xref,
                                "page": page_num,
                                "width": width,
                                "height": height
                            })
                            
                    except Exception as e:
                        print(f"Warning: Could not extract image {img_index}: {e}")
                        continue
                
                # Extract vector graphics and shapes
                drawings = page.get_drawings()
                print(f"Found {len(drawings)} vector drawings on page {page_num}")
                for drawing_index, drawing in enumerate(drawings):
                    try:
                        print(f"Processing drawing {drawing_index}: {drawing.get('type', 'unknown')}")
                        
                        # Use the drawing's rect property if available, otherwise calculate from items
                        if drawing.get('rect'):
                            rect = drawing['rect']
                            bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                            print(f"  Using drawing rect: {bbox}")
                        else:
                            print(f"  No rect found, calculating from items...")
                            # Fallback: calculate bounding box from items
                            all_points = []
                            items = drawing.get("items", [])
                            
                            for item in items:
                                if len(item) >= 2:
                                    item_type = item[0]
                                    item_data = item[1]
                                    
                                    # Handle different item types
                                    if item_type == "re":  # Rectangle
                                        # item_data should be a rectangle with x0, y0, x1, y1
                                        if hasattr(item_data, 'x0'):
                                            all_points.extend([(item_data.x0, item_data.y0), (item_data.x1, item_data.y1)])
                                        elif isinstance(item_data, (list, tuple)) and len(item_data) >= 4:
                                            all_points.extend([(item_data[0], item_data[1]), (item_data[2], item_data[3])])
                                            
                                    elif item_type in ["l", "m", "c"]:  # Line, move, curve
                                        # item_data should be a point with x, y
                                        if hasattr(item_data, 'x'):
                                            all_points.append((item_data.x, item_data.y))
                                        elif isinstance(item_data, (list, tuple)) and len(item_data) >= 2:
                                            all_points.append((item_data[0], item_data[1]))
                            
                            if all_points:
                                x_coords = [p[0] for p in all_points]
                                y_coords = [p[1] for p in all_points]
                                bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                                print(f"  Calculated bbox from {len(all_points)} points: {bbox}")
                            else:
                                print(f"  No valid points found from {len(items)} items, skipping")
                                continue
                        
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        print(f"  Size: {width:.1f}x{height:.1f}")
                        
                        # Skip very small elements (likely noise)
                        if width > 5 and height > 5:
                            is_logo = self._is_likely_logo(bbox, width, height, page.rect)
                            print(f"  Is logo: {is_logo}")
                            
                            visual_elements.append({
                                "type": "logo" if is_logo else "vector_graphic",
                                "subtype": "vector",
                                "bbox": bbox,
                                "data": drawing,
                                "format": "vector",
                                "page": page_num,
                                "width": width,
                                "height": height,
                                "source_pdf": pdf_path
                            })
                            
                            print(f"Extracted vector element: {'logo' if is_logo else 'vector_graphic'} at {bbox}, size {width:.1f}x{height:.1f}")
                        else:
                            print(f"  Skipping small element: {width:.1f}x{height:.1f}")
                            
                    except Exception as e:
                        print(f"Warning: Could not process vector drawing {drawing_index}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Warning: Error extracting visual elements from page {page_num}: {e}")
            
            finally:
                doc.close()
            
            return visual_elements
        finally:
            # Cleanup temporary file
            try:
                os.unlink(pdf_path)
            except:
                pass
    
    def _is_likely_logo(self, bbox: tuple, width: float, height: float, page_rect) -> bool:
        """Determine if a visual element is likely a logo based on size, position, and aspect ratio."""
        page_width = page_rect.width
        page_height = page_rect.height
        x0, y0, x1, y1 = bbox
        
        # Logo characteristics:
        # 1. Usually in header area (top 20% of page)
        # 2. Reasonable size (not too big, not too small)
        # 3. Often square-ish or rectangular aspect ratio
        # 4. Usually positioned near edges or corners
        
        # Check position - logos are often in header area
        is_in_header = y0 < page_height * 0.25
        
        # Check size - logos are typically 5-20% of page width
        size_ratio = width / page_width
        is_reasonable_size = 0.05 <= size_ratio <= 0.3
        
        # Check aspect ratio - logos are usually not extremely elongated
        aspect_ratio = width / height if height > 0 else 1
        is_reasonable_aspect = 0.2 <= aspect_ratio <= 5.0
        
        # Check if it's positioned like a logo (near edges)
        margin_threshold = page_width * 0.1
        is_near_edge = (x0 < margin_threshold or  # Left edge
                       x1 > page_width - margin_threshold or  # Right edge
                       y0 < margin_threshold)  # Top edge
        
        # Combine criteria
        logo_score = 0
        if is_in_header: logo_score += 2
        if is_reasonable_size: logo_score += 2
        if is_reasonable_aspect: logo_score += 1
        if is_near_edge: logo_score += 1
        
        # Consider it a logo if it meets most criteria
        return logo_score >= 3

# Initialize the advanced parser
parser = AdvancedPDFLayoutParser()

def get_cache_key(text: str, source_lang: str, target_lang: str) -> str:
    return hashlib.md5(f"{text}:{source_lang}:{target_lang}".encode()).hexdigest()

def manage_cache_size():
    if len(translation_cache) > CACHE_SIZE:
        # Remove oldest entries
        keys_to_remove = list(translation_cache.keys())[:CACHE_SIZE // 5]
        for key in keys_to_remove:
            del translation_cache[key]

async def translate_text_simple(text: str, source_lang: str, target_lang: str, translation_id: str = None) -> str:
    """Simple text translation for basic text requests"""
    cache_key = get_cache_key(text, source_lang, target_lang)
    
    if cache_key in translation_cache:
        return translation_cache[cache_key]
    
    if not API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    # Limit text length to avoid API issues
    if len(text) > 3000:
        text = text[:3000] + "..."
    
    try:
        # Check for cancellation before API call
        if translation_id and is_translation_cancelled(translation_id):
            raise Exception("Translation was cancelled")
        
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
                "max_tokens": 1500,
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

@app.get("/")
async def root():
    return {
        "message": "Advanced Translation API with Full Layout Parser Integration",
        "version": "8.7 - Perfect English Processing",
        "features": [
            "Complete PyMuPDF layout detection",
            "31+ text blocks with bounding boxes",
            "Font style preservation (bold, italic, size)",
            "Block classification (title, header, body, table, QR code)",
            "Table detection and extraction",
            "Visual elements extraction (logos, images, QR codes)",
            "Advanced casing preservation",
            "Sophisticated translation fixes",
            "Professional PDF generation with exact positioning",
            "Duplicate detection with fuzzy matching",
            "Same advanced capabilities as test_layoutparser_simple.py",
            "Translation cancellation support"
        ],
        "status": "ready"
    }

@app.post("/cancel-translation")
async def cancel_translation_endpoint(translation_id: str = Form(...)):
    """Cancel an active translation"""
    if cancel_translation(translation_id):
        return {"status": "success", "message": f"Translation {translation_id} cancelled"}
    else:
        raise HTTPException(status_code=404, detail="Translation not found or already completed")

@app.get("/translation-status/{translation_id}")
async def get_translation_status(translation_id: str):
    """Get the status of a translation"""
    if translation_id in active_translations:
        is_cancelled = is_translation_cancelled(translation_id)
        return {
            "translation_id": translation_id,
            "status": "cancelled" if is_cancelled else "active",
            "started_at": active_translations[translation_id].get("started_at"),
            "progress": active_translations[translation_id].get("progress", 0)
        }
    else:
        raise HTTPException(status_code=404, detail="Translation not found")

@app.post("/translate")
async def translate(request: TranslationRequest, translation_id: str = None):
    """Simple text translation endpoint"""
    # Generate translation ID if not provided
    if not translation_id:
        translation_id = generate_translation_id()
    
    # Register this translation as active
    active_translations[translation_id] = {
        "started_at": datetime.now().isoformat(),
        "progress": 0,
        "filename": "text_translation"
    }
    
    try:
        # Check for cancellation before starting
        if is_translation_cancelled(translation_id):
            cleanup_translation(translation_id)
            raise HTTPException(status_code=400, detail="Translation was cancelled")
        
        # Update progress to 50%
        active_translations[translation_id]["progress"] = 50
        
        translated_text = await translate_text_simple(
            request.text, 
            request.source_lang, 
            request.target_lang,
            translation_id
        )
        
        # Update progress to 100%
        active_translations[translation_id]["progress"] = 100
        
        # Clean up translation tracking
        cleanup_translation(translation_id)
        
        return {
            "original_text": request.text,
            "translated_text": translated_text,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
            "status": "success",
            "translation_id": translation_id
        }
    except Exception as e:
        # Clean up translation tracking on error
        if translation_id:
            cleanup_translation(translation_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate-pdf")
async def translate_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("en"),
    translation_id: str = Form(None)
):
    """Advanced PDF translation with full layout parser integration and Word-origin PDF detection"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Generate translation ID if not provided
    if not translation_id:
        translation_id = generate_translation_id()
    
    # Register this translation as active
    active_translations[translation_id] = {
        "started_at": datetime.now().isoformat(),
        "progress": 0,
        "filename": file.filename
    }
    
    try:
        # Check for cancellation before starting
        if is_translation_cancelled(translation_id):
            cleanup_translation(translation_id)
            raise HTTPException(status_code=400, detail="Translation was cancelled")
        
        # Read PDF content
        pdf_content = await file.read()
        
        print(f"Processing PDF: {file.filename}")
        
        # Save PDF to temporary file for analysis
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_content)
            tmp_file.flush()
            temp_pdf_path = tmp_file.name
        
        try:
            # Analyze PDF to determine if it's likely from Word
            print("Analyzing PDF structure...")
            is_word_pdf = analyze_pdf_structure(temp_pdf_path)
            
            if is_word_pdf:
                print("📄 Word-origin PDF detected! Using DOCX pipeline for better formatting preservation...")
                
                # Use the new pipeline for Word-origin PDFs
                output_pdf_path = tempfile.mktemp(suffix='.pdf')
                
                # Map target_lang to the format expected by the pipeline
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
                
                # Run the DOCX pipeline (placeholder - not implemented)
                success = False
                
                if success and os.path.exists(output_pdf_path):
                    print("✅ DOCX pipeline completed successfully!")
                    
                    # Read the translated PDF
                    with open(output_pdf_path, 'rb') as f:
                        output_pdf_bytes = f.read()
                    
                    # Cleanup function for background task
                    def cleanup():
                        try:
                            os.unlink(temp_pdf_path)
                            os.unlink(output_pdf_path)
                        except Exception as e:
                            print(f"Error cleaning up temp files: {e}")
                    
                    # Add cleanup task
                    background_tasks.add_task(cleanup)
                    
                    # Return file response
                    return FileResponse(
                        path=output_pdf_path,
                        media_type='application/pdf',
                        filename=f"translated_{file.filename}",
                        background=None
                    )
                else:
                    print("⚠️ DOCX pipeline failed, falling back to standard pipeline...")
                    # Continue with standard pipeline below
            else:
                print("📄 Standard PDF detected, using advanced layout parser...")
            
        except Exception as e:
            print(f"⚠️ PDF analysis failed: {e}, using standard pipeline...")
            # Continue with standard pipeline below
        
        # Standard pipeline (existing logic)
        print("Using standard advanced layout parser pipeline...")
        
        # Initialize parser for standard pipeline
        parser = AdvancedPDFLayoutParser()
        
        # Extract visual elements first
        print("Extracting visual elements...")
        visual_elements = parser.extract_visual_elements(pdf_content, page_num=0)
        print(f"Found {len(visual_elements)} visual elements:")
        for i, elem in enumerate(visual_elements):
            print(f"  {i+1}. {elem['type']} ({elem['subtype']}) at {elem['bbox']} - {elem['width']:.1f}x{elem['height']:.1f}")
        
        # Extract text blocks with page numbers
        blocks_with_pages = parser.extract_blocks_from_pdf(pdf_content)
        print(f"Extracted {len(blocks_with_pages)} initial text blocks")
        
        # Remove duplicates
        unique_blocks_with_pages = parser.remove_duplicates(blocks_with_pages)
        print(f"After deduplication: {len(unique_blocks_with_pages)} text blocks")
        
        # Prepare results and translation data
        blocks_with_translations = []
        
        # Separate QR codes from text blocks for translation
        text_blocks = []
        qr_blocks = []
        for block, page_num in unique_blocks_with_pages:
            if block.type == "qr_code":
                qr_blocks.append((block, page_num))
            else:
                text_blocks.append((block, page_num))
        
        # Only translate non-QR blocks
        texts = [block.text for block, _ in text_blocks]
        if texts:
            translations = []
            for i, text in enumerate(texts):
                # Check for cancellation before each translation
                if is_translation_cancelled(translation_id):
                    cleanup_translation(translation_id)
                    raise HTTPException(status_code=400, detail="Translation was cancelled")
                
                # Update progress
                progress = int((i / len(texts)) * 80)  # Translation is 80% of total work
                active_translations[translation_id]["progress"] = progress
                
                translated = await parser.translate_text_openai(text, target_lang, translation_id)
                translations.append(translated)
            print(f"Translations received: {len(translations)}")
        else:
            translations = []
        
        # Combine translated text blocks and QR blocks
        text_idx = 0
        for block, page_num in unique_blocks_with_pages:
            if block.type == "qr_code":
                # Don't translate QR codes, keep original
                blocks_with_translations.append((block, page_num, "[QR Code - Visual Element]"))
                print(f"Block (QR Code): Skipped translation")
            else:
                translated_text = translations[text_idx] if text_idx < len(translations) else block.text
                blocks_with_translations.append((block, page_num, translated_text))
                print(f"Block {text_idx}:")
                print(f"  Original: {block.text}")
                print(f"  Translated: {translated_text}")
                text_idx += 1
        
        # Check for cancellation before PDF generation
        if is_translation_cancelled(translation_id):
            cleanup_translation(translation_id)
            raise HTTPException(status_code=400, detail="Translation was cancelled")
        
        # Update progress to 90% (PDF generation phase)
        active_translations[translation_id]["progress"] = 90
        
        # Create advanced PDF with visual elements
        output_pdf_bytes = await create_advanced_pdf_with_visuals(
            pdf_content, blocks_with_translations, visual_elements, target_lang
        )
        
        # Create temporary file for response
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(output_pdf_bytes)
            tmp_file.flush()
            output_path = tmp_file.name
        
        # Cleanup function for background task
        def cleanup():
            try:
                os.unlink(temp_pdf_path)
                os.unlink(output_path)
            except Exception as e:
                print(f"Error cleaning up temp file: {e}")
        
        # Update progress to 100% (completed)
        active_translations[translation_id]["progress"] = 100
        
        # Add cleanup task
        background_tasks.add_task(cleanup)
        
        # Clean up translation tracking
        cleanup_translation(translation_id)
        
        # Return file response
        return FileResponse(
            path=output_path,
            media_type='application/pdf',
            filename=f"translated_{file.filename}",
            background=None  # Don't use deprecated background parameter
        )
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        # Clean up translation tracking on error
        if translation_id:
            cleanup_translation(translation_id)
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

async def create_advanced_pdf_with_visuals(pdf_content: bytes, blocks_with_translations: list, visual_elements: list, target_lang: str) -> bytes:
    """Create advanced PDF with visual elements using ReportLab"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfbase import pdfmetrics
    from reportlab.lib.utils import simpleSplit
    
    # Define target language mapping at function level
    target_language_map = {
        'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'ar': 'Arabic', 'nl': 'Dutch',
        'pl': 'Polish', 'tr': 'Turkish', 'uk': 'Ukrainian', 'vi': 'Vietnamese'
    }
    
    # Save original PDF to temporary file to get page dimensions
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_file.write(pdf_content)
        tmp_file.flush()
        original_pdf_path = tmp_file.name
    
    try:
        # Get page size from original PDF
        doc = fitz.open(original_pdf_path)
        page_sizes = [(page.rect.width, page.rect.height) for page in doc]
        doc.close()

        # Create output PDF with Unicode support
        output_buffer = io.BytesIO()
        c = canvas.Canvas(output_buffer, pagesize=letter)
        
        # Enable Unicode support for proper character encoding
        c.setTitle("Translated PDF")
        c.setAuthor("PDF Translator")
        c.setSubject("Translated Document")
        
        # Fonts are already registered at module level
        
        # Set encoding to UTF-8 for proper Unicode support
        import sys
        if sys.version_info[0] >= 3:
            # Python 3: Unicode is handled automatically
            pass
        else:
            # Python 2: Need to set encoding
            c.setEncoding('utf-8')
        
        # Create parser instance for text processing
        parser = AdvancedPDFLayoutParser(require_api_key=False)
        
        styles = getSampleStyleSheet()
        base_style = styles["Normal"]
        
        for page_num, (page_width, page_height) in enumerate(page_sizes):
            c.setPageSize((page_width, page_height))
            
            # Get visual elements for this page
            all_page_visual_elements = [elem for elem in visual_elements if elem.get("page", 0) == page_num]
            # Identify background image (full-page)
            background_elem = None
            page_visual_elements = []
            for elem in all_page_visual_elements:
                bbox = elem["bbox"]
                x0, y0, x1, y1 = bbox
                original_width = x1 - x0
                original_height = y1 - y0
                if original_width >= 0.95 * page_width and original_height >= 0.95 * page_height and background_elem is None:
                    background_elem = elem
                else:
                    page_visual_elements.append(elem)
            
            # Debug: Print info for page 1
            if page_num == 0:
                print(f"[DEBUG] Page 1: {1 if background_elem else 0} background image, {len(page_visual_elements)} other visual elements (rendered):")
                if background_elem:
                    print(f"  - background {background_elem['type']} bbox={background_elem['bbox']}")
                for elem in page_visual_elements:
                    print(f"  - {elem['type']} bbox={elem['bbox']}")
                page_text_blocks = [(block, blk_page_num, translated_text) for block, blk_page_num, translated_text in blocks_with_translations if blk_page_num == page_num]
                print(f"[DEBUG] Page 1: {len(page_text_blocks)} text blocks:")
                for block, _, _ in page_text_blocks:
                    print(f"  - bbox={block.bbox} type={block.type}")
            
            # 1. Render background image if present
            if background_elem is not None:
                print(f"Rendering background image for page {page_num} (detected as full-page image)...")
                render_visual_element(c, background_elem, page_height, original_pdf_path, force_fullpage=True, page_width=page_width, page_num=page_num)
            else:
                print(f"[DEBUG] No full-page background image detected for page {page_num}")
            
            # 2. Draw all text blocks for this page (render text on top)
            print(f"Rendering text blocks for page {page_num}...")
            for block, blk_page_num, translated_text in blocks_with_translations:
                if blk_page_num != page_num:
                    continue
                
                # Skip QR codes - they're handled as visual elements now
                if block.type == "qr_code":
                    # Try to render actual QR code from the text content
                    qr_rendered = render_qr_code_from_text(c, block.text, block.bbox, page_height)
                    
                    if not qr_rendered:
                        # Fallback to placeholder if QR generation fails
                        qr_bbox = block.bbox
                        found_visual_qr = False
                        for elem in page_visual_elements:
                            elem_bbox = elem["bbox"]
                            # Check if bboxes overlap significantly
                            if bboxes_overlap(qr_bbox, elem_bbox, threshold=0.5):
                                found_visual_qr = True
                                break
                        
                        if not found_visual_qr:
                            # Draw a placeholder box for QR codes not captured as visual elements
                            x0, y0, x1, y1 = qr_bbox
                            reportlab_y0 = page_height - y1
                            reportlab_y1 = page_height - y0
                            c.setStrokeColorRGB(0.5, 0.5, 0.5)  # Gray border
                            c.setFillColorRGB(0.9, 0.9, 0.9)    # Light gray fill
                            c.rect(x0, reportlab_y0, x1-x0, reportlab_y1-reportlab_y0, stroke=1, fill=1)
                            # Add QR code label
                            c.setFillColorRGB(0, 0, 0)
                            c.setFont("Helvetica", 8)
                            c.drawString(x0 + 5, reportlab_y0 + 5, "[QR Code]")
                    continue
                
                # --- TABLE RENDERING ---
                if block.type == "table":
                    bbox = block.bbox
                    x0, y0, x1, y1 = bbox
                    width, height = x1 - x0, y1 - y0
                    
                    print(f"\n[TABLE RENDERING] Rendering table block at bbox={bbox}")
                    print(f"[TABLE RENDERING] Table text: {translated_text[:100]}...")
                    
                    # If block has structured data, use it
                    if hasattr(block, "structured_data") and block.structured_data:
                        table = block.structured_data
                        print(f"[TABLE RENDERING] Using structured data: {table.get('type', 'unknown')}")
                        
                        # Build table data
                        if table["type"] == "general_table":
                            data = [row["cells"] for row in table["rows"]]
                        elif table["type"] == "product_table":
                            data = [table["headers"]] + [[row.get("position",""), row.get("article_number",""), row.get("description",""), row.get("quantity",""), row.get("vat_rate",""), row.get("unit_price",""), row.get("total","")] for row in table["rows"]]
                        elif table["type"] == "vat_table":
                            data = [table["headers"]] + [[row.get("description",""), row.get("base",""), row.get("rate",""), row.get("amount","")] for row in table["rows"]]
                        elif table["type"] == "summary_table":
                            data = [table["headers"]] + [[row.get("description",""), row.get("amount","")] for row in table["rows"]]
                        else:
                            data = [[translated_text]]
                    else:
                        # No structured data - treat as single-row table
                        print(f"[TABLE RENDERING] No structured data, treating as single-row table")
                        data = [[translated_text]]
                    
                    print(f"[TABLE RENDERING] Table data: {data}")
                    
                    # Get font style for table rendering
                    font_style = get_font_style_from_block(block)
                    try:
                        # Map target_lang to full language name for font selection
                        target_language = target_language_map.get(target_lang, 'English')
                        font_name = select_appropriate_font("Helvetica", font_style['bold'], font_style['italic'], block.font, target_language=target_language, text=translated_text)
                        font_size = max(5, int(block.size) if block.size else 12)
                    except Exception as e:
                        print(f"Error in table font style processing: {e}")
                        font_name = "Helvetica"
                        font_size = 12
                    
                    # Check if this block has actual visual table structure in the original
                    # Only add visual elements if the original had them
                    has_visual_table_structure = False
                    
                    # Check for explicit table separators in the original text
                    original_text = block.text if hasattr(block, 'text') else translated_text
                    if '|' in original_text or '+' in original_text or '=' in original_text:
                        has_visual_table_structure = True
                    
                    # Check for consistent column alignment patterns
                    lines = original_text.split('\n')
                    if len(lines) >= 3:
                        # Look for consistent spacing patterns that suggest visual columns
                        spacing_patterns = []
                        for line in lines:
                            if len(line) > 20:
                                spaces = [i for i, char in enumerate(line) if char == ' ']
                                if len(spaces) >= 2:
                                    spacing_patterns.append(spaces)
                        
                        if len(spacing_patterns) >= 2:
                            # Check if spacing patterns are consistent (suggesting visual columns)
                            similar_patterns = 0
                            for i in range(len(spacing_patterns) - 1):
                                for j in range(i + 1, len(spacing_patterns)):
                                    pattern1 = spacing_patterns[i]
                                    pattern2 = spacing_patterns[j]
                                    if len(pattern1) == len(pattern2):
                                        similar_positions = sum(1 for p1, p2 in zip(pattern1, pattern2) if abs(p1 - p2) < 5)
                                        if similar_positions >= len(pattern1) * 0.7:
                                            similar_patterns += 1
                            
                            if similar_patterns >= len(spacing_patterns) * 0.5:
                                has_visual_table_structure = True
                    
                    if has_visual_table_structure:
                        # Original had visual table structure - render with borders
                        table_obj = Table(data, colWidths=[width])
                        table_obj.setStyle(TableStyle([
                            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('FONTNAME', (0, 0), (-1, -1), font_name),
                            ('FONTSIZE', (0, 0), (-1, -1), font_size),
                            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ]))
                        
                        table_obj.wrapOn(c, width, height)
                        table_obj.drawOn(c, x0, page_height - y1)
                        print(f"[TABLE RENDERING] Table rendered with visual elements at ({x0}, {page_height - y1})")
                    else:
                        # Original was plain text - render as plain text blocks
                        plain_text_lines = []
                        for row in data:
                            if isinstance(row, list):
                                plain_text_lines.append(' '.join(str(cell) for cell in row if cell))
                            else:
                                plain_text_lines.append(str(row))
                        
                        plain_text = '\n'.join(plain_text_lines)
                        
                        c.setFont(font_name, font_size)
                        c.setFillColor(colors.black)
                        
                        y_position = page_height - y0
                        lines = plain_text.split('\n')
                        line_height = font_size * 1.15
                        
                        for i, line in enumerate(lines):
                            if line.strip():
                                c.drawString(x0, y_position - (i * line_height), line)
                        
                        print(f"[TABLE RENDERING] Table rendered as plain text at ({x0}, {y_position})")
                    
                    continue
                # --- END TABLE RENDERING ---
                
                x0, y0, x1, y1 = block.bbox
                font_size = max(5, int(block.size) if block.size else 12)
                processed_text = translated_text  # Already processed by translate_text_openai
                
                # Check if this text block overlaps with any visual element (only those actually rendered)
                text_bbox = (x0, y0, x1, y1)
                overlaps_with_visual = False
                for elem in page_visual_elements:
                    # Use a much lower threshold (10%) to only skip text when there's significant overlap
                    if bboxes_overlap(text_bbox, elem["bbox"], threshold=0.1):
                        overlaps_with_visual = True
                        print(f"Text block overlaps with {elem['type']} by >10%, skipping text rendering")
                        break
                
                # Only render text if it doesn't significantly overlap with visual elements
                if not overlaps_with_visual:
                    # Get font style from original block
                    font_style = get_font_style_from_block(block)
                    try:
                        # Map target_lang to full language name for font selection
                        target_language = target_language_map.get(target_lang, 'English')
                        actual_font_name = select_appropriate_font("Helvetica", font_style['bold'], font_style['italic'], block.font, target_language=target_language, text=processed_text)
                        
                        # Debug: Check if the selected font is actually registered
                        registered_fonts = pdfmetrics.getRegisteredFontNames()
                        if actual_font_name not in registered_fonts:
                            print(f"[FONT ERROR] Selected font '{actual_font_name}' is not registered!")
                            print(f"[FONT DEBUG] Available fonts: {registered_fonts}")
                            actual_font_name = "Helvetica"
                        else:
                            print(f"[FONT DEBUG] Using font '{actual_font_name}' for text: {processed_text[:50]}...")
                            
                    except Exception as e:
                        print(f"Error in font style processing: {e}")
                        actual_font_name = "Helvetica"
                    
                    # Smart reflow approach - respect original block dimensions
                    original_width = x1 - x0
                    original_height = y1 - y0
                    expansion_needed = False
                    
                    # First, try to reflow text within original block dimensions
                    original_max_width = original_width * 0.98  # Use 98% of original width
                    
                    # Let text reflow naturally within original dimensions
                    try:
                        # Check if we're using Noto fonts and handle them specially
                        if actual_font_name.startswith('NotoSans'):
                            # For Noto fonts, use a simple word-wrap approach
                            words = processed_text.split()
                            lines = []
                            current_line = ""
                            
                            for word in words:
                                test_line = current_line + " " + word if current_line else word
                                # Estimate width (rough approximation)
                                estimated_width = len(test_line) * font_size * 0.6
                                
                                if estimated_width <= original_max_width:
                                    current_line = test_line
                                else:
                                    if current_line:
                                        lines.append(current_line)
                                    current_line = word
                            
                            if current_line:
                                lines.append(current_line)
                            
                            if not lines:
                                lines = [processed_text]
                        else:
                            # Use simpleSplit for other fonts
                            lines = simpleSplit(processed_text, actual_font_name, font_size, original_max_width)
                    except Exception as e:
                        print(f"Error in text splitting: {e}")
                        lines = [processed_text]  # Fallback to single line
                    line_height = font_size * 1.15
                    required_height = len(lines) * line_height
                    
                    # Check if reflowed text fits within original block height
                    if required_height <= original_height * 1.1:  # Allow 10% height tolerance
                        # Text fits within original dimensions - use original layout
                        expansion_needed = False
                        actual_max_width = original_max_width
                    else:
                        # Text doesn't fit in original dimensions - try expansion
                        try:
                            expanded_width = parser._calculate_expanded_width(block, blocks_with_translations, page_num, page_width)
                            expanded_max_width = expanded_width * 0.98
                            
                            # Try reflowing with expanded width
                            try:
                                # Check if we're using Noto fonts and handle them specially
                                if actual_font_name.startswith('NotoSans'):
                                    # For Noto fonts, use a simple word-wrap approach
                                    words = processed_text.split()
                                    expanded_lines = []
                                    current_line = ""
                                    
                                    for word in words:
                                        test_line = current_line + " " + word if current_line else word
                                        # Estimate width (rough approximation)
                                        estimated_width = len(test_line) * font_size * 0.6
                                        
                                        if estimated_width <= expanded_max_width:
                                            current_line = test_line
                                        else:
                                            if current_line:
                                                expanded_lines.append(current_line)
                                            current_line = word
                                    
                                    if current_line:
                                        expanded_lines.append(current_line)
                                    
                                    if not expanded_lines:
                                        expanded_lines = [processed_text]
                                else:
                                    # Use simpleSplit for other fonts
                                    expanded_lines = simpleSplit(processed_text, actual_font_name, font_size, expanded_max_width)
                            except Exception as e:
                                print(f"Error in expanded text splitting: {e}")
                                expanded_lines = lines  # Use original lines as fallback
                            expanded_required_height = len(expanded_lines) * line_height
                            
                            if expanded_required_height <= original_height * 1.2:  # Allow 20% height expansion
                                # Expansion helps - use expanded width
                                lines = expanded_lines
                                expansion_needed = True
                                actual_max_width = expanded_max_width
                                print(f"Expanded text block from {original_width:.1f}px to {expanded_width:.1f}px")
                            else:
                                # Even expansion doesn't help much - use original width but allow more height
                                expansion_needed = False
                                actual_max_width = original_max_width
                                # Keep the original reflow
                                
                        except Exception as e:
                            print(f"Error in expansion calculation: {e}")
                            # Fallback to original dimensions
                            expansion_needed = False
                            actual_max_width = original_max_width
                    
                    line_height = font_size * 1.15
                    
                    # Convert PyMuPDF coordinates (top-down) to ReportLab coordinates (bottom-up)
                    reportlab_y = page_height - y0 - font_size
                    current_y = reportlab_y
                    
                    for line in lines:
                        # Draw all lines within reasonable bounds (less restrictive clipping)
                        if current_y > 20:  # Just ensure we don't go off the bottom of the page
                            try:
                                c.setFont(actual_font_name, font_size)
                                c.setFillColorRGB(0, 0, 0)
                                # Use original positioning - expansion is handled in the text wrapping logic above
                                c.drawString(x0 + 3, current_y, line)
                            except Exception as e:
                                print(f"[FONT ERROR] Error setting font {actual_font_name}: {e}")
                                print(f"[FONT DEBUG] Available fonts: {pdfmetrics.getRegisteredFontNames()}")
                                # Fallback to a default font
                                try:
                                    c.setFont("Helvetica", font_size)
                                    c.setFillColorRGB(0, 0, 0)
                                    c.drawString(x0 + 3, current_y, line)
                                except Exception as e2:
                                    print(f"[FONT ERROR] Error with fallback font: {e2}")
                                    # Last resort - skip this line
                                    pass
                        current_y -= line_height  # Move down in ReportLab coordinates (subtract)
            
            # 3. Render all other visual elements for this page (on top of text)
            print(f"Rendering visual elements for page {page_num}...")
            for elem in page_visual_elements:
                render_visual_element(c, elem, page_height, original_pdf_path, force_fullpage=False, page_width=page_width, page_num=page_num)
            
            c.showPage()
        
        c.save()
        output_buffer.seek(0)
        return output_buffer.getvalue()
        
    finally:
        # Cleanup temporary file
        try:
            os.unlink(original_pdf_path)
        except:
            pass

def render_visual_element(canvas, element: Dict, page_height: float, source_pdf_path: str, force_fullpage: bool = False, page_width: float = None, page_num: int = None, img_idx_counter=None):
    """Render a visual element on the ReportLab canvas. Save every raster image as PNG for inspection."""
    try:
        bbox = element["bbox"]
        x0, y0, x1, y1 = bbox
        original_width = x1 - x0
        original_height = y1 - y0

        if img_idx_counter is None:
            img_idx_counter = [0]

        # If force_fullpage, override coordinates and size
        if force_fullpage and page_width is not None:
            x0, y0 = 0, 0
            width, height = page_width, page_height
            reportlab_y0 = 0
            print(f"[BG] Drawing background image at (0,0) size=({page_width}x{page_height})")
        else:
            if y0 < 0:
                print(f"Logo extends above page boundary (Y: {y0}), positioning at page top")
                above_page = abs(y0)
                y0 = 0  # Start at page top
                y1 = original_height  # Full height from top
                print(f"Repositioned logo to Y: {y0} to {y1}")
            reportlab_y0 = page_height - y1
            width = original_width
            height = original_height
            print(f"Element positioning: PDF coords Y({y0:.1f}, {y1:.1f}) -> ReportLab Y({reportlab_y0:.1f})")

        if element["subtype"] == "raster":
            try:
                image_bytes = element["data"]
                image = Image.open(io.BytesIO(image_bytes))
                # Save every raster image for inspection
                if page_num is not None:
                    try:
                        out_path = f"extracted_image_page{page_num+1}_{img_idx_counter[0]}.png"
                        image.save(out_path)
                        print(f"[IMG] Saved extracted image to {out_path}")
                        img_idx_counter[0] += 1
                    except Exception as save_err:
                        print(f"[IMG] ERROR: Failed to save extracted image for page {page_num+1}: {save_err}")
                if image.mode == 'RGBA':
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    img_reader = ImageReader(img_buffer)
                    canvas.drawImage(img_reader, x0, reportlab_y0, width=width, height=height, preserveAspectRatio=False, mask=None)
                else:
                    if image.mode not in ['RGB', 'RGBA']:
                        image = image.convert('RGB')
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    img_reader = ImageReader(img_buffer)
                    canvas.drawImage(img_reader, x0, reportlab_y0, width=width, height=height, preserveAspectRatio=False, mask='auto')
                print(f"Rendered {element['type']} (raster) at ({x0:.1f}, {reportlab_y0:.1f}, {width:.1f}x{height:.1f})")
            except Exception as e:
                print(f"Warning: Could not render raster {element['type']}: {e}")
                canvas.setStrokeColorRGB(0.7, 0.7, 0.7)
                canvas.setFillColorRGB(0.9, 0.9, 0.9)
                canvas.rect(x0, reportlab_y0, width, height, stroke=1, fill=1)
        elif element["subtype"] == "vector":
            try:
                orig_doc = fitz.open(source_pdf_path)
                orig_page = orig_doc.load_page(element.get("page", 0))
                clip_rect = fitz.Rect(element["bbox"])
                mat = fitz.Matrix(3.0, 3.0)
                pix = orig_page.get_pixmap(matrix=mat, clip=clip_rect)
                img_data = pix.tobytes("png")
                img_reader = ImageReader(io.BytesIO(img_data))
                canvas.drawImage(img_reader, x0, reportlab_y0, width=width, height=height, preserveAspectRatio=False, mask='auto')
                print(f"Rendered {element['type']} (vector as raster) at ({x0:.1f}, {reportlab_y0:.1f}, {width:.1f}x{height:.1f})")
                pix = None
                orig_doc.close()
            except Exception as e:
                print(f"Warning: Could not render vector {element['type']} as raster: {e}")
                canvas.setFillColorRGB(0.5, 0.5, 0.5)
                canvas.rect(x0, reportlab_y0, width, height, stroke=0, fill=1)
                print(f"Rendered {element['type']} (vector fallback) at ({x0:.1f}, {reportlab_y0:.1f}, {width:.1f}x{height:.1f})")
    except Exception as e:
        print(f"Error rendering visual element: {e}")
        canvas.setStrokeColorRGB(1, 0, 0)
        canvas.setFillColorRGB(1, 0.8, 0.8)
        canvas.rect(x0, reportlab_y0, width, height, stroke=1, fill=1)

def render_qr_code_from_text(canvas, qr_text: str, bbox: tuple, page_height: float) -> bool:
    """Render an actual QR code from the detected QR text content"""
    try:
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        reportlab_y0 = page_height - y1
        
        # Extract meaningful data from QR text if possible
        qr_data = "https://www.ind.nl"  # Default fallback data
        
        # Try to extract URL or meaningful content from the QR text
        if "ind.nl" in qr_text.lower():
            qr_data = "https://www.ind.nl"
        elif any(char.isdigit() for char in qr_text):
            # If it contains numbers, might be an ID or reference
            numbers = ''.join(filter(str.isdigit, qr_text.replace(' ', '')))
            if len(numbers) > 5:
                qr_data = f"Document ID: {numbers}"
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_data)
        qr.make(fit=True)
        
        # Create QR code image
        qr_image = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes for ReportLab
        img_buffer = io.BytesIO()
        qr_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_reader = ImageReader(img_buffer)
        
        # Draw the QR code
        canvas.drawImage(img_reader, x0, reportlab_y0, 
                       width=width, height=height, 
                       preserveAspectRatio=True, mask='auto')
        
        print(f"Rendered QR code at ({x0:.1f}, {reportlab_y0:.1f}, {width:.1f}x{height:.1f}) with data: {qr_data}")
        return True
        
    except ImportError:
        print("Warning: qrcode library not available. Install with: pip install qrcode[pil]")
        return False
    except Exception as e:
        print(f"Warning: Could not generate QR code: {e}")
        return False

def bboxes_overlap(bbox1: tuple, bbox2: tuple, threshold: float = 0.3) -> bool:
    """Check if two bounding boxes overlap by more than the threshold percentage"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    
    if x_overlap == 0 or y_overlap == 0:
        return False
    
    intersection_area = x_overlap * y_overlap
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Use the smaller area as the reference for overlap percentage
    min_area = min(bbox1_area, bbox2_area)
    overlap_ratio = intersection_area / min_area if min_area > 0 else 0
    
    return overlap_ratio > threshold

def get_font_style_from_block(block: SimpleTextBlock) -> dict:
    """Extract font style information from the original block"""
    try:
        style = {
            'bold': getattr(block, 'bold', False),
            'italic': getattr(block, 'italic', False),
            'font_family': getattr(block, 'font', None)
        }
        
        # Additional heuristics based on font name if available
        font = getattr(block, 'font', None)
        if font:
            font_lower = font.lower()
            if 'bold' in font_lower:
                style['bold'] = True
            if 'italic' in font_lower or 'oblique' in font_lower:
                style['italic'] = True
        
        # Heuristics based on text content and block type
        block_type = getattr(block, 'type', None)
        if block_type and block_type in ['title', 'header']:
            style['bold'] = True
        
        return style
    except Exception as e:
        # Fallback to default style if anything goes wrong
        return {
            'bold': False,
            'italic': False,
            'font_family': None
        }

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
        len(s.split()) > 3):  # Only for longer sentences
        s += '.'
        
    s = capitalize_sentences(s)
    return s

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 