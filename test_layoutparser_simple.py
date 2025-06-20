#!/usr/bin/env python3

"""
Simplified Layout Parser for PDF Processing
Uses PyMuPDF for basic layout detection without requiring detectron2
"""

import fitz  # PyMuPDF
import json
import re
from typing import List, Dict, Any, Tuple
import argparse
import os
import requests
from dotenv import load_dotenv
import numpy as np
from PIL import Image, ImageDraw
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import simpleSplit
import io
import openai
from collections import defaultdict
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import getSampleStyleSheet
from fuzzywuzzy import fuzz

# Set up your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

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

class SimplePDFLayoutParser:
    """Simplified PDF layout parser using PyMuPDF"""
    
    def __init__(self, require_api_key=True):
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if require_api_key:
            assert self.api_key, "OPENAI_API_KEY not found in environment!"
    
    def google_translate_text(self, text: str, target_lang: str) -> str:
        """Translate text using OpenAI's API."""
        if not text.strip():
            return ""
        try:
            response = openai.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Translate the following text to {target_lang}."},
                {"role": "user", "content": text}
            ])
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during OpenAI translation: {e}")
            return text
    
    def extract_blocks_from_pdf(self, pdf_path: str) -> List[Tuple[SimpleTextBlock, int]]:
        """Extract paragraph blocks from PDF using PyMuPDF with page numbers, storing font/style info."""
        doc = fitz.open(pdf_path)
        blocks = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_dict = page.get_text("dict")
            tables = self._detect_table_regions(text_dict)
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
    
    def _format_table_as_text(self, table: Dict) -> str:
        """Format table data as readable text for display purposes"""
        if table["type"] == "product_table":
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
    
    def _classify_block_type(self, text: str, bbox: Tuple[float, float, float, float], page_rect) -> str:
        """Simple rule-based block type classification"""
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
        
        # Check for table-like content (contains numbers, currency, dates)
        if re.search(r'\d+[.,]\d+|€|\$|USD|EUR|\d{2}[-/]\d{2}[-/]\d{4}', text):
            return "table"
        
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
    
    def translate_text(self, text: str, target_lang: str = 'en') -> str:
        """Translate text using OpenAI API with post-processing improvements"""
        if not text.strip():
            return text
        try:
            preprocessed_text = self._preprocess_for_translation(text)
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            prompt = (
                f"Translate the following text to {target_lang}. "
                f"CRITICAL PRESERVATION RULES - DO NOT TRANSLATE THESE: "
                f"1. Keep ALL web addresses EXACTLY as they are (www.example.com, http://..., https://...) "
                f"2. Keep ALL postal codes EXACTLY as written (1087 EM, 9560 AA, etc.) - These are location codes, NOT words to translate "
                f"3. Keep ALL street names, addresses, and city names UNCHANGED "
                f"4. Keep ALL phone numbers, email addresses, and reference numbers UNCHANGED "
                f"5. Keep ALL URLs and website paths UNCHANGED (including /path/to/page) "
                f"6. IMPORTANT: Letters like 'EM', 'AA', 'BB' in postal codes are NOT Dutch words - they are postal district codes "
                f"Only provide the translation, no explanations or additional text:\n\n"
                f"{preprocessed_text}\n\nTranslation:"
            )
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1024
            }
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            result_text = result["choices"][0]["message"]["content"].strip()
            # Post-processing fixes for common translation issues
            result_text = self._fix_translation_issues(result_text, text)
            # Convert placeholders back to proper English terms - do this LAST
            result_text = self._convert_placeholders(result_text)
            return result_text
        except Exception as e:
            print(f"Translation error for '{text[:50]}...': {e}")
            return text
    
    def _preprocess_for_translation(self, text: str) -> str:
        """Pre-process text to handle Dutch terms that might be dropped by Google Translate"""
        # Dictionary of Dutch terms that should be preserved/replaced before translation
        dutch_terms = {
            'INLOGGEGEVENS': 'LOGIN_DETAILS_PLACEHOLDER',
            'AFKOOP EIGEN RISICO': 'DEDUCTIBLE_BUYOUT_PLACEHOLDER',
            'SEPA MACHTIGING': 'SEPA_AUTHORIZATION_PLACEHOLDER',
            
            # Single word terms that Google Translate often gets wrong
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
                # Replace with placeholder that Google Translate won't drop
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
            # Handle case-insensitive replacement since Google Translate may lowercase placeholders
            text = re.sub(re.escape(placeholder), english_term, text, flags=re.IGNORECASE)
        
        return text
    
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
    
    def _apply_sentence_word_casing(self, word: str, position: int, all_words: list, is_sentence_start: bool) -> str:
        """Apply proper casing for a word in sentence context"""
        if is_sentence_start:
            return word.capitalize()
        elif self._is_acronym(word):
            return word.upper()
        elif self._is_likely_proper_noun(word, position, all_words):
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
    
    def _fix_translation_issues(self, translated: str, original: str) -> str:
        """Fix common translation issues"""
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
        
        # Fix specific translation errors that Google Translate makes
        translated = re.sub(r'\bSurrender deductible\b', 'DEDUCTIBLE BUYOUT', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bAfkoop eigen risico\b', 'DEDUCTIBLE BUYOUT', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bSurrender own risk\b', 'DEDUCTIBLE BUYOUT', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bStuberty\b', 'Unit Price', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bNumber\b', 'Quantity', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bPiece price\b', 'Unit Price', translated, flags=re.IGNORECASE)
        
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
        
        # Apply improved casing preservation BEFORE the specific fixes
        translated = self._preserve_original_casing(original, translated)
        
        # Apply URL and web address formatting
        translated = self._format_urls_and_web_addresses(translated)
        
        # Apply post-casing fixes to handle remaining issues
        translated = self._apply_post_casing_fixes(translated)
        
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
        
        return translated.strip()
    
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
        
        # Fix mid-sentence capitalized common words
        common_words_pattern = r'\b(And|Or|But|With|By|For|At|In|On|Of|The|A|An|This|That|These|Those|Your|Our|Give|Through|From|Can|Have|It|About|During|As|Contact|Ask|Based|Do|Not|Agree|Permission|Log|Change|Details)\b'
        
        def lowercase_unless_sentence_start(match):
            word = match.group(1)
            start_pos = match.start()
            
            # Check if this word is at the start of a sentence
            # Look backwards for sentence-ending punctuation
            text_before = text[:start_pos].rstrip()
            if not text_before or text_before.endswith(('.', '!', '?', ':')):
                return word.capitalize()  # Keep capitalized if sentence start
            else:
                return word.lower()  # Make lowercase if mid-sentence
        
        text = re.sub(common_words_pattern, lowercase_unless_sentence_start, text)
        
        # Fix specific remaining issues
        # Fix "Deductible" when it should be "deductible" (not at sentence start)
        text = re.sub(r'(?<!^)(?<![.!?]\s)\bDeductible\b', 'deductible', text)
        
        # Final specific fixes - apply these last
        text = re.sub(r'\bYOU can\b', 'You can', text)
        text = re.sub(r'\bYOU Can\b', 'You can', text)
        
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
    
    def create_translated_pdf(self, original_pdf_path: str, blocks_with_translations: list, output_pdf_path: str):
        import fitz
        doc = fitz.open(original_pdf_path)
        # Group blocks by page
        pages_blocks = {}
        for block, blk_page_num, translated_text in blocks_with_translations:
            if blk_page_num not in pages_blocks:
                pages_blocks[blk_page_num] = []
            pages_blocks[blk_page_num].append((block, translated_text))

        # Build a mapping of original font sizes (within a small tolerance) to a common output font size.
        # (This ensures that blocks with the same font size in the input always get the same output font size.)
        font_size_map = {}
        for block, _, _ in blocks_with_translations:
            if block.size is not None:
                # Round to 1 decimal place (or use a small tolerance) to group similar font sizes.
                orig_size = round(block.size, 1)
                if orig_size not in font_size_map:
                    # Use the original size (or a minimum of 5) as the common output font size.
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
                # Insert translated text: treat each block as a paragraph
                for block, translated_text in pages_blocks[page_num]:
                    x0, y0, x1, y1 = block.bbox
                    # Use a Unicode TTF font for better character support
                    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                    font_name = "dejavusans"
                    try:
                        # Register the font if not already registered
                        if font_name not in fitz.Font(fonts=True):
                            fitz.Font(fontfile=font_path, fontname=font_name)
                    except Exception as e:
                        print(f"Warning: Could not register DejaVuSans.ttf, falling back to helv. Error: {e}")
                        font_name = "helv"
                    processed_text = self._postprocess_english(translated_text, block.text)
                    max_width = x1 - x0 - 6
                    max_height = y1 - y0 - 6
                    # Use the common font size (from font_size_map) if available, otherwise fallback to a default (e.g. 12).
                    orig_size = round(block.size, 1) if block.size is not None else None
                    font_size = font_size_map.get(orig_size, 12)
                    min_font_size = 5  # enforce a minimum for legibility
                    if font_size < min_font_size:
                         font_size = min_font_size
                    # (Optional: if you want to wrap text and adjust font size further, uncomment the following loop.)
                    # while font_size >= min_font_size:
                    #     wrapped_lines = self.wrap_text_to_width(processed_text, font_size, font_name, max_width, page)
                    #     line_height = font_size * 1.15
                    #     total_height = len(wrapped_lines) * line_height
                    #     if total_height <= max_height:
                    #         break
                    #     font_size -= 1
                    # (For now, we force the common font size.)
                    wrapped_lines = self.wrap_text_to_width(processed_text, font_size, font_name, max_width, page)
                    line_height = font_size * 1.15
                    start_y = y0 + font_size + 2
                    current_y = start_y
                    for line in wrapped_lines:
                        if current_y < y1 + 3:
                            try:
                                # Clean the text before inserting to prevent encoding issues
                                clean_line = self._clean_text_for_pdf(str(line))
                                # Insert as Unicode string, using the Unicode font
                                page.insert_text((x0 + 3, current_y), clean_line, fontsize=font_size, color=(0, 0, 0), fontname=font_name)
                            except Exception as e:
                                print(f"Error inserting text: {line} – {e}")
                        current_y += line_height
        doc.save(output_pdf_path)
        doc.close()
        print(f"Translated PDF saved to: {output_pdf_path}")
    
    def wrap_text_to_width(self, text, font_size, font_name, max_width, page):
        import fitz
        lines = []
        for para in text.split('\n'):
            words = para.split()
            if not words:
                lines.append('')
                continue
            current_line = words[0]
            for word in words[1:]:
                test_line = current_line + ' ' + word
                w = fitz.get_text_length(test_line, fontname=font_name, fontsize=font_size)
                if w <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            lines.append(current_line)
        return lines
    
    def openai_translate_batch(self, texts, target_lang='en', batch_size=10):
        import re
        import requests
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        assert api_key, "OPENAI_API_KEY not found in environment!"
        all_translations = []
        for text in texts:
            # Split large text into paragraphs (by double newlines or '\n\n')
            paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
            translated_paragraphs = []
            for para in paragraphs:
                if len(para) > 500:
                    # Split long paragraph into sentences
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    translated_sentences = []
                    for i in range(0, len(sentences), 5):
                        batch = sentences[i:i+5]
                        numbered_texts = "\n".join(f"{j+1}. {t}" for j, t in enumerate(batch))
                        prompt = (
                            f"Translate the following sentences to {target_lang}. Return ONLY a numbered list of translations in the same order as the input. Do not include explanations.\n\n"
                            f"Sentences to translate:\n{numbered_texts}"
                        )
                        url = "https://api.openai.com/v1/chat/completions"
                        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                        data = {
                            "model": "gpt-4o-mini",
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.2
                        }
                        response = requests.post(url, headers=headers, json=data)
                        print(f"OpenAI API response for long sentence batch: {response.text}")
                        if response.status_code == 200:
                            content = response.json()["choices"][0]["message"]["content"]
                            # Extract translations from numbered list
                            lines = [l.strip() for l in content.split('\n') if l.strip()]
                            translations = []
                            for l in lines:
                                m = re.match(r"\d+\.\s*(.*)", l)
                                if m:
                                    translations.append(m.group(1))
                            if not translations:
                                translations = [content.strip()]
                            translated_sentences.extend(translations)
                        else:
                            print(f"OpenAI API error: {response.status_code} {response.text}")
                            translated_sentences.extend(batch)
                    # Join sentences for this paragraph, preserving paragraph breaks
                    translated_paragraphs.append(' '.join(translated_sentences))
                else:
                    # Normal paragraph translation
                    prompt = (
                        f"Translate the following text to {target_lang}. Return ONLY the translation. Do not include explanations.\n\n"
                        f"Text to translate:\n{para}"
                    )
                    url = "https://api.openai.com/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    data = {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.2
                    }
                    response = requests.post(url, headers=headers, json=data)
                    print(f"OpenAI API response for paragraph: {response.text}")
                    if response.status_code == 200:
                        content = response.json()["choices"][0]["message"]["content"]
                        translated_paragraphs.append(content.strip())
                    else:
                        print(f"OpenAI API error: {response.status_code} {response.text}")
                        translated_paragraphs.append(para)
            # Join translated paragraphs with double newlines to preserve paragraph breaks
            all_translations.append('\n\n'.join(translated_paragraphs))
        return all_translations
    
    def process_pdf(self, pdf_path: str, output_path: str = None, translate: bool = True, target_lang: str = 'en'):
        print(f"Processing PDF: {pdf_path}")
        
        # Extract visual elements first
        print("Extracting visual elements...")
        visual_elements = self.extract_visual_elements(pdf_path, page_num=0)
        print(f"Found {len(visual_elements)} visual elements:")
        for i, elem in enumerate(visual_elements):
            print(f"  {i+1}. {elem['type']} ({elem['subtype']}) at {elem['bbox']} - {elem['width']:.1f}x{elem['height']:.1f}")
        
        # Extract text blocks with page numbers
        blocks_with_pages = self.extract_blocks_from_pdf(pdf_path)
        print(f"Extracted {len(blocks_with_pages)} initial text blocks")
        
        # Remove duplicates
        unique_blocks_with_pages = self.remove_duplicates(blocks_with_pages)
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
        if translate and texts:
                translations = self.openai_translate_batch(texts, target_lang=target_lang)
                print(f"Translations received: {len(translations)}")
            else:
                translations = texts
            
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
        
        # Always create output PDF with translated text and visual elements
        if output_path and output_path.endswith('.pdf'):
            output_pdf_path = output_path
        else:
            output_pdf_path = pdf_path.replace('.pdf', '_translated.pdf')
        print(f"Creating translated PDF: {output_pdf_path}")
        
        # Create a Unicode-compatible PDF using ReportLab with visual elements
        self.create_translated_pdf_reportlab_with_visuals(pdf_path, blocks_with_translations, visual_elements, output_pdf_path)
        
        # Visualize paragraph blocks
        self.visualize_paragraph_blocks_on_page(pdf_path, unique_blocks_with_pages, 'paragraph_blocks.png')
        print("Processing complete!")

    def _detect_table_regions(self, text_dict: dict) -> List[Dict]:
        """Detect table-like regions in the PDF and extract them cell-by-cell"""
        tables = []
        
        for block_idx, block in enumerate(text_dict["blocks"]):
            if "lines" not in block:
                continue
                
            lines = block["lines"]
            if len(lines) < 4:  # Tables should have at least 4 elements
                continue
            
            # Check if this looks like a table based on content patterns
            line_texts = [" ".join([span["text"] for span in line["spans"]]).strip() for line in lines]
            
            # Product table detection
            if self._is_product_table(line_texts):
                table = self._extract_product_table(block, line_texts, block_idx)
                if table:
                    tables.append(table)
            
            # VAT table detection  
            elif self._is_vat_table(line_texts):
                table = self._extract_vat_table(block, line_texts, block_idx)
                if table:
                    tables.append(table)
            
            # Summary table detection
            elif self._is_summary_table(line_texts):
                table = self._extract_summary_table(block, line_texts, block_idx)
                if table:
                    tables.append(table)
        
        return tables
    
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

    def visualize_paragraph_blocks_on_page(self, pdf_path: str, blocks_with_pages: list, output_png: str):
        import fitz
        from PIL import Image, ImageDraw
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(img)
        for block, page_num in blocks_with_pages:
            if page_num == 0:
                x0, y0, x1, y1 = block.bbox
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                draw.text((x0, y0-12), "Paragraph", fill="red")
        img.save(output_png)
        print(f"Paragraph block visualization saved to {output_png}")

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
    
    def _get_font_style_from_block(self, block: SimpleTextBlock) -> dict:
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
    
    def _register_font_family(self, base_font_name: str):
        """Register a font family with all style variants (regular, bold, italic, bold-italic)"""
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import os
        
        # Font path mappings for different systems
        font_paths = {
            "DejaVuSans": {
                "regular": [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/Library/Fonts/DejaVuSans.ttf",
                    "/System/Library/Fonts/Arial.ttf",
                    "/System/Library/Fonts/Helvetica.ttc"
                ],
                "bold": [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    "/Library/Fonts/DejaVuSans-Bold.ttf",
                    "/System/Library/Fonts/Arial Bold.ttf",
                    "/System/Library/Fonts/Helvetica.ttc"
                ],
                "italic": [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
                    "/Library/Fonts/DejaVuSans-Oblique.ttf",
                    "/System/Library/Fonts/Arial Italic.ttf",
                    "/System/Library/Fonts/Helvetica.ttc"
                ],
                "bold-italic": [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
                    "/Library/Fonts/DejaVuSans-BoldOblique.ttf",
                    "/System/Library/Fonts/Arial Bold Italic.ttf",
                    "/System/Library/Fonts/Helvetica.ttc"
                ]
            }
        }
        
        if base_font_name not in font_paths:
            print(f"Warning: No font paths defined for {base_font_name}")
            return
        
        # Register each font variant
        for style, paths in font_paths[base_font_name].items():
            font_registered = False
            for path in paths:
                if os.path.exists(path):
                    try:
                        if style == "regular":
                            font_reg_name = base_font_name
                        else:
                            font_reg_name = f"{base_font_name}-{style.title()}"
                        
                        pdfmetrics.registerFont(TTFont(font_reg_name, path))
                        font_registered = True
                        break
                    except Exception as e:
                        continue
            
            if not font_registered:
                print(f"Warning: Could not register {base_font_name} {style} variant")

    def _select_appropriate_font(self, default_font_name, is_bold, is_italic, original_font_name):
        """
        Select the most appropriate font based on the original font characteristics.
        Attempts to match bold/italic styling when possible.
        """
        from reportlab.pdfbase import pdfmetrics
        
        try:
            # Determine the font style suffix
            if is_bold and is_italic:
                style_suffix = "-Bold-Italic"
            elif is_bold:
                style_suffix = "-Bold"
            elif is_italic:
                style_suffix = "-Italic"
            else:
                style_suffix = ""
            
            # Try the styled version of the default font first
            styled_font_name = default_font_name + style_suffix
            
            # Check if the styled font is registered
            registered_fonts = pdfmetrics.getRegisteredFontNames()
            
            if styled_font_name in registered_fonts:
                return styled_font_name
            
            # If DejaVu fonts are not available, use built-in Helvetica variants
            helvetica_variants = {
                "": "Helvetica",
                "-Bold": "Helvetica-Bold", 
                "-Italic": "Helvetica-Oblique",
                "-Bold-Italic": "Helvetica-BoldOblique"
            }
            
            helvetica_font = helvetica_variants.get(style_suffix, "Helvetica")
            
            # Helvetica variants are built-in, so they should always be available
            return helvetica_font
        except Exception as e:
            return "Helvetica"

    def create_translated_pdf_reportlab(self, original_pdf_path: str, blocks_with_translations: list, output_pdf_path: str):
        """
        Generate a translated PDF using ReportLab with Unicode font support.
        Each block is rendered at its bounding box with the appropriate font size.
        """
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.utils import simpleSplit
        import os
        import fitz
        
        debug = False  # Enable debug output

        # Register Unicode fonts with style variants
        font_name = "Helvetica"  # Force Helvetica for better compatibility
        # Skip DejaVu registration for now to avoid potential issues
        # try:
        #     self._register_font_family(font_name)
        #     # Check if DejaVu fonts were actually registered
        #     registered_fonts = pdfmetrics.getRegisteredFontNames()
        #     if "DejaVuSans-Bold" not in registered_fonts:
        #         print("DejaVu fonts not available, using Helvetica variants")
        #         font_name = "Helvetica"
        # except Exception as e:
        #     print(f"Warning: Font registration failed: {e}")
        #     font_name = "Helvetica"
        
        # If font registration fails, fall back to Helvetica
        if font_name not in pdfmetrics.getRegisteredFontNames():
            font_name = "Helvetica"

        # Get page size from original PDF
        doc = fitz.open(original_pdf_path)
        page_sizes = [ (page.rect.width, page.rect.height) for page in doc ]
        doc.close()

        c = canvas.Canvas(output_pdf_path, pagesize=letter)
        for page_num, (page_width, page_height) in enumerate(page_sizes):
            c.setPageSize((page_width, page_height))
            # Draw all blocks for this page
            for block, blk_page_num, translated_text in blocks_with_translations:
                if blk_page_num != page_num:
                    continue
                
                # Skip QR codes - treat them as visual elements that shouldn't be rendered as text
                if block.type == "qr_code":
                    # Draw a placeholder box instead
                    x0, y0, x1, y1 = block.bbox
                    reportlab_y0 = page_height - y1
                    reportlab_y1 = page_height - y0
                    c.setStrokeColorRGB(0.5, 0.5, 0.5)  # Gray border
                    c.setFillColorRGB(0.9, 0.9, 0.9)    # Light gray fill
                    c.rect(x0, reportlab_y0, x1-x0, reportlab_y1-reportlab_y0, stroke=1, fill=1)
                    # Add QR code label
                    c.setFillColorRGB(0, 0, 0)
                    c.setFont(font_name, 8)
                    c.drawString(x0 + 5, reportlab_y0 + 5, "[QR Code]")
                    continue
                
                x0, y0, x1, y1 = block.bbox
                font_size = max(5, int(block.size) if block.size else 12)
                processed_text = self._postprocess_english(translated_text, block.text)
                
                # Get font style from original block
                try:
                    font_style = self._get_font_style_from_block(block)
                    actual_font_name = self._select_appropriate_font(font_name, font_style['bold'], font_style['italic'], block.font)
                    
                    if debug:
                        print(f"Block font: {block.font}, Bold: {font_style['bold']}, Italic: {font_style['italic']}, Selected font: {actual_font_name}")
                except Exception as e:
                    if debug:
                        print(f"Error in font style processing: {e}")
                    actual_font_name = font_name
                    font_style = {'bold': False, 'italic': False}
                
                # Smart reflow approach - respect original block dimensions
                original_width = x1 - x0
                original_height = y1 - y0
                expansion_needed = False
                
                # First, try to reflow text within original block dimensions
                original_max_width = original_width * 0.98  # Use 98% of original width
                
                # Let text reflow naturally within original dimensions
                try:
                    lines = simpleSplit(processed_text, actual_font_name, font_size, original_max_width)
                except Exception as e:
                    if debug:
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
                        expanded_width = self._calculate_expanded_width(block, blocks_with_translations, page_num, page_width)
                        expanded_max_width = expanded_width * 0.98
                        
                        # Try reflowing with expanded width
                        try:
                            expanded_lines = simpleSplit(processed_text, actual_font_name, font_size, expanded_max_width)
                        except Exception as e:
                            if debug:
                                print(f"Error in expanded text splitting: {e}")
                            expanded_lines = lines  # Use original lines as fallback
                        expanded_required_height = len(expanded_lines) * line_height
                        
                        if expanded_required_height <= original_height * 1.2:  # Allow 20% height expansion
                            # Expansion helps - use expanded width
                            lines = expanded_lines
                            expansion_needed = True
                            actual_max_width = expanded_max_width
                        else:
                            # Even expansion doesn't help much - use original width but allow more height
                            expansion_needed = False
                            actual_max_width = original_max_width
                            # Keep the original reflow
                            
                    except:
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
                        c.setFont(actual_font_name, font_size)
                        c.setFillColorRGB(0, 0, 0)
                        # Use original positioning - expansion is handled in the text wrapping logic above
                        c.drawString(x0 + 3, current_y, line)
                    current_y -= line_height  # Move down in ReportLab coordinates (subtract)
            c.showPage()
        c.save()
        print(f"[ReportLab] Translated PDF saved to: {output_pdf_path}")

    def extract_visual_elements(self, pdf_path: str, page_num: int = 0) -> List[Dict]:
        """
        Extract all visual elements (images, vector graphics, logos) from a PDF page.
        Returns a list of visual element dictionaries with type, bbox, and data.
        """
        import fitz
        import io
        from PIL import Image
        
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
    
    def _is_likely_logo(self, bbox: tuple, width: float, height: float, page_rect) -> bool:
        """
        Determine if a visual element is likely a logo based on size, position, and aspect ratio.
        """
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
    
    def render_visual_element(self, canvas, element: Dict, page_height: float):
        """
        Render a visual element (logo, image, vector graphic) on the ReportLab canvas.
        """
        from reportlab.lib.utils import ImageReader
        from reportlab.graphics.shapes import Drawing, Rect, Line, Path
        from reportlab.graphics import renderPDF
        from reportlab.lib.colors import Color
        import io
        from PIL import Image
        
        try:
            bbox = element["bbox"]
        x0, y0, x1, y1 = bbox
            original_width = x1 - x0
            original_height = y1 - y0
            
            # Handle negative coordinates properly for logos at page top
            if y0 < 0:
                # Logo extends above page boundary - position it at the very top
                print(f"Logo extends above page boundary (Y: {y0}), positioning at page top")
                # Calculate how much of the logo was above the page
                above_page = abs(y0)
                # Position the logo at the top of the page, showing the full logo
                y0 = 0  # Start at page top
                y1 = original_height  # Full height from top
                print(f"Repositioned logo to Y: {y0} to {y1}")
            
            # Convert coordinates to ReportLab (bottom-up coordinate system)
            # ReportLab Y=0 is at bottom, so we need to flip coordinates
            reportlab_y0 = page_height - y1  # Bottom of the image in ReportLab coords
            reportlab_y1 = page_height - y0   # Top of the image in ReportLab coords
            width = original_width
            height = original_height
            
            # Debug output
            print(f"Element positioning: PDF coords Y({y0:.1f}, {y1:.1f}) -> ReportLab Y({reportlab_y0:.1f}, {reportlab_y1:.1f})")
            
            if element["subtype"] == "raster":
                # Render raster images (logos, photos, etc.)
                try:
                    image_bytes = element["data"]
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert to RGB if necessary
                    if image.mode not in ['RGB', 'RGBA']:
                        image = image.convert('RGB')
                    
                    # Create ImageReader for ReportLab
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    img_reader = ImageReader(img_buffer)
                    
                    # Draw the image at exact position and size, preserving full dimensions
                    # reportlab_y0 is the bottom-left corner of the image in ReportLab coordinates
                    canvas.drawImage(img_reader, x0, reportlab_y0, 
                                   width=width, height=height, 
                                   preserveAspectRatio=True, mask='auto')
                    
                    print(f"Rendered {element['type']} (raster) at ({x0:.1f}, {reportlab_y0:.1f}, {width:.1f}x{height:.1f})")
                    
                except Exception as e:
                    print(f"Warning: Could not render raster {element['type']}: {e}")
                    # Draw a placeholder rectangle
                    canvas.setStrokeColorRGB(0.7, 0.7, 0.7)
                    canvas.setFillColorRGB(0.9, 0.9, 0.9)
                    canvas.rect(x0, reportlab_y0, width, height, stroke=1, fill=1)
                    
            elif element["subtype"] == "vector":
                # Render vector graphics by converting to raster image
                try:
                    # Get the original drawing data
                    drawing_data = element["data"]
                    
                    # Create a temporary PDF page with just this vector graphic
                    import tempfile
                    import fitz
                    
                    # Create a temporary document with the vector graphic
                    temp_doc = fitz.open()
                    temp_page = temp_doc.new_page(width=width, height=height)
                    
                    # Try to recreate the vector graphic on the temporary page
                    # This is complex, so let's use a different approach:
                    # Extract the vector graphic as a raster image from the original PDF
                    
                    # Open the original PDF and get the specific drawing
                    orig_doc = fitz.open(element.get("source_pdf", "sample.pdf"))
                    orig_page = orig_doc.load_page(element.get("page", 0))
                    
                    # Create a clip rectangle for just this vector element
                    clip_rect = fitz.Rect(element["bbox"])
                    
                    # Render just this portion of the page as a high-resolution image
                    mat = fitz.Matrix(3.0, 3.0)  # 3x resolution for better quality
                    pix = orig_page.get_pixmap(matrix=mat, clip=clip_rect)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    img_reader = ImageReader(io.BytesIO(img_data))
                    
                    # Draw the image
                    canvas.drawImage(img_reader, x0, reportlab_y0, 
                                   width=width, height=height, 
                                   preserveAspectRatio=True, mask='auto')
                    
                    print(f"Rendered {element['type']} (vector as raster) at ({x0:.1f}, {reportlab_y0:.1f}, {width:.1f}x{height:.1f})")
                    
                    # Clean up
                    pix = None
                    orig_doc.close()
                    temp_doc.close()
                    
                except Exception as e:
                    print(f"Warning: Could not render vector {element['type']} as raster: {e}")
                    # Fall back to simplified rectangle rendering
                    drawing_data = element["data"]
                    fill_color = drawing_data.get("fill", (0.5, 0.5, 0.5))
                    
                    if fill_color and len(fill_color) >= 3:
                        r, g, b = fill_color[:3]
                        canvas.setFillColorRGB(r, g, b)
                    else:
                        canvas.setFillColorRGB(0.5, 0.5, 0.5)
                    
                    canvas.rect(x0, reportlab_y0, width, height, stroke=0, fill=1)
                    print(f"Rendered {element['type']} (vector fallback) at ({x0:.1f}, {reportlab_y0:.1f}, {width:.1f}x{height:.1f})")
        
        except Exception as e:
            print(f"Error rendering visual element: {e}")
            # Draw error placeholder
            canvas.setStrokeColorRGB(1, 0, 0)
            canvas.setFillColorRGB(1, 0.8, 0.8)
            canvas.rect(x0, reportlab_y0, width, height, stroke=1, fill=1)

    def render_qr_code_from_text(self, canvas, qr_text: str, bbox: tuple, page_height: float):
        """
        Render an actual QR code from the detected QR text content.
        """
        try:
            import qrcode
            from PIL import Image
            import io
            from reportlab.lib.utils import ImageReader
            
            x0, y0, x1, y1 = bbox
            width = x1 - x0
            height = y1 - y0
            reportlab_y0 = page_height - y1
            
            # Extract meaningful data from QR text if possible
            # For now, we'll create a QR code with placeholder data
            # In a real implementation, you'd decode the visual QR pattern
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

    def create_translated_pdf_reportlab_with_visuals(self, original_pdf_path: str, blocks_with_translations: list, visual_elements: list, output_pdf_path: str):
        """
        Generate a translated PDF using ReportLab with Unicode font support and visual elements.
        Each text block is rendered at its bounding box with the appropriate font size.
        Visual elements (logos, images, QR codes) are preserved in their original positions.
        """
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
    from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.utils import simpleSplit
    import os
        import fitz
        
        debug = False  # Enable debug output

        # Register Unicode fonts with style variants
        font_name = "Helvetica"  # Force Helvetica for better compatibility
        
        # If font registration fails, fall back to Helvetica
        if font_name not in pdfmetrics.getRegisteredFontNames():
            font_name = "Helvetica"

        # Get page size from original PDF
        doc = fitz.open(original_pdf_path)
        page_sizes = [ (page.rect.width, page.rect.height) for page in doc ]
        doc.close()

        c = canvas.Canvas(output_pdf_path, pagesize=letter)
        for page_num, (page_width, page_height) in enumerate(page_sizes):
            c.setPageSize((page_width, page_height))
            
            # First, render all visual elements for this page
            print(f"Rendering visual elements for page {page_num}...")
            page_visual_elements = [elem for elem in visual_elements if elem.get("page", 0) == page_num]
            for elem in page_visual_elements:
                self.render_visual_element(c, elem, page_height)
            
            # Then, draw all text blocks for this page
            print(f"Rendering text blocks for page {page_num}...")
            for block, blk_page_num, translated_text in blocks_with_translations:
                if blk_page_num != page_num:
                    continue
                
                # Skip QR codes - they're handled as visual elements now
                if block.type == "qr_code":
                    # Try to render actual QR code from the text content
                    qr_rendered = self.render_qr_code_from_text(c, block.text, block.bbox, page_height)
                    
                    if not qr_rendered:
                        # Fallback to placeholder if QR generation fails
                        qr_bbox = block.bbox
                        found_visual_qr = False
                        for elem in page_visual_elements:
                            elem_bbox = elem["bbox"]
                            # Check if bboxes overlap significantly
                            if self._bboxes_overlap(qr_bbox, elem_bbox, threshold=0.5):
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
                            c.setFont(font_name, 8)
                            c.drawString(x0 + 5, reportlab_y0 + 5, "[QR Code]")
                    continue
                
                x0, y0, x1, y1 = block.bbox
                font_size = max(5, int(block.size) if block.size else 12)
                processed_text = self._postprocess_english(translated_text, block.text)
                
                # Check if this text block overlaps with any visual element
                text_bbox = (x0, y0, x1, y1)
                overlaps_with_visual = False
                for elem in page_visual_elements:
                    if self._bboxes_overlap(text_bbox, elem["bbox"], threshold=0.3):
                        overlaps_with_visual = True
                        if debug:
                            print(f"Text block overlaps with {elem['type']}, skipping text rendering")
                        break
                
                # Only render text if it doesn't significantly overlap with visual elements
                if not overlaps_with_visual:
                    # Get font style from original block
                    try:
                        font_style = self._get_font_style_from_block(block)
                        actual_font_name = self._select_appropriate_font(font_name, font_style['bold'], font_style['italic'], block.font)
                        
                        if debug:
                            print(f"Block font: {block.font}, Bold: {font_style['bold']}, Italic: {font_style['italic']}, Selected font: {actual_font_name}")
            except Exception as e:
                        if debug:
                            print(f"Error in font style processing: {e}")
                        actual_font_name = font_name
                        font_style = {'bold': False, 'italic': False}
                    
                    # Smart reflow approach - respect original block dimensions
                    original_width = x1 - x0
                    original_height = y1 - y0
                    expansion_needed = False
                    
                    # First, try to reflow text within original block dimensions
                    original_max_width = original_width * 0.98  # Use 98% of original width
                    
                    # Let text reflow naturally within original dimensions
                    try:
                        lines = simpleSplit(processed_text, actual_font_name, font_size, original_max_width)
                    except Exception as e:
                        if debug:
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
                            expanded_width = self._calculate_expanded_width(block, blocks_with_translations, page_num, page_width)
                            expanded_max_width = expanded_width * 0.98
                            
                            # Try reflowing with expanded width
                            try:
                                expanded_lines = simpleSplit(processed_text, actual_font_name, font_size, expanded_max_width)
                            except Exception as e:
                                if debug:
                                    print(f"Error in expanded text splitting: {e}")
                                expanded_lines = lines  # Use original lines as fallback
                            expanded_required_height = len(expanded_lines) * line_height
                            
                            if expanded_required_height <= original_height * 1.2:  # Allow 20% height expansion
                                # Expansion helps - use expanded width
                                lines = expanded_lines
                                expansion_needed = True
                                actual_max_width = expanded_max_width
                            else:
                                # Even expansion doesn't help much - use original width but allow more height
                                expansion_needed = False
                                actual_max_width = original_max_width
                                # Keep the original reflow
                                
                        except:
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
                            c.setFont(actual_font_name, font_size)
                            c.setFillColorRGB(0, 0, 0)
                            # Use original positioning - expansion is handled in the text wrapping logic above
                            c.drawString(x0 + 3, current_y, line)
                        current_y -= line_height  # Move down in ReportLab coordinates (subtract)
                else:
                    if debug:
                        print(f"Skipped text block due to visual element overlap")
            
            c.showPage()
        c.save()
        print(f"[ReportLab] Translated PDF with visual elements saved to: {output_pdf_path}")

    def _bboxes_overlap(self, bbox1: tuple, bbox2: tuple, threshold: float = 0.3) -> bool:
        """
        Check if two bounding boxes overlap by more than the threshold percentage.
        """
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

    def _format_urls_and_web_addresses(self, text: str) -> str:
        """
        Detect and properly format URLs, web addresses, and email addresses.
        Ensures proper capitalization like 'www' instead of 'Www' or 'WWW'.
        """
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
        
        # Fix common web-related terms (but preserve sentence capitalization)
        text = re.sub(r'(?<!^)(?<![.!?]\s)\bwebsite\b', 'website', text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!^)(?<![.!?]\s)\bemail\b', 'email', text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!^)(?<![.!?]\s)\be-mail\b', 'email', text, flags=re.IGNORECASE)
        
        return text

    def _fix_translation_issues(self, translated: str, original: str) -> str:
        """Fix common translation issues"""
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
        
        # Fix specific translation errors that Google Translate makes
        translated = re.sub(r'\bSurrender deductible\b', 'DEDUCTIBLE BUYOUT', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bAfkoop eigen risico\b', 'DEDUCTIBLE BUYOUT', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bSurrender own risk\b', 'DEDUCTIBLE BUYOUT', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bStuberty\b', 'Unit Price', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bNumber\b', 'Quantity', translated, flags=re.IGNORECASE)
        translated = re.sub(r'\bPiece price\b', 'Unit Price', translated, flags=re.IGNORECASE)
        
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
        
        # Apply improved casing preservation BEFORE the specific fixes
        translated = self._preserve_original_casing(original, translated)
        
        # Apply post-casing fixes to handle remaining issues
        translated = self._apply_post_casing_fixes(translated)
        
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
        
        return translated.strip()

def main():
    """Main function for command-line usage"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Translate PDF documents while preserving layout')
    parser.add_argument('input_pdf', help='Input PDF file path')
    parser.add_argument('-o', '--output', help='Output PDF file path (default: input_translated.pdf)')
    parser.add_argument('--target-lang', default='en', help='Target language code (default: en)')
    parser.add_argument('--no-translate', action='store_true', help='Skip translation, only process layout')
    
    args = parser.parse_args()
    
    if not args.input_pdf.endswith('.pdf'):
        print("Error: Input file must be a PDF")
        sys.exit(1)
    
    output_path = args.output or args.input_pdf.replace('.pdf', '_translated.pdf')
    
    try:
        pdf_parser = SimplePDFLayoutParser()
        pdf_parser.process_pdf(
            args.input_pdf, 
            output_path, 
            translate=not args.no_translate,
            target_lang=args.target_lang
        )
        print(f"Successfully processed: {args.input_pdf} -> {output_path}")
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Only run the main PDF processing pipeline
    exit(main()) 