#!/usr/bin/env python3

"""
Simplified Layout Parser for PDF Processing
Uses PyMuPDF for basic layout detection without requiring detectron2
"""

import fitz  # PyMuPDF
import json
import re
from typing import List, Dict, Any, Tuple
from googletrans import Translator
from fuzzywuzzy import fuzz
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
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        assert self.api_key, "DEEPSEEK_API_KEY not found in environment!"
    
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
        """Translate text using Deepseek API with post-processing improvements"""
        if not text.strip():
            return text
        try:
            preprocessed_text = self._preprocess_for_translation(text)
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            prompt = (
                f"Translate the following text to {target_lang}. Only provide the translation, no explanations or additional text:\n\n"
                f"{preprocessed_text}\n\nTranslation:"
            )
            data = {
                "model": "deepseek-chat",
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
                                # Insert as Unicode string, using the Unicode font
                                page.insert_text((x0 + 3, current_y), str(line), fontsize=font_size, color=(0, 0, 0), fontname=font_name)
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
    
    def deepseek_translate_batch(self, texts, target_lang='en', batch_size=10):
        import re
        import requests
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('DEEPSEEK_API_KEY')
        assert api_key, "DEEPSEEK_API_KEY not found in environment!"
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
                        url = "https://api.deepseek.com/v1/chat/completions"
                        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                        data = {
                            "model": "deepseek-chat",
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.2
                        }
                        response = requests.post(url, headers=headers, json=data)
                        print(f"Deepseek API response for long sentence batch: {response.text}")
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
                            print(f"Deepseek API error: {response.status_code} {response.text}")
                            translated_sentences.extend(batch)
                    # Join sentences for this paragraph, preserving paragraph breaks
                    translated_paragraphs.append(' '.join(translated_sentences))
                else:
                    # Normal paragraph translation
                    prompt = (
                        f"Translate the following text to {target_lang}. Return ONLY the translation. Do not include explanations.\n\n"
                        f"Text to translate:\n{para}"
                    )
                    url = "https://api.deepseek.com/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    data = {
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.2
                    }
                    response = requests.post(url, headers=headers, json=data)
                    print(f"Deepseek API response for paragraph: {response.text}")
                    if response.status_code == 200:
                        content = response.json()["choices"][0]["message"]["content"]
                        translated_paragraphs.append(content.strip())
                    else:
                        print(f"Deepseek API error: {response.status_code} {response.text}")
                        translated_paragraphs.append(para)
            # Join translated paragraphs with double newlines to preserve paragraph breaks
            all_translations.append('\n\n'.join(translated_paragraphs))
        return all_translations
    
    def process_pdf(self, pdf_path: str, output_path: str = None, translate: bool = True, target_lang: str = 'en'):
        print(f"Processing PDF: {pdf_path}")
        # Extract blocks with page numbers
        blocks_with_pages = self.extract_blocks_from_pdf(pdf_path)
        print(f"Extracted {len(blocks_with_pages)} initial blocks")
        # Remove duplicates
        unique_blocks_with_pages = self.remove_duplicates(blocks_with_pages)
        print(f"After deduplication: {len(unique_blocks_with_pages)} blocks")
        # Prepare results and translation data
        blocks_with_translations = []
        texts = [block.text for block, _ in unique_blocks_with_pages]
        if translate:
            translations = self.deepseek_translate_batch(texts, target_lang=target_lang)
            print(f"Translations received: {len(translations)}")
        else:
            translations = texts
        for idx, (block, page_num) in enumerate(unique_blocks_with_pages):
            translated_text = translations[idx]
            blocks_with_translations.append((block, page_num, translated_text))
            print(f"Block {idx}:")
            print(f"  Original: {block.text}")
            print(f"  Translated: {translated_text}")
        # Always create output PDF with translated text
        if output_path and output_path.endswith('.pdf'):
            output_pdf_path = output_path
        else:
            output_pdf_path = pdf_path.replace('.pdf', '_translated.pdf')
        print(f"Creating translated PDF: {output_pdf_path}")
        self.create_translated_pdf(pdf_path, blocks_with_translations, output_pdf_path)
        # Also create a Unicode-compatible PDF using ReportLab
        reportlab_output = output_pdf_path.replace('.pdf', '_unicode.pdf')
        self.create_translated_pdf_reportlab(pdf_path, blocks_with_translations, reportlab_output)
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
        """Improve English punctuation and capitalization, matching original casing where possible."""
        import re
        # Capitalize first letter of each sentence
        def capitalize_sentences(s):
            s = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), s)
            s = s[:1].upper() + s[1:] if s else s
            return s
        # Ensure proper punctuation at end of sentences
        s = text.strip()
        if s and s[-1] not in '.!?':
            s += '.'
        s = capitalize_sentences(s)
        # Optionally, match original casing for acronyms, etc.
        # (For now, just basic sentence capitalization)
        return s

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

        # Try to register DejaVuSans (Unicode font)
        font_name = "DejaVuSans"
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        if not os.path.exists(font_path):
            # Try a common Mac path
            font_path = "/Library/Fonts/Arial Unicode.ttf"
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_path))
        except Exception as e:
            print(f"Warning: Could not register Unicode font. Falling back to Helvetica. Error: {e}")
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
                x0, y0, x1, y1 = block.bbox
                font_size = max(5, int(block.size) if block.size else 12)
                # Wrap text to fit block width
                max_width = x1 - x0 - 6
                processed_text = self._postprocess_english(translated_text, block.text)
                lines = simpleSplit(processed_text, font_name, font_size, max_width)
                line_height = font_size * 1.15
                start_y = y0 + font_size + 2
                current_y = start_y
                for line in lines:
                    if current_y < y1 + 3:
                        c.setFont(font_name, font_size)
                        c.setFillColorRGB(0, 0, 0)
                        c.drawString(x0 + 3, current_y, line)
                    current_y += line_height
            c.showPage()
        c.save()
        print(f"[ReportLab] Translated PDF saved to: {output_pdf_path}")

def save_pdf_page_as_image(pdf_path, page_num=0, out_path="page.png"):
    import fitz
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=150)
    pix.save(out_path)
    doc.close()
    print(f"Saved page {page_num} as {out_path}")

def main():
    parser = argparse.ArgumentParser(description='Simple PDF Layout Parser with PDF Output')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--no-translate', action='store_true', help='Skip translation')
    parser.add_argument('--lang', default='en', help='Target language for translation (default: en)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found.")
        return 1
    
    # Create parser instance
    layout_parser = SimplePDFLayoutParser()
    
    # Set output path if not provided
    if not args.output:
        args.output = args.pdf_path.replace('.pdf', '_layout_simple.json')
    
    try:
        # Process PDF
        layout_parser.process_pdf(
            pdf_path=args.pdf_path,
            output_path=args.output,
            translate=not args.no_translate,
            target_lang=args.lang
        )
        
        # Print summary
        print(f"\nProcessing complete!")
        print(f"Total blocks processed: {len(layout_parser.extract_blocks_from_pdf(args.pdf_path))}")
        print(f"Translated PDF created: {args.output}")
        
        # Show block type distribution
        type_counts = {}
        for block, _ in layout_parser.extract_blocks_from_pdf(args.pdf_path):
            block_type = block.type
            type_counts[block_type] = type_counts.get(block_type, 0) + 1
        
        print("\nBlock type distribution:")
        for block_type, count in sorted(type_counts.items()):
            print(f"  {block_type}: {count}")
        
        return 0
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return 1

if __name__ == "__main__":
    # Only run the main PDF processing pipeline
    exit(main()) 