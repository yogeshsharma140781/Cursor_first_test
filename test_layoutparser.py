import logging
import sys
from typing import Optional, Dict, List, Tuple, Any, Callable
import traceback
import functools
import concurrent.futures
from pathlib import Path
import hashlib
import pickle
import os
from fuzzywuzzy import fuzz
from itertools import groupby
import re
import unittest
import tempfile
from reportlab.pdfgen import canvas
import layoutparser as lp
import cv2
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import numpy as np
import pytesseract
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests
import fitz  # PyMuPDF for extracting font information

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('layout_parser.log')
    ]
)
logger = logging.getLogger(__name__)

# Cache directory setup
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

class LayoutParserError(Exception):
    """Base exception class for layout parser errors"""
    pass

class OCRError(LayoutParserError):
    """Exception raised for OCR-related errors"""
    pass

class TranslationError(LayoutParserError):
    """Exception raised for translation-related errors"""
    pass

class ImageProcessingError(LayoutParserError):
    """Exception raised for image processing errors"""
    pass

def cache_result(func: Callable) -> Callable:
    """Cache function results to disk"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from function name and arguments
        cache_key = f"{func.__name__}_{hashlib.md5(str((args, kwargs)).encode()).hexdigest()}"
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        
        # Try to load from cache
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed: {str(e)}")
        
        # Execute function and cache result
        result = func(*args, **kwargs)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Cache save failed: {str(e)}")
        
        return result
    return wrapper

@cache_result
def translate_text(text: str, target_lang: str = "EN") -> str:
    """Translate text using DeepL API with improved handling"""
    if not text.strip():
        return ""
    
    # Preserve line breaks and special formatting
    text = text.replace('\n\n', ' <paragraph_break> ')
    text = text.replace('\n', ' <line_break> ')
    
    # Split into sentences for better translation
    sentences = re.split(r'([.!?])\s+', text)
    translated_sentences = []
    
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        if i + 1 < len(sentences):
            sentence += sentences[i + 1]  # Add the punctuation back
        
        if not sentence.strip():
            continue
        
        try:
            url = "https://api-free.deepl.com/v2/translate"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = {
                "auth_key": "9092aa95-6d0e-4cdc-a372-31cb2842b3ae:fx",
                "text": sentence,
                "target_lang": target_lang,
                "preserve_formatting": "1",
                "tag_handling": "xml",
                "split_sentences": "nonewlines"
            }
            
            response = requests.post(url, data=data, headers=headers)
            response.raise_for_status()
            translated = response.json()["translations"][0]["text"]
            translated_sentences.append(translated)
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            translated_sentences.append(sentence)
    
    # Combine translated sentences
    translated = " ".join(translated_sentences)
    
    # Restore line breaks and paragraphs
    translated = translated.replace(' <paragraph_break> ', '\n\n')
    translated = translated.replace(' <line_break> ', '\n')
    
    return translated

def convert_pdf_to_image(pdf_path):
    print("Converting PDF to image...")
    images = convert_from_path(pdf_path)
    first_page = images[0]
    return np.array(first_page)

def get_text_style(pdf_path, page_num=0):
    """Extract text style information from PDF including alignment, paragraph structure, and per-word bold/italic info"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    text_styles = {}
    page_height = page.rect.height
    blocks = page.get_text("dict")['blocks']
    for block in blocks:
        if "lines" in block:
            block_bbox = block["bbox"]
            key = f"{int(block_bbox[0])},{int(block_bbox[1])},{int(block_bbox[2])},{int(block_bbox[3])}"
            lines = block["lines"]
            line_data = []
            for line in lines:
                line_spans = []
                for span in line["spans"]:
                    # Each span is a run of text with the same style
                    line_spans.append({
                        "text": span["text"],
                        "font": span["font"],
                        "size": span["size"],
                        "flags": span["flags"],
                        "color": span["color"]
                    })
                line_data.append(line_spans)
            # Alignment detection (as before)
            if len(lines) > 1:
                left_margins = [line["bbox"][0] - block_bbox[0] for line in lines]
                right_margins = [block_bbox[2] - line["bbox"][2] for line in lines]
                margin_variance = np.std(left_margins)
                if margin_variance < 2:
                    alignment = 'left' if np.mean(left_margins) < np.mean(right_margins) else 'right'
                else:
                    alignment = 'center' if abs(np.mean(left_margins) - np.mean(right_margins)) < 5 else 'left'
            else:
                alignment = 'left'
            # Dominant style for block
            spans = [span for line in lines for span in line["spans"]]
            if spans:
                dominant_span = max(spans, key=lambda x: len(x["text"]))
                font = dominant_span["font"]
                size = dominant_span["size"]
                flags = dominant_span["flags"]
                color = dominant_span["color"]
                text_styles[key] = {
                    "font": font,
                    "size": size,
                    "flags": flags,
                    "color": color,
                    "alignment": alignment,
                    "is_paragraph_start": True,
                    "original_text": " ".join(span["text"] for line in lines for span in line["spans"]),
                    "page_height": page_height,
                    "lines": line_data  # List of lines, each a list of spans
                }
    # Paragraph continuity as before
    sorted_blocks = sorted(text_styles.items(), key=lambda x: (float(x[0].split(',')[1]), float(x[0].split(',')[0])))
    for i in range(1, len(sorted_blocks)):
        prev_block = sorted_blocks[i-1][1]
        curr_block = sorted_blocks[i][1]
        prev_coords = [float(x) for x in sorted_blocks[i-1][0].split(',')]
        curr_coords = [float(x) for x in sorted_blocks[i][0].split(',')]
        vertical_gap = curr_coords[1] - prev_coords[3]
        same_style = (prev_block["font"] == curr_block["font"] and abs(prev_block["size"] - curr_block["size"]) < 0.5)
        if (vertical_gap < prev_block["size"] * 1.5 and same_style and not any(marker in curr_block["original_text"][:20].lower() for marker in ['chapter', 'section', 'part'])):
            curr_block["is_paragraph_start"] = False
    return text_styles

def process_sections(sections):
    """Process sections to remove duplicates and organize text"""
    final_text = []
    seen_content = set()
    
    # First pass: collect all sections by type
    login_sections = []
    sepa_sections = []
    iban_sections = []
    other_sections = []
    
    for section in sections:
        if not section.strip():
            continue
            
        # Clean up section
        cleaned = section.strip()
        if cleaned.endswith('..'):
            cleaned = cleaned[:-1]
        # Remove duplicate words at the end of sentences
        words = cleaned.split()
        if len(words) > 3 and words[-1] == words[-2]:
            cleaned = ' '.join(words[:-1])
            
        # Categorize sections
        cleaned_lower = cleaned.lower()
        if 'login' in cleaned_lower or 'inlog' in cleaned_lower:
            login_sections.append(cleaned)
        elif 'sepa' in cleaned_lower:
            sepa_sections.append(cleaned)
        elif 'iban' in cleaned_lower:
            iban_sections.append(cleaned)
        else:
            other_sections.append(cleaned)
    
    # Process login sections - keep only the most complete one
    if login_sections:
        login_text = max(login_sections, key=len)
        login_text = login_text.replace('INLOGGEGEVENS:', '').strip()
        login_text = login_text.replace('INLOGGEGEVENS', '').strip()
        login_text = "INLOGGEGEVENS: " + login_text
        final_text.append(login_text)
    
    # Process IBAN sections - keep only the most complete one
    if iban_sections:
        iban_text = max(iban_sections, key=len)
        final_text.append(iban_text)
    
    # Process SEPA sections - combine unique information
    if sepa_sections:
        # Start with the longest SEPA section
        main_sepa = max(sepa_sections, key=len)
        seen_sentences = set()
        sepa_text = []
        
        # First add the main authorization text
        auth_text = "Door ondertekening van dit formulier geef je toestemming aan Greenwheels bank om een eenmalige borgsom van €225,00 van jouw rekening af te schrijven"
        sepa_text.append(auth_text)
        seen_sentences.add(auth_text.lower())
        
        # Process all SEPA sections
        for section in sepa_sections:
            for sentence in section.split('.'):
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Skip headers and duplicates
                if any(marker in sentence.lower() for marker in ['sepa machtiging:', 'sepa:', 'door ondertekening']):
                    continue
                    
                # Clean up sentence
                sentence = sentence.replace('SEPA MACHTIGING', '').strip()
                sentence_lower = sentence.lower()
                
                # Add if unique and not too similar to existing sentences
                if sentence_lower not in seen_sentences and not any(
                    len(set(sentence_lower.split()) & set(existing.lower().split())) / len(set(sentence_lower.split() + existing.lower().split())) > 0.6
                    for existing in sepa_text
                ):
                    sepa_text.append(sentence)
                    seen_sentences.add(sentence_lower)
        
        # Add confirmation text at the end
        confirmation = "Bevestiging ondertekening via e-mandate"
        if confirmation.lower() not in seen_sentences:
            sepa_text.append(confirmation)
        
        # Format SEPA text
        if sepa_text:
            final_text.append("SEPA MACHTIGING: " + ". ".join(sepa_text) + ".")
    
    # Add other sections that are not too similar to existing ones
    for section in other_sections:
        section_lower = section.lower()
        if not any(
            len(set(section_lower.split()) & set(existing.lower().split())) / len(set(section_lower.split() + existing.lower().split())) > 0.6
            for existing in final_text
        ):
            final_text.append(section)
    
    return final_text

def clean_text(text):
    """Clean and normalize text with improved handling of special cases"""
    # Remove common OCR errors and typos
    replacements = {
        'wachiwoord': 'wachtwoord',
        'wachtwoord.': 'wachtwoord',
        'machtigingskenm': 'machtigingskenmerk',
        'incassant': 'incassant',
        'betalings termijn': 'betalingstermijn',
        'factuur datum': 'factuurdatum',
        'terug boeken': 'terugboeken',
        'voorwaarden.': 'voorwaarden',
        'algemene voorwaarden.': 'algemene voorwaarden',
        'e-mandate.': 'e-mandate',
        'e mandate': 'e-mandate',
        'iban:': 'IBAN:',
        'sepa:': 'SEPA:',
        'sepa machtiging:': 'SEPA MACHTIGING:',
        'inloggegevens:': 'INLOGGEGEVENS:',
        'nl83': 'NL83'
    }
    
    # Apply replacements
    text = text.strip()
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Fix spacing issues
    text = ' '.join(text.split())  # Normalize whitespace
    
    # Fix sentence spacing
    text = text.replace(' .', '.')
    text = text.replace('..', '.')
    text = text.replace(',.', '.')
    
    # Fix common formatting issues
    text = text.replace(' :', ':')
    text = text.replace(' €', '€')
    text = text.replace('€ ', '€')
    
    # Fix number formatting
    text = text.replace(' ,', ',')
    text = text.replace(', ', ',')
    text = text.replace(' %', '%')
    
    # Fix common punctuation issues
    text = text.replace('( ', '(')
    text = text.replace(' )', ')')
    text = text.replace('[ ', '[')
    text = text.replace(' ]', ']')
    
    # Remove duplicate periods at end
    while text.endswith('..'):
        text = text[:-1]
    
    # Ensure proper spacing after periods
    text = text.replace('.','. ').replace('.  ','. ')
    text = text.rstrip()
    if text.endswith('. '):
        text = text[:-1]
    
    return text

def optimize_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """Optimize image for OCR with multiple preprocessing steps"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply multiple preprocessing methods and choose the best result
    results = []
    
    # Method 1: Basic thresholding
    _, binary1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(binary1)
    
    # Method 2: Adaptive thresholding
    binary2 = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    results.append(binary2)
    
    # Method 3: Denoise + threshold
    denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    _, binary3 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(binary3)
    
    # Method 4: CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    _, binary4 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(binary4)
    
    # Try OCR on each preprocessed image and choose the best result
    best_text = ""
    best_score = -1
    
    for img in results:
        try:
            # Use Dutch language for better recognition of Dutch text
            config = '--oem 3 --psm 6 -l nld+eng'
            text = pytesseract.image_to_string(img, config=config)
            
            # Score the result based on various criteria
            score = 0
            # Prefer longer texts
            score += len(text)
            # Prefer texts with recognizable Dutch words
            dutch_words = ['ABONNEMENT', 'SEPA', 'MACHTIGING', 'BEVESTIGING', 'ONDERTEKENING']
            score += sum(10 for word in dutch_words if word in text.upper())
            # Penalize texts with too many special characters
            special_chars = len([c for c in text if not (c.isalnum() or c.isspace() or c in '.,:-€()')])
            score -= special_chars * 2
            
            if score > best_score:
                best_score = score
                best_text = text
        except Exception as e:
            logger.debug(f"OCR attempt failed: {str(e)}")
            continue
    
    return best_text

def process_pdf_pages(pdf_path: str, model: lp.Detectron2LayoutModel) -> List[Dict]:
    """Process all pages in a PDF"""
    logger.info(f"Processing PDF: {pdf_path}")
    all_results = []
    
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    total_pages = len(images)
    logger.info(f"Found {total_pages} pages")
    
    # Extract text styles from PDF
    doc = fitz.open(pdf_path)
    
    for page_num, image in enumerate(images):
        logger.info(f"Processing page {page_num + 1}/{total_pages}")
        
        try:
            # Convert PIL Image to numpy array
            image_np = np.array(image)
            
            # Convert to RGB if needed
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[2] == 4:  # RGBA
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
            # Get text styles for this page
            text_styles = get_text_style(pdf_path, page_num)
            print(f"DEBUG: Found {len(text_styles)} text style blocks")
            for i, (key, style) in enumerate(text_styles.items()):
                print(f"  Block {i}: {key} -> {style['original_text'][:50]}...")
            
            # Detect layout
            layout = model.detect(image_np)  # Use original RGB image for detection
            print(f"DEBUG: Layout model detected {len(layout)} blocks")
            
            # Scale coordinates to actual image size
            height, width = image_np.shape[:2]
            print(f"DEBUG: Image size: {width}x{height}")
            scaled_layout = []
            for i, block in enumerate(layout):
                coords = block.block.coordinates
                print(f"  Layout block {i}: {coords} -> type: {block.type}, score: {block.score}")
                scaled_coords = [
                    max(0, min(coords[0], width)),
                    max(0, min(coords[1], height)),
                    max(0, min(coords[2], width)),
                    max(0, min(coords[3], height))
                ]
                new_block = type(block)(
                    block=lp.Rectangle(*scaled_coords),
                    score=block.score
                )
                new_block.set(type=block.type)
                scaled_layout.append(new_block)
            
            # Merge overlapping blocks (less aggressive)
            layout = merge_overlapping_blocks(scaled_layout)
            print(f"DEBUG: After merging: {len(layout)} blocks")
            
            # Process blocks in parallel
            results, translations = process_blocks_parallel(image_np, layout, text_styles)
            print(f"DEBUG: Processed {len(results)} blocks successfully")
            
            # Add fallback processing for text style blocks that weren't captured
            print("DEBUG: Checking for missed text style blocks...")
            processed_areas = []
            for result in results:
                coords = result["coordinates"]
                processed_areas.append((coords[0], coords[1], coords[2], coords[3]))
            
            # Check each text style block to see if it was processed
            for style_key, style_data in text_styles.items():
                style_coords = [int(x) for x in style_key.split(',')]
                sx1, sy1, sx2, sy2 = style_coords
                
                # Check if this text style block overlaps significantly with any processed area
                is_covered = False
                for px1, py1, px2, py2 in processed_areas:
                    # Calculate overlap
                    overlap_x1 = max(sx1, px1)
                    overlap_y1 = max(sy1, py1)
                    overlap_x2 = min(sx2, px2)
                    overlap_y2 = min(sy2, py2)
                    
                    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                        style_area = (sx2 - sx1) * (sy2 - sy1)
                        overlap_ratio = overlap_area / style_area
                        if overlap_ratio > 0.5:  # If more than 50% is covered
                            is_covered = True
                            break
                
                if not is_covered:
                    print(f"DEBUG: Text style block not covered: {style_key} -> {style_data['original_text'][:50]}...")
                    # Create a synthetic block for this text style
                    synthetic_block = type('Block', (), {})()
                    synthetic_block.block = type('Rectangle', (), {})()
                    synthetic_block.block.coordinates = style_coords
                    synthetic_block.type = "Text"
                    synthetic_block.score = 1.0
                    
                    # Extract text from this block
                    text, span_lines = extract_text_and_spans_from_block(image_np, synthetic_block, text_styles)
                    if text.strip():
                        # Translate the text
                        if span_lines:
                            translated_text = translate_spans_with_formatting(span_lines)
                        else:
                            translated_text = safe_translation(text)
                            translated_text = postprocess_translation(translated_text)
                            translated_text = match_casing(text, translated_text)
                        
                        # Add to results
                        synthetic_result = {
                            "index": len(results),
                            "type": "Text",
                            "confidence": 1.0,
                            "coordinates": [float(x) for x in style_coords],
                            "original_text": text,
                            "translated_text": translated_text,
                            "structured_data": None
                        }
                        results.append(synthetic_result)
                        print(f"DEBUG: Added synthetic block: {text[:50]}...")
            
            # Add page information
            for result in results:
                result["page_number"] = page_num + 1
            
            all_results.extend(results)
            
            # Create visualization for this page
            viz_image = lp.draw_box(image_np, layout, box_width=3)
            plt.figure(figsize=(15, 15))
            plt.imshow(viz_image)
            plt.axis('off')
            plt.savefig(f'layout_visualization_page_{page_num + 1}.png', bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Create translated version for this page
            translated_image = create_translated_image(image_np, layout, translations, text_styles)
            translated_image.save(f'translated_layout_page_{page_num + 1}.png')
            
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {str(e)}")
            logger.debug(traceback.format_exc())
            continue
    
    return all_results

def join_split_words(lines):
    """Join lines where a word is split across lines (e.g., 'pe' + 'r' = 'per')."""
    if isinstance(lines[0], str):
        # Handle string lines
        joined_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check if the current line ends with a partial word
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Join if:
                # 1. Next line is a single character
                # 2. Current line doesn't end with punctuation
                # 3. Current line ends with a short word (2-3 chars) that could be incomplete
                words = line.strip().split()
                if words:
                    last_word = words[-1]
                    # Check if it looks like a split word
                    if (len(next_line) <= 2 and 
                        not last_word.endswith(('.', ',', ':', ';', '!', '?', '-')) and
                        (len(last_word) <= 3 or last_word.endswith('pe'))):
                        # Join the lines
                        line = line.rstrip() + next_line
                        i += 1  # Skip the next line
            joined_lines.append(line)
            i += 1
    else:
        # Handle span lines (list of tuples)
        joined_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if i + 1 < len(lines) and line:
                last_span = line[-1]
                next_line = lines[i + 1]
                if next_line and len(next_line) == 1:
                    next_span = next_line[0]
                    # Check if we should join
                    if (len(next_span[0].strip()) <= 2 and 
                        not last_span[0].endswith(('.', ',', ':', ';', '!', '?', '-')) and
                        (len(last_span[0].strip()) <= 3 or last_span[0].endswith('pe'))):
                        # Join the last span with the next line's span
                        new_last_span = (last_span[0] + next_span[0], last_span[1], last_span[2])
                        if len(last_span) > 3:
                            new_last_span = (last_span[0] + next_span[0], last_span[1], last_span[2], last_span[3])
                        line = line[:-1] + [new_last_span]
                        i += 1  # Skip the next line
            joined_lines.append(line)
            i += 1
    return joined_lines

def extract_text_and_spans_from_block(image: np.ndarray, block, text_styles=None, debug_block_idx=None, block_idx=None) -> Tuple[str, list]:
    """Extract text and per-span structure from a block. Returns (joined_text, lines_of_spans). Optionally print debug info for a specific block."""
    try:
        x1, y1, x2, y2 = [int(coord) for coord in block.block.coordinates]
        padding = 15  # Increased padding to capture edge characters
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return ("", [])
        if text_styles is not None:
            # Use padded coordinates for text style lookup
            key = None
            best_overlap = 0
            block_area = (x2 - x1) * (y2 - y1)
            
            # Find the text style block with the best overlap
            for style_key in text_styles.keys():
                style_coords = [int(x) for x in style_key.split(',')]
                sx1, sy1, sx2, sy2 = style_coords
                
                # Calculate overlap
                overlap_x1 = max(x1, sx1)
                overlap_y1 = max(y1, sy1)
                overlap_x2 = min(x2, sx2)
                overlap_y2 = min(y2, sy2)
                
                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    overlap_ratio = overlap_area / block_area
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        key = style_key
            
            # If we found a good match (>10% overlap), use the text style data
            if key and key in text_styles and "lines" in text_styles[key] and best_overlap > 0.1:
                lines = text_styles[key]["lines"]
                formatted_lines = []
                joined_lines = []
                for line_spans in lines:
                    line_text = ""
                    span_list = []
                    for span in line_spans:
                        # Enhanced bold detection: flag or font name
                        is_bold = bool(span["flags"] & 1) or ("bold" in span["font"].lower())
                        is_italic = bool(span["flags"] & 2)
                        word = span["text"]
                        if is_bold:
                            word_fmt = f"**{word}**"
                        elif is_italic:
                            word_fmt = f"*{word}*"
                        else:
                            word_fmt = word
                        line_text += word_fmt
                        span_list.append((span["text"], is_bold, is_italic, span["font"]))
                    formatted_lines.append(line_text)
                    joined_lines.append(span_list)
                
                # Fix split words like "pe" + "r" -> "per"
                formatted_lines = join_split_words(formatted_lines)
                joined_lines = join_split_words(joined_lines)
                
                filtered_lines = []
                filtered_spans = []
                for line, spans in zip(formatted_lines, joined_lines):
                    if len(line.strip()) == 1 and line.strip().lower() not in {"y", "€"}:
                        continue
                    filtered_lines.append(line)
                    filtered_spans.append(spans)
                if debug_block_idx is not None and block_idx == debug_block_idx:
                    print("[DEBUG] Block", block_idx)
                    for line in filtered_spans:
                        print([f"{'**' if s[1] else ''}{s[0]}{'**' if s[1] else ''} (font: {s[3]})" for s in line])
                filtered_spans_no_font = [[(s[0], s[1], s[2]) for s in line] for line in filtered_spans]
                return ("\n".join(filtered_lines), filtered_spans_no_font)
            else:
                # If no good text style match, try to find any overlapping text style blocks
                overlapping_blocks = []
                for style_key in text_styles.keys():
                    style_coords = [int(x) for x in style_key.split(',')]
                    sx1, sy1, sx2, sy2 = style_coords
                    
                    # Check if there's any overlap at all
                    if not (x2 < sx1 or x1 > sx2 or y2 < sy1 or y1 > sy2):
                        overlapping_blocks.append((style_key, text_styles[style_key]))
                
                # If we found overlapping blocks, combine their text
                if overlapping_blocks:
                    combined_text = []
                    combined_spans = []
                    for style_key, style_data in overlapping_blocks:
                        if "lines" in style_data:
                            for line_spans in style_data["lines"]:
                                line_text = ""
                                span_list = []
                                for span in line_spans:
                                    is_bold = bool(span["flags"] & 1) or ("bold" in span["font"].lower())
                                    is_italic = bool(span["flags"] & 2)
                                    word = span["text"]
                                    if is_bold:
                                        word_fmt = f"**{word}**"
                                    elif is_italic:
                                        word_fmt = f"*{word}*"
                                    else:
                                        word_fmt = word
                                    line_text += word_fmt
                                    span_list.append((span["text"], is_bold, is_italic))
                                if line_text.strip():
                                    combined_text.append(line_text)
                                    combined_spans.append(span_list)
                    
                    if combined_text:
                        return ("\n".join(combined_text), combined_spans)
        
        # Fall back to OCR if no text styles found or matched
        crop_img = image[y1:y2, x1:x2]
        text = optimize_image_for_ocr(crop_img)
        text = text.strip()
        lines = text.splitlines()
        lines = join_split_words(lines)
        filtered_lines = []
        filtered_spans = []
        for line in lines:
            if len(line.strip()) == 1 and line.strip().lower() not in {"y", "€"}:
                continue
            filtered_lines.append(line)
            filtered_spans.append([(line, False, False)])
        return ("\n".join(filtered_lines), filtered_spans)
    except Exception as e:
        logger.error(f"Error in text extraction: {str(e)}")
        return ("", [])

def merge_overlapping_blocks(layout):
    """Merge blocks that have significant overlap or are part of the same paragraph"""
    merged_layout = []
    used = set()
    
    # Sort blocks by y-coordinate and then x-coordinate
    sorted_layout = sorted(layout, key=lambda x: (x.block.coordinates[1], x.block.coordinates[0]))
    
    for i, block in enumerate(sorted_layout):
        if i in used:
            continue
            
        current_block = block
        merged = False
        
        # Look for blocks to merge
        for j in range(i + 1, len(sorted_layout)):
            if j in used:
                continue
                
            next_block = sorted_layout[j]
            
            # Calculate positions and dimensions
            curr_x1, curr_y1, curr_x2, curr_y2 = current_block.block.coordinates
            next_x1, next_y1, next_x2, next_y2 = next_block.block.coordinates
            
            curr_height = curr_y2 - curr_y1
            next_height = next_y2 - next_y1
            avg_height = (curr_height + next_height) / 2
            
            # Check for different merging conditions:
            
            # 1. Direct overlap
            x1 = max(curr_x1, next_x1)
            y1 = max(curr_y1, next_y1)
            x2 = min(curr_x2, next_x2)
            y2 = min(curr_y2, next_y2)
            
            has_overlap = x2 > x1 and y2 > y1
            overlap_area = (x2 - x1) * (y2 - y1) if has_overlap else 0
            
            # 2. Vertical proximity (for paragraph continuation)
            vertical_gap = next_y1 - curr_y2
            is_close_vertical = vertical_gap < avg_height * 0.3  # Reduced from 0.5
            
            # 3. Horizontal alignment (for paragraph continuation)
            horizontal_overlap = (
                min(curr_x2, next_x2) - max(curr_x1, next_x1)
            ) / min(curr_x2 - curr_x1, next_x2 - next_x1)
            
            # 4. Similar block heights (likely same paragraph)
            height_ratio = min(curr_height, next_height) / max(curr_height, next_height)
            similar_heights = height_ratio > 0.9  # Increased from 0.8
            
            # Decide whether to merge based on conditions (more restrictive)
            should_merge = (
                # Direct overlap case - only merge if significant overlap
                (has_overlap and overlap_area > 0.6 * min(  # Increased from 0.3
                    (curr_x2 - curr_x1) * (curr_y2 - curr_y1),
                    (next_x2 - next_x1) * (next_y2 - next_y1)
                )) or
                # Paragraph continuation case - more restrictive
                (is_close_vertical and horizontal_overlap > 0.7 and similar_heights)  # Increased from 0.3
            )
            
            if should_merge:
                # Create merged block
                new_coords = [
                    min(curr_x1, next_x1),
                    min(curr_y1, next_y1),
                    max(curr_x2, next_x2),
                    max(curr_y2, next_y2)
                ]
                
                # Use the block type of the larger block
                curr_area = (curr_x2 - curr_x1) * (curr_y2 - curr_y1)
                next_area = (next_x2 - next_x1) * (next_y2 - next_y1)
                block_type = current_block.type if curr_area > next_area else next_block.type
                
                merged_block = type(current_block)(
                    block=lp.Rectangle(*new_coords),
                    score=(current_block.score + next_block.score) / 2
                )
                merged_block.set(type=block_type)
                
                current_block = merged_block
                used.add(j)
                merged = True
        
        if not merged:
            merged_layout.append(current_block)
        else:
            merged_layout.append(current_block)
        used.add(i)
    
    return merged_layout

def get_font_set(size: int, style_info: Optional[Dict] = None) -> Dict[str, ImageFont.FreeTypeFont]:
    """Get a set of fonts with proper style handling"""
    # Default font paths
    font_paths = {
        'regular': {
            'mac': "/System/Library/Fonts/Arial Unicode.ttf",
            'linux': "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            'windows': "C:\\Windows\\Fonts\\arial.ttf"
        },
        'bold': {
            'mac': "/System/Library/Fonts/Arial Bold.ttf",
            'linux': "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            'windows': "C:\\Windows\\Fonts\\arialbd.ttf"
        },
        'italic': {
            'mac': "/System/Library/Fonts/Arial Italic.ttf",
            'linux': "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
            'windows': "C:\\Windows\\Fonts\\ariali.ttf"
        },
        'bold_italic': {
            'mac': "/System/Library/Fonts/Arial Bold Italic.ttf",
            'linux': "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
            'windows': "C:\\Windows\\Fonts\\arialbi.ttf"
        }
    }
    
    # Determine OS
    import platform
    system = platform.system().lower()
    if 'darwin' in system:
        os_type = 'mac'
    elif 'linux' in system:
        os_type = 'linux'
    else:
        os_type = 'windows'
    
    # Try to load fonts with fallbacks
    fonts = {}
    for style, paths in font_paths.items():
        try:
            # Try OS-specific path first
            fonts[style] = ImageFont.truetype(paths[os_type], size)
        except:
            try:
                # Try other OS paths as fallback
                for other_os in paths:
                    if other_os != os_type:
                        try:
                            fonts[style] = ImageFont.truetype(paths[other_os], size)
                            break
                        except:
                            continue
            except:
                # Use default font as last resort
                fonts[style] = ImageFont.load_default()
    
    # Apply style flags if provided
    if style_info and 'flags' in style_info:
        flags = style_info['flags']
        # Common PDF font flags:
        # 1 = bold
        # 2 = italic
        # 4 = monospace
        # 8 = serif
        if flags & 3 == 3:  # Bold + Italic
            return fonts.get('bold_italic', fonts['regular'])
        elif flags & 1:  # Bold
            return fonts.get('bold', fonts['regular'])
        elif flags & 2:  # Italic
            return fonts.get('italic', fonts['regular'])
    
    return fonts['regular']

def create_translated_image(original_image: np.ndarray, layout: List, translations: List[str], text_styles: Optional[Dict] = None) -> Image.Image:
    """Create a new image with translated text overlaid with improved formatting"""
    # Convert to PIL Image for processing
    img = Image.fromarray(original_image)
    
    # Create a white background for text areas
    mask = Image.new('L', img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    
    # First pass: create mask of all text areas and clear them
    for block, _ in zip(layout, translations):
        coords = block.block.coordinates
        x1, y1, x2, y2 = [int(coord) for coord in coords]
        # Add larger padding to ensure complete text clearing
        padding = 5
        mask_draw.rectangle([x1-padding, y1-padding, x2+padding, y2+padding], fill=255)
    
    # Apply the mask to clear text areas
    img_array = np.array(img)
    mask_array = np.array(mask)
    img_array[mask_array == 255] = 255  # Set masked areas to white
    img = Image.fromarray(img_array)
    
    # Create drawing context for translated text
    draw = ImageDraw.Draw(img)
    
    # Calculate image scaling factor
    image_height = original_image.shape[0]
    
    # Sort blocks by y-coordinate and then x-coordinate for consistent processing
    blocks_with_text = list(zip(layout, translations))
    blocks_with_text.sort(key=lambda x: (x[0].block.coordinates[1], x[0].block.coordinates[0]))
    
    # Track processed regions to avoid overlaps
    processed_regions = []
    
    # Process blocks in order
    for i, (block, translated_text) in enumerate(blocks_with_text):
        if not translated_text.strip():
            continue
            
        coords = block.block.coordinates
        x1, y1, x2, y2 = [int(coord) for coord in coords]
        box_width = x2 - x1
        original_height = y2 - y1
        
        # Get style information for this block
        block_key = f"{x1},{y1},{x2},{y2}"
        style = text_styles.get(block_key, {
            "alignment": "left",
            "is_paragraph_start": True,
            "size": 12,  # Default size if not found
            "page_height": 842  # Default A4 height in points
        })
        
        # Calculate font size based on the original PDF's font size and scaling
        pdf_font_size = style.get("size", 12)
        pdf_page_height = style.get("page_height", 842)
        scale_factor = image_height / pdf_page_height
        font_size = int(pdf_font_size * scale_factor)
        font_size = max(font_size, 10)  # Ensure minimum readable size
        
        # Get appropriate font based on style
        font = get_font_set(font_size, style)
        
        # Calculate padding based on font size
        padding = int(font_size * 0.5)  # Increased padding for better separation
        
        # Check if this is a new paragraph or section
        is_new_section = False
        if translated_text.isupper() or any(marker in translated_text.lower() for marker in ['sepa', 'login', 'iban']):
            is_new_section = True
            y1 += padding * 2  # Add extra spacing before new sections
        
        # Process the text
        words = translated_text.split()
        lines = []
        current_line = []
        current_width = 0
        max_width = box_width - (padding * 2)
        
        for word in words:
            word_width = draw.textlength(word, font=font)
            space_width = draw.textlength(" ", font=font)
            
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width + space_width
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width + space_width
        
        if current_line:
            lines.append(" ".join(current_line))
        
        # Calculate actual height needed
        line_height = int(font_size * 1.3)  # Increased line height for better readability
        needed_height = (len(lines) * line_height) + (padding * 2)
        
        # Find a suitable position for the text block that avoids overlaps
        current_y = y1
        overlaps = True
        max_attempts = 10
        attempt = 0
        
        while overlaps and attempt < max_attempts:
            overlaps = False
            for region in processed_regions:
                rx1, ry1, rx2, ry2 = region
                # Check if current position would overlap
                if not (x2 < rx1 or x1 > rx2 or current_y + needed_height < ry1 or current_y > ry2):
                    overlaps = True
                    current_y = ry2 + padding  # Move below the overlapping region
                    break
            attempt += 1
        
        # Draw background for this text block
        draw.rectangle([x1, current_y, x2, current_y + needed_height], fill='white')
        
        # Draw text with proper font
        text_y = current_y + padding
        for line in lines:
            line_width = draw.textlength(line, font=font)
            
            # Calculate x position based on alignment
            if style.get("alignment") == "center":
                start_x = x1 + (box_width - line_width) // 2
            elif style.get("alignment") == "right":
                start_x = x2 - line_width - padding
            else:  # left alignment
                start_x = x1 + padding
            
            # Draw the text with proper font
            draw.text((start_x, text_y), line, fill='black', font=font)
            text_y += line_height
        
        # Add this region to processed regions
        processed_regions.append((x1, current_y, x2, current_y + needed_height))
    
    
    return img

def safe_ocr(image: np.ndarray, config: str) -> Dict[str, Any]:
    """Safely perform OCR with error handling"""
    try:
        return pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        logger.debug(traceback.format_exc())
        raise OCRError(f"OCR processing failed: {str(e)}")

def safe_translation(text: str, target_lang: str = "EN") -> str:
    """Safely perform translation with error handling"""
    try:
        return translate_text(text, target_lang)
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        logger.debug(traceback.format_exc())
        # Return original text if translation fails
        return text

def safe_image_processing(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Safely process image with error handling"""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        denoised = cv2.fastNlMeansDenoising(binary)
        return gray, binary, denoised
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        logger.debug(traceback.format_exc())
        raise ImageProcessingError(f"Image processing failed: {str(e)}")

def is_similar_fuzzy(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """Check if two texts are similar using fuzzy matching"""
    # Normalize texts
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    # Quick exact match check
    if text1 == text2:
        return True
    
    # Calculate different similarity ratios
    ratio = fuzz.ratio(text1, text2) / 100
    partial_ratio = fuzz.partial_ratio(text1, text2) / 100
    token_sort_ratio = fuzz.token_sort_ratio(text1, text2) / 100
    
    # Return true if any ratio exceeds threshold
    return any(r > threshold for r in [ratio, partial_ratio, token_sort_ratio])

def extract_table_data(block: Dict, text_items: List[Dict]) -> List[List[str]]:
    """Extract structured data from table blocks"""
    # Sort text items by position
    rows = []
    current_row = []
    last_y = None
    row_height_threshold = 5  # pixels
    
    # Sort by y position first, then x position
    sorted_items = sorted(text_items, key=lambda x: (x['top'], x['left']))
    
    for item in sorted_items:
        if last_y is None:
            current_row.append(item['text'])
        else:
            # Check if this item is on a new row
            if abs(item['top'] - last_y) > row_height_threshold:
                if current_row:
                    rows.append(current_row)
                current_row = [item['text']]
            else:
                current_row.append(item['text'])
        last_y = item['top']
    
    if current_row:
        rows.append(current_row)
    
    # Try to detect header row
    if len(rows) > 1:
        header_candidates = rows[0]
        if all(any(word.isupper() for word in cell.split()) for cell in header_candidates):
            # Mark as header by adding HTML-style formatting
            rows[0] = [f"<th>{cell}</th>" for cell in header_candidates]
            rows[1:] = [[f"<td>{cell}</td>" for cell in row] for row in rows[1:]]
    
    return rows

def extract_list_items(text_items: List[Dict]) -> List[str]:
    """Extract and format list items"""
    list_items = []
    bullet_patterns = [
        r'^\s*[\u2022\u2023\u25E6\u2043\u2027]\s+',  # Unicode bullets
        r'^\s*[\-\*]\s+',  # Hyphen or asterisk
        r'^\s*\d+[\.\)]\s+',  # Numbers with dot or parenthesis
        r'^\s*[a-zA-Z][\.\)]\s+',  # Letters with dot or parenthesis
    ]
    
    # Sort by vertical position
    sorted_items = sorted(text_items, key=lambda x: x['top'])
    
    for item in sorted_items:
        text = item['text'].strip()
        
        # Check if text starts with any bullet pattern
        is_list_item = any(re.match(pattern, text) for pattern in bullet_patterns)
        
        if is_list_item:
            # Clean up the bullet point
            for pattern in bullet_patterns:
                text = re.sub(pattern, '', text)
            list_items.append(text.strip())
        elif list_items:  # Check if this might be a continuation of the last item
            if not text.endswith('.'):  # Likely a continuation
                list_items[-1] = f"{list_items[-1]} {text}"
            else:
                list_items.append(text)
    
    return list_items

def postprocess_translation(text: str) -> str:
    """Post-process translated text to fix common issues"""
    if not text.strip():
        return text
    
    # First fix spacing issues to normalize the text
    text = re.sub(r':(\S)', r': \1', text)  # Add space after colons
    text = re.sub(r'\.([a-zA-Z])', r'. \1', text)  # Add space after periods
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'\((\S)', r'( \1', text)  # Add space after opening parenthesis
    text = re.sub(r'(\S)\)', r'\1 )', text)  # Add space before closing parenthesis
    
    # Fix currency and number formatting
    text = re.sub(r'€(\d+)', r'€ \1', text)  # Add space after euro symbol
    text = re.sub(r'(\d+)€', r'\1 €', text)  # Add space before euro symbol
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)  # Add space between number and letter
    text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)  # Add space between letter and number
    
    # Fix IBAN formatting - keep it as one unit without spaces
    text = re.sub(r'NL\s*(\d{2})\s*([A-Z]{4})\s*(\d{3,4})', r'NL\1\2\3', text)
    
    # Fix date formatting
    text = re.sub(r'(\d{1,2})-(\d{1,2})-(\d{4})', r'\1-\2-\3', text)
    
    # Fix time formatting
    text = re.sub(r'(\d{1,2})\s*:\s*(\d{2})\s*:\s*(\d{2})', r'\1:\2:\3', text)
    
    # Fix spacing issues in common phrases
    text = re.sub(r'OpMy\s*Profile', 'On My Profile', text)
    text = re.sub(r'OpMy\s*profile', 'On My profile', text)
    text = re.sub(r'profileyou', 'profile you', text)
    text = re.sub(r'Profileyou', 'Profile you', text)
    text = re.sub(r'yourselfyou', 'yourself. You', text)
    text = re.sub(r'yourself\.\s*Je\s*can', 'yourself. You can', text)
    text = re.sub(r'yourself\s*Je\s*can', 'yourself. You can', text)
    
    # Fix common translation issues - apply these after spacing fixes
    fixes = {
        # Main translation fixes
        'ABONNEMET': 'SUBSCRIPTION',
        'AFKOOPEIGENRISICO': 'DEDUCTIBLE BUYOUT',
        'SEPAMACHTIGING': 'SEPA AUTHORIZATION',
        'SEPA MACHINING': 'SEPA AUTHORIZATION',
        'MACHINING': 'AUTHORIZATION',
        'INLOGGEGEVENS': 'LOGIN DETAILS',
        'LOGIN Data': 'LOGIN DETAILS',
        'LOGIN data': 'LOGIN DETAILS',
        'login Data': 'LOGIN DETAILS',
        'login data': 'LOGIN DETAILS',
        
        # Dutch field names
        'Bedrijfsnaam:': 'Company Name:',
        'Voornaam:': 'First Name:',
        'Voorvoegsel:': 'Prefix:',
        'Naam:': 'Last Name:',
        'Adres:': 'Address:',
        'Telefoonnummer:': 'Phone Number:',
        'Postcode:': 'Postal Code:',
        'E-mailadres:': 'Email Address:',
        'Woonplaats:': 'City:',
        'Geboortedatum:': 'Date of Birth:',
        'Klantnummer:': 'Customer Number:',
        
        # Specific word fixes
        'Mijneigenicope': 'My deductible',
        'MiinProfile': 'My Profile',
        'benjeagreedwiththeGeneralConditions': 'you have agreed to the General Terms and Conditions',
        'Incasso': 'Collection',
        'Machtigingske nmark': 'Mandate reference',
        'Machtigingskenmerk': 'Mandate reference',
        'Incasso ID': 'Collection ID',
        'Collector ID': 'Collection ID',
        'Ondertekenddoor': 'Signed by',
        'Bevestiging ondertekening': 'Confirmation of signature',
        'Datum ondertekening': 'Date of signature',
        'e-mandate': 'e-mandate',
        
        # Capitalization fixes
        'terms And conditions': 'Terms and Conditions',
        'Terms And Conditions': 'Terms and Conditions',
        'terms and Conditions': 'Terms and Conditions',
        'invoice Date': 'invoice date',
        'Invoice Date': 'invoice date',
        'Invoice date': 'invoice date',
        
        # Mixed language fixes
        'Je log in': 'You can log in',
        'Je kunt': 'You can',
        'Je can': 'You can',
        'username you provided password': 'password you provided',
        'the username you provided password': 'the password you provided',
        'password you provided password': 'password you provided',
        'the password you provided password': 'the password you provided',
        
        # Section headers
        'ABONNEMENT': 'SUBSCRIPTION',
        'SEPA MACHTIGING': 'SEPA AUTHORIZATION',
        'MACHTIGINGSKENMERK': 'MANDATE REFERENCE',
        'INCASSANT ID': 'COLLECTION ID',
        'BEVESTIGING ONDERTEKENING': 'CONFIRMATION OF SIGNATURE',
        'ONDERTEKEND DOOR': 'SIGNED BY',
        'DATUM ONDERTEKENING': 'DATE OF SIGNATURE',
        
        # Common OCR/translation errors
        'nmark': 'reference',
        'nmerk': 'reference',
        'signing via': 'signature via',
        'Confirmation signing': 'Confirmation of signature',
        'Date of signature': 'Date of signature',
        'Signed by': 'Signed by',
        'Dark Curtiusstraat': 'Donker Curtiusstraat',  # Don't translate street names
        'shar@gmail. com': 'shar@gmail.com',  # Fix email spacing
    }
    
    # Apply fixes with case sensitivity
    for wrong, right in fixes.items():
        # Try exact match first
        text = text.replace(wrong, right)
        # Try case-insensitive for some common patterns
        if wrong.lower() in ['login data', 'terms and conditions']:
            text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    
    # Fix specific patterns with regex
    text = re.sub(r'\bLOGIN\s+[Dd]ata\b', 'LOGIN DETAILS', text)
    text = re.sub(r'\bterms\s+[Aa]nd\s+[Cc]onditions\b', 'Terms and Conditions', text)
    text = re.sub(r'\bJe\s+can\s+log\s+in\b', 'You can log in', text, flags=re.IGNORECASE)
    
    return text

def match_casing(original: str, translated: str) -> str:
    """Match casing word-by-word: for each word in original, match casing to corresponding word in translated."""
    orig_words = original.split()
    trans_words = translated.split()
    result = []
    
    # Special case: if original is a single all-caps word and translation is multiple words
    if len(orig_words) == 1 and orig_words[0].isupper() and len(trans_words) > 1:
        # Make all translated words uppercase
        return translated.upper()
    
    # Special case for INLOGGEGEVENS -> LOGIN DATA/DETAILS
    if original.strip() == 'INLOGGEGEVENS' and translated.strip().startswith('LOGIN'):
        return translated.upper()
    
    for i, tword in enumerate(trans_words):
        if i < len(orig_words):
            oword = orig_words[i]
            if oword.isupper():
                result.append(tword.upper())
            elif oword.istitle():
                result.append(tword.capitalize())
            elif oword.islower():
                result.append(tword.lower())
            else:
                result.append(tword)
        else:
            # If there are more translated words than original words,
            # use the casing of the last original word
            if orig_words and orig_words[-1].isupper():
                result.append(tword.upper())
            else:
                result.append(tword)
    return ' '.join(result)

def translate_spans_with_formatting(lines, target_lang="EN"):
    """Translate a list of lines, each line is a list of (text, is_bold, is_italic) spans. Preserve formatting and match original casing. Each line in the original corresponds to a line in the translation."""
    translated_lines = []
    for line in lines:
        translated_line = ""
        for original, is_bold, is_italic in line:
            translated = translate_text(original, target_lang)
            translated = postprocess_translation(translated)
            translated = match_casing(original, translated)
            if is_bold:
                translated = f"**{translated}**"
            if is_italic:
                translated = f"*{translated}*"
            translated_line += translated
        translated_lines.append(translated_line)
    return "\n".join(translated_lines)

def process_blocks_parallel(image: np.ndarray, layout: List, text_styles: Dict = None) -> Tuple[List, List]:
    results = []
    translations = []
    def process_single_block(block_data):
        i, block = block_data
        try:
            # Extract text and spans, print debug for first block
            text, span_lines = extract_text_and_spans_from_block(image, block, text_styles, debug_block_idx=0, block_idx=i)
            
            # Skip empty blocks
            if not text.strip():
                print(f"DEBUG: Block {i} is empty, skipping")
                return None
                
            text_items = []
            table_data = []
            list_items = []
            
            if block.type == "Table":
                table_data = extract_table_data(block, text_items)
                text = "\n".join([" | ".join(row) for row in table_data])
                span_lines = [[(cell, False, False) for cell in row] for row in table_data]
            elif block.type == "List":
                list_items = extract_list_items(text_items)
                text = "\n".join([f"• {item}" for item in list_items])
                span_lines = [[(item, False, False)] for item in list_items]
            
            # Always try to use span-based translation for better formatting
            if span_lines:
                print(f"DEBUG: Block {i} using span-based translation")
                translated_text = translate_spans_with_formatting(span_lines)
            else:
                print(f"DEBUG: Block {i} using fallback translation")
                # For non-span text, apply translation and post-processing
                translated_text = safe_translation(text)
                translated_text = postprocess_translation(translated_text)
                # Apply casing match line by line
                if '\n' in text or '\n' in translated_text:
                    orig_lines = text.split('\n')
                    trans_lines = translated_text.split('\n')
                    matched_lines = []
                    for j, trans_line in enumerate(trans_lines):
                        if j < len(orig_lines):
                            matched_lines.append(match_casing(orig_lines[j], trans_line))
                        else:
                            matched_lines.append(trans_line)
                    translated_text = '\n'.join(matched_lines)
                else:
                    translated_text = match_casing(text, translated_text)
            
            print(f"DEBUG: Block {i} processed successfully: '{text[:50]}...' -> '{translated_text[:50]}...'")
            
            return {
                "index": i,
                "type": block.type,
                "confidence": float(block.score),
                "coordinates": [float(x) for x in block.block.coordinates],
                "original_text": text,
                "translated_text": translated_text,
                "structured_data": table_data if block.type == "Table" else list_items if block.type == "List" else None
            }
        except Exception as e:
            logger.error(f"Error processing block {i}: {str(e)}")
            logger.debug(traceback.format_exc())
            print(f"DEBUG: Block {i} failed with error: {str(e)}")
            return None
    
    # Process blocks sequentially for better debugging
    for i, block in enumerate(layout):
        result = process_single_block((i, block))
        if result is not None:
            results.append(result)
            translations.append(result["translated_text"])
        else:
            print(f"DEBUG: Block {i} returned None")
    
    print(f"DEBUG: Total results before deduplication: {len(results)}")
    
    # Remove duplicates based on text similarity and coordinate overlap
    final_results = []
    used_indices = set()
    
    for i, current in enumerate(results):
        if i in used_indices:
            continue
            
        # Check if this block is similar to any already processed block
        is_duplicate = False
        for j, existing in enumerate(final_results):
            # Check text similarity (more lenient for exact duplicates)
            current_text_clean = re.sub(r'\s+', ' ', current["original_text"].lower().strip())
            existing_text_clean = re.sub(r'\s+', ' ', existing["original_text"].lower().strip())
            
            # Check for exact text match or very high similarity
            exact_match = current_text_clean == existing_text_clean
            text_similarity = is_similar_fuzzy(current["original_text"], existing["original_text"], threshold=0.9)
            
            # Check coordinate overlap
            curr_coords = current["coordinates"]
            exist_coords = existing["coordinates"]
            
            # Calculate overlap area
            x1 = max(curr_coords[0], exist_coords[0])
            y1 = max(curr_coords[1], exist_coords[1])
            x2 = min(curr_coords[2], exist_coords[2])
            y2 = min(curr_coords[3], exist_coords[3])
            
            has_overlap = x2 > x1 and y2 > y1
            if has_overlap:
                overlap_area = (x2 - x1) * (y2 - y1)
                curr_area = (curr_coords[2] - curr_coords[0]) * (curr_coords[3] - curr_coords[1])
                exist_area = (exist_coords[2] - exist_coords[0]) * (exist_coords[3] - exist_coords[1])
                overlap_ratio = overlap_area / min(curr_area, exist_area)
            else:
                overlap_ratio = 0
            
            # Consider it a duplicate if:
            # 1. Exact text match, OR
            # 2. High text similarity AND significant coordinate overlap
            if exact_match or (text_similarity and overlap_ratio > 0.3):
                is_duplicate = True
                print(f"DEBUG: Block {current['index']} is duplicate of block {existing['index']} (exact: {exact_match}, sim: {text_similarity}, overlap: {overlap_ratio:.2f})")
                # Keep the block with more text content or better confidence
                if len(current["original_text"]) > len(existing["original_text"]) or current["confidence"] > existing["confidence"]:
                    final_results[j] = current
                    print(f"DEBUG: Keeping better version from block {current['index']}")
                break
        
        if not is_duplicate:
            final_results.append(current)
        
        used_indices.add(i)
    
    print(f"DEBUG: Final results after deduplication: {len(final_results)}")
    return final_results, [r["translated_text"] for r in final_results]

class TestLayoutParser(unittest.TestCase):
    """Test cases for layout parser functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test resources"""
        # Create a temporary test image
        cls.test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.putText(cls.test_image, "Test", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        
        # Save test image
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_image_path = os.path.join(cls.temp_dir, "test.png")
        Image.fromarray(cls.test_image).save(cls.test_image_path)
        
        # Create test PDF
        cls.test_pdf_path = os.path.join(cls.temp_dir, "test.pdf")
        c = canvas.Canvas(cls.test_pdf_path)
        c.drawString(100, 750, "Test PDF")
        c.save()
    
    def test_image_optimization(self):
        """Test image optimization function"""
        optimized = optimize_image_for_ocr(self.test_image)
        self.assertEqual(optimized.shape, self.test_image.shape)
        self.assertTrue(np.any(optimized))  # Should not be empty
    
    def test_text_similarity(self):
        """Test fuzzy text matching"""
        text1 = "This is a test"
        text2 = "This is a test."
        text3 = "Something completely different"
        
        self.assertTrue(is_similar_fuzzy(text1, text2))
        self.assertFalse(is_similar_fuzzy(text1, text3))
    
    def test_table_extraction(self):
        """Test table data extraction"""
        test_items = [
            {'text': 'Header 1', 'top': 0, 'left': 0},
            {'text': 'Header 2', 'top': 0, 'left': 100},
            {'text': 'Data 1', 'top': 50, 'left': 0},
            {'text': 'Data 2', 'top': 50, 'left': 100}
        ]
        
        table_data = extract_table_data({}, test_items)
        self.assertEqual(len(table_data), 2)  # Should have 2 rows
        self.assertEqual(len(table_data[0]), 2)  # Each row should have 2 columns
    
    def test_list_extraction(self):
        """Test list item extraction"""
        test_items = [
            {'text': '• First item', 'top': 0},
            {'text': '• Second item', 'top': 20},
            {'text': 'continuation', 'top': 30}
        ]
        
        list_items = extract_list_items(test_items)
        self.assertEqual(len(list_items), 2)
        self.assertTrue('First item' in list_items[0])
        self.assertTrue('Second item continuation' in list_items[1])
    
    def test_pdf_validation(self):
        """Test PDF validation"""
        self.assertTrue(validate_pdf(self.test_pdf_path))
        with self.assertRaises(ValueError):
            validate_pdf("nonexistent.pdf")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test resources"""
        import shutil
        shutil.rmtree(cls.temp_dir)

def validate_pdf(pdf_path: str) -> bool:
    """Validate PDF file"""
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        return page_count > 0
    except Exception as e:
        raise ValueError(f"Invalid PDF file: {str(e)}")

def validate_output_path(path: str) -> bool:
    """Validate output path"""
    try:
        if not path:
            raise ValueError("Empty path provided")
            
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return True
    except Exception as e:
        raise ValueError(f"Invalid output path: {str(e)}")

def validate_image(image: np.ndarray) -> bool:
    """Validate image array"""
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    if len(image.shape) not in [2, 3]:
        raise ValueError("Image must be 2D or 3D array")
    
    if image.size == 0:
        raise ValueError("Image is empty")
    
    return True

def main():
    """Main function with improved validation"""
    try:
        # Validate and load the model
        logger.info("Loading the model...")
        model = lp.Detectron2LayoutModel(
            config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.2,
                         "MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.1]
        )
        
        # Process the PDF
        logger.info("Loading the PDF...")
        pdf_path = 'sample.pdf'
        
        try:
            # Validate PDF
            if not validate_pdf(pdf_path):
                raise ValueError("Invalid PDF file")
            
            # Validate output paths
            output_paths = [
                'layout_results.json',
                'extracted_text.txt'
            ]
            for path in output_paths:
                validate_output_path(path)
            
            # Process all pages
            results = process_pdf_pages(pdf_path, model)
            
            # Save outputs with validation
            logger.info("Saving outputs...")
            
            try:
                # Save JSON output with validation
                with open('layout_results.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info("Saved layout_results.json")
            except Exception as e:
                logger.error(f"Failed to save JSON output: {str(e)}")
            
            try:
                # Create text-only output with proper formatting
                with open('extracted_text.txt', 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(f"Type: {result['type']}\n")
                        f.write(f"Original: {result['original_text']}\n")
                        f.write(f"Translated: {result['translated_text']}\n")
                        f.write("-" * 50 + "\n")
                logger.info("Saved extracted_text.txt")
            except Exception as e:
                logger.error(f"Failed to save text output: {str(e)}")
            
            logger.info("\nProcessing completed successfully!")
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
    except Exception as e:
        logger.error("An error occurred during processing")
        logger.error(str(e))
        logger.debug(traceback.format_exc())
        
        # Print user-friendly error message
        print("\nAn error occurred during processing.")
        print("Please check the following:")
        print("1. Make sure your PDF file exists and is readable")
        print("2. Ensure you have poppler installed for PDF processing")
        print("3. Ensure you have tesseract installed for OCR")
        print("4. Check if you have sufficient memory for processing")
        print("\nFor detailed error information, check layout_parser.log")
        print("\nFor macOS users:")
        print("brew install poppler tesseract")
        
        sys.exit(1)

if __name__ == "__main__":
    # Run tests if in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        unittest.main(argv=['first-arg-is-ignored'])
    else:
        main() 