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
        logging.FileHandler('layout_parser_fixed.log')
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
    
    try:
        url = "https://api-free.deepl.com/v2/translate"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "auth_key": "9092aa95-6d0e-4cdc-a372-31cb2842b3ae:fx",
            "text": text,
            "target_lang": target_lang,
            "preserve_formatting": "1",
            "tag_handling": "xml",
            "split_sentences": "nonewlines"
        }
        
        response = requests.post(url, data=data, headers=headers)
        response.raise_for_status()
        translated = response.json()["translations"][0]["text"]
        
        # Restore line breaks and paragraphs
        translated = translated.replace(' <paragraph_break> ', '\n\n')
        translated = translated.replace(' <line_break> ', '\n')
        
        return translated
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

def get_text_style(pdf_path, page_num=0):
    """Extract text style information from PDF"""
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
                    line_spans.append({
                        "text": span["text"],
                        "font": span["font"],
                        "size": span["size"],
                        "flags": span["flags"],
                        "color": span["color"]
                    })
                line_data.append(line_spans)
            
            # Determine alignment
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
            
            # Get dominant style
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
                    "lines": line_data
                }
    
    return text_styles

def postprocess_translation(text: str) -> str:
    """Post-process translated text to fix common issues"""
    if not text.strip():
        return text
    
    # Fix spacing issues
    text = re.sub(r':(\S)', r': \1', text)
    text = re.sub(r'\.([a-zA-Z])', r'. \1', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common translation issues
    fixes = {
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
        'OpMy Profileyou': 'On My Profile you',
        'profileyou': 'profile you',
        'Profileyou': 'Profile you',
        'yourselfyou': 'yourself. You',
        'yourself Je can': 'yourself. You can',
        'Je can': 'You can',
        'Je log in': 'You can log in',
        'Je kunt': 'You can',
        'username you provided password': 'password you provided',
        'the username you provided password': 'the password you provided',
        'password you provided password': 'password you provided',
        'Dark Curtiusstraat': 'Donker Curtiusstraat',
        'shar@gmail. com': 'shar@gmail.com',
        'terms And conditions': 'Terms and Conditions',
        'Terms And Conditions': 'Terms and Conditions',
        'invoice Date': 'invoice date',
        'Invoice Date': 'invoice date',
        'Machtigingske nmerk': 'Machtigingskenmerk',
        'Machtigingske\nnmerk': 'Machtigingskenmerk',
        'Machtigingske\nreference': 'Mandate reference',
        'xMachtigingske': 'x Machtigingske',
        'Collection ID:\n68 ZZZ': 'Collection ID: 68ZZZ',
        'Confirmation signature via': 'Confirmation of signature via',
        'Date of signature': 'Date of signature',
        'Signed by': 'Signed by',
    }
    
    # Apply fixes
    for wrong, right in fixes.items():
        text = text.replace(wrong, right)
    
    # Fix IBAN formatting
    text = re.sub(r'NL\s*(\d{2})\s*([A-Z]{4})\s*(\d{3,4})', r'NL\1\2\3', text)
    
    # Fix currency formatting
    text = re.sub(r'€(\d+)', r'€ \1', text)
    text = re.sub(r'(\d+)€', r'\1 €', text)
    
    return text.strip()

def match_casing(original: str, translated: str) -> str:
    """Match casing of translated text to original"""
    orig_words = original.split()
    trans_words = translated.split()
    result = []
    
    # Special case: if original is all caps and translation is multiple words
    if len(orig_words) == 1 and orig_words[0].isupper() and len(trans_words) > 1:
        return translated.upper()
    
    # Special case for specific headers
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
            # Use casing of last original word for extra translated words
            if orig_words and orig_words[-1].isupper():
                result.append(tword.upper())
            else:
                result.append(tword)
    
    return ' '.join(result)

def optimize_image_for_ocr(image: np.ndarray) -> str:
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
    
    # Try OCR on each preprocessed image and choose the best result
    best_text = ""
    best_score = -1
    
    for img in results:
        try:
            config = '--oem 3 --psm 6 -l nld+eng'
            text = pytesseract.image_to_string(img, config=config)
            
            # Score the result
            score = len(text)
            dutch_words = ['ABONNEMENT', 'SEPA', 'MACHTIGING', 'BEVESTIGING', 'ONDERTEKENING']
            score += sum(10 for word in dutch_words if word in text.upper())
            special_chars = len([c for c in text if not (c.isalnum() or c.isspace() or c in '.,:-€()')])
            score -= special_chars * 2
            
            if score > best_score:
                best_score = score
                best_text = text
        except Exception as e:
            logger.debug(f"OCR attempt failed: {str(e)}")
            continue
    
    return best_text

def extract_text_and_spans_from_block(image: np.ndarray, block, text_styles=None) -> Tuple[str, list]:
    """Extract text and per-span structure from a block"""
    try:
        x1, y1, x2, y2 = [int(coord) for coord in block.block.coordinates]
        padding = 15
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.shape[1], x2 + padding)
        y2 = min(image.shape[0], y2 + padding)
        
        if x2 - x1 < 5 or y2 - y1 < 5:
            return ("", [])
        
        if text_styles is not None:
            # Find the best matching text style block
            key = None
            best_overlap = 0
            block_area = (x2 - x1) * (y2 - y1)
            
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
            
            # If we found a good match, use the text style data
            if key and key in text_styles and "lines" in text_styles[key] and best_overlap > 0.1:
                lines = text_styles[key]["lines"]
                formatted_lines = []
                joined_lines = []
                
                for line_spans in lines:
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
                    formatted_lines.append(line_text)
                    joined_lines.append(span_list)
                
                # Filter out single character lines (except meaningful ones)
                filtered_lines = []
                filtered_spans = []
                for line, spans in zip(formatted_lines, joined_lines):
                    if len(line.strip()) == 1 and line.strip().lower() not in {"y", "€"}:
                        continue
                    filtered_lines.append(line)
                    filtered_spans.append(spans)
                
                filtered_spans_no_font = [[(s[0], s[1], s[2]) for s in line] for line in filtered_spans]
                return ("\n".join(filtered_lines), filtered_spans_no_font)
        
        # Fall back to OCR if no text styles found
        crop_img = image[y1:y2, x1:x2]
        text = optimize_image_for_ocr(crop_img)
        text = text.strip()
        lines = text.splitlines()
        
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
    """Merge blocks that have significant overlap - more conservative approach"""
    merged_layout = []
    used = set()
    
    # Sort blocks by y-coordinate and then x-coordinate
    sorted_layout = sorted(layout, key=lambda x: (x.block.coordinates[1], x.block.coordinates[0]))
    
    for i, block in enumerate(sorted_layout):
        if i in used:
            continue
            
        current_block = block
        
        # Look for blocks to merge - be more conservative
        for j in range(i + 1, len(sorted_layout)):
            if j in used:
                continue
                
            next_block = sorted_layout[j]
            
            # Calculate positions and dimensions
            curr_x1, curr_y1, curr_x2, curr_y2 = current_block.block.coordinates
            next_x1, next_y1, next_x2, next_y2 = next_block.block.coordinates
            
            # Check for significant overlap only
            x1 = max(curr_x1, next_x1)
            y1 = max(curr_y1, next_y1)
            x2 = min(curr_x2, next_x2)
            y2 = min(curr_y2, next_y2)
            
            has_overlap = x2 > x1 and y2 > y1
            
            if has_overlap:
                overlap_area = (x2 - x1) * (y2 - y1)
                curr_area = (curr_x2 - curr_x1) * (curr_y2 - curr_y1)
                next_area = (next_x2 - next_x1) * (next_y2 - next_y1)
                min_area = min(curr_area, next_area)
                
                # Only merge if overlap is very significant (>70%)
                if overlap_area > 0.7 * min_area:
                    # Create merged block
                    new_coords = [
                        min(curr_x1, next_x1),
                        min(curr_y1, next_y1),
                        max(curr_x2, next_x2),
                        max(curr_y2, next_y2)
                    ]
                    
                    # Use the block type of the larger block
                    block_type = current_block.type if curr_area > next_area else next_block.type
                    
                    merged_block = type(current_block)(
                        block=lp.Rectangle(*new_coords),
                        score=(current_block.score + next_block.score) / 2
                    )
                    merged_block.set(type=block_type)
                    
                    current_block = merged_block
                    used.add(j)
        
        merged_layout.append(current_block)
        used.add(i)
    
    return merged_layout

def is_similar_fuzzy(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """Check if two texts are similar using fuzzy matching"""
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    if text1 == text2:
        return True
    
    ratio = fuzz.ratio(text1, text2) / 100
    partial_ratio = fuzz.partial_ratio(text1, text2) / 100
    token_sort_ratio = fuzz.token_sort_ratio(text1, text2) / 100
    
    return any(r > threshold for r in [ratio, partial_ratio, token_sort_ratio])

def remove_duplicates(results: List[Dict]) -> List[Dict]:
    """Remove duplicate results based on text similarity and coordinate overlap"""
    final_results = []
    
    for current in results:
        is_duplicate = False
        
        for existing in final_results:
            # Check text similarity
            current_text_clean = re.sub(r'\s+', ' ', current["original_text"].lower().strip())
            existing_text_clean = re.sub(r'\s+', ' ', existing["original_text"].lower().strip())
            
            # Check for exact match or high similarity
            exact_match = current_text_clean == existing_text_clean
            text_similarity = is_similar_fuzzy(current["original_text"], existing["original_text"], threshold=0.9)
            
            # Check coordinate overlap
            curr_coords = current["coordinates"]
            exist_coords = existing["coordinates"]
            
            x1 = max(curr_coords[0], exist_coords[0])
            y1 = max(curr_coords[1], exist_coords[1])
            x2 = min(curr_coords[2], exist_coords[2])
            y2 = min(curr_coords[3], exist_coords[3])
            
            has_overlap = x2 > x1 and y2 > y1
            overlap_ratio = 0
            
            if has_overlap:
                overlap_area = (x2 - x1) * (y2 - y1)
                curr_area = (curr_coords[2] - curr_coords[0]) * (curr_coords[3] - curr_coords[1])
                exist_area = (exist_coords[2] - exist_coords[0]) * (exist_coords[3] - exist_coords[1])
                overlap_ratio = overlap_area / min(curr_area, exist_area)
            
            # Consider it a duplicate if exact match OR high similarity with overlap
            if exact_match or (text_similarity and overlap_ratio > 0.3):
                is_duplicate = True
                # Keep the one with more content or better confidence
                if len(current["original_text"]) > len(existing["original_text"]) or current["confidence"] > existing["confidence"]:
                    # Replace existing with current
                    idx = final_results.index(existing)
                    final_results[idx] = current
                break
        
        if not is_duplicate:
            final_results.append(current)
    
    return final_results

def translate_spans_with_formatting(lines, target_lang="EN"):
    """Translate spans while preserving formatting"""
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
    """Process blocks and extract text with translations"""
    results = []
    
    for i, block in enumerate(layout):
        try:
            # Extract text and spans
            text, span_lines = extract_text_and_spans_from_block(image, block, text_styles)
            
            # Skip empty blocks
            if not text.strip():
                continue
            
            # Translate text
            if span_lines:
                translated_text = translate_spans_with_formatting(span_lines)
            else:
                translated_text = translate_text(text)
                translated_text = postprocess_translation(translated_text)
                translated_text = match_casing(text, translated_text)
            
            # Determine block type based on content
            block_type = "Text"  # Default type
            if hasattr(block, 'type') and block.type:
                block_type = block.type
            else:
                # Infer type from content
                text_upper = text.upper()
                if any(header in text_upper for header in ['SEPA', 'IBAN', 'LOGIN', 'ABONNEMENT', 'CONTRACT']):
                    block_type = "Title"
                elif len(text.split('\n')) > 3:
                    block_type = "Text"
                elif '•' in text or re.search(r'^\s*\d+\.', text):
                    block_type = "List"
            
            result = {
                "index": i,
                "type": block_type,
                "confidence": float(block.score) if hasattr(block, 'score') else 1.0,
                "coordinates": [float(x) for x in block.block.coordinates],
                "original_text": text,
                "translated_text": translated_text,
                "structured_data": None
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing block {i}: {str(e)}")
            continue
    
    # Remove duplicates
    results = remove_duplicates(results)
    
    return results, [r["translated_text"] for r in results]

def process_pdf_pages(pdf_path: str, model: lp.Detectron2LayoutModel) -> List[Dict]:
    """Process all pages in a PDF"""
    logger.info(f"Processing PDF: {pdf_path}")
    all_results = []
    
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    total_pages = len(images)
    logger.info(f"Found {total_pages} pages")
    
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
            logger.info(f"Found {len(text_styles)} text style blocks")
            
            # Detect layout
            layout = model.detect(image_np)
            logger.info(f"Layout model detected {len(layout)} blocks")
            
            # Scale coordinates to actual image size
            height, width = image_np.shape[:2]
            scaled_layout = []
            for block in layout:
                coords = block.block.coordinates
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
            
            # Merge overlapping blocks (conservative)
            layout = merge_overlapping_blocks(scaled_layout)
            logger.info(f"After merging: {len(layout)} blocks")
            
            # Process blocks
            results, translations = process_blocks_parallel(image_np, layout, text_styles)
            logger.info(f"Processed {len(results)} blocks successfully")
            
            # Add page information
            for result in results:
                result["page_number"] = page_num + 1
            
            all_results.extend(results)
            
            # Create visualization for this page
            viz_image = lp.draw_box(image_np, layout, box_width=3)
            plt.figure(figsize=(15, 15))
            plt.imshow(viz_image)
            plt.axis('off')
            plt.savefig(f'layout_visualization_fixed_page_{page_num + 1}.png', bbox_inches='tight', pad_inches=0)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {str(e)}")
            logger.debug(traceback.format_exc())
            continue
    
    return all_results

def main():
    """Main function with improved validation"""
    try:
        # Load the model
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
        
        if not os.path.exists(pdf_path):
            raise ValueError(f"PDF file not found: {pdf_path}")
        
        # Process all pages
        results = process_pdf_pages(pdf_path, model)
        
        # Save outputs
        logger.info("Saving outputs...")
        
        # Save JSON output
        with open('layout_results_fixed.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Saved layout_results_fixed.json")
        
        # Create improved text output
        with open('extracted_text_fixed.txt', 'w', encoding='utf-8') as f:
            f.write("IMPROVED LAYOUT PARSER RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            # Group by page
            pages = {}
            for result in results:
                page_num = result.get('page_number', 1)
                if page_num not in pages:
                    pages[page_num] = []
                pages[page_num].append(result)
            
            for page_num in sorted(pages.keys()):
                f.write(f"PAGE {page_num}\n")
                f.write("-" * 20 + "\n\n")
                
                for result in pages[page_num]:
                    f.write(f"Type: {result['type']}\n")
                    f.write(f"Confidence: {result['confidence']:.2f}\n")
                    f.write(f"Original: {result['original_text']}\n")
                    f.write(f"Translated: {result['translated_text']}\n")
                    f.write("-" * 50 + "\n\n")
        
        logger.info("Saved extracted_text_fixed.txt")
        logger.info(f"\nProcessing completed successfully! Found {len(results)} text blocks.")
        
    except Exception as e:
        logger.error("An error occurred during processing")
        logger.error(str(e))
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 