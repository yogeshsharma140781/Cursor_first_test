#!/usr/bin/env python3
"""
Requirements:
PyMuPDF==1.23.6
openai>=1.0.0
pytesseract
Pillow
python-dotenv

# For OCR, you need Tesseract installed on your system.
# On macOS: brew install tesseract
# On Ubuntu: sudo apt-get install tesseract-ocr
"""
import fitz  # PyMuPDF
import argparse
import os
import sys
import openai
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from typing import List, Tuple
import io
import time
import layoutparser as lp
import cv2
import numpy as np

# Load OpenAI key from .env or environment
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Utility Functions ---
def ocr_image_from_pix(pix) -> str:
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img)

def extract_blocks_from_pdf(pdf_path: str, ocr_threshold=0.5) -> List[dict]:
    doc = fitz.open(pdf_path)
    all_blocks = []
    for page_num, page in enumerate(doc):
        text_dict = page.get_text("dict")
        # If page is mostly image, use OCR
        if len(text_dict["blocks"]) == 0 or sum(b.get("type", 0) == 1 for b in text_dict["blocks"]) / max(1, len(text_dict["blocks"])) > ocr_threshold:
            print(f"[INFO] Page {page_num+1}: Using OCR fallback.")
            pix = page.get_pixmap()
            ocr_text = ocr_image_from_pix(pix)
            all_blocks.append({
                "text": ocr_text.strip(),
                "bbox": (50, 50, page.rect.width-50, page.rect.height-50),
                "size": 12,
                "font": "helv",
                "page": page_num,
                "type": "ocr"
            })
            continue
        for block in text_dict["blocks"]:
            if block.get("type", 0) == 0 and "lines" in block:
                block_text = " ".join(
                    "".join(span["text"] for span in line["spans"]) for line in block["lines"]
                ).strip()
                if not block_text:
                    continue
                # Use the first span for font/size
                first_span = block["lines"][0]["spans"][0]
                all_blocks.append({
                    "text": block_text,
                    "bbox": block["bbox"],
                    "size": first_span.get("size", 12),
                    "font": first_span.get("font", "helv"),
                    "page": page_num,
                    "type": "text"
                })
            elif block.get("type", 0) == 1:
                # Image block
                all_blocks.append({
                    "image": block.get("image", None),
                    "bbox": block["bbox"],
                    "page": page_num,
                    "type": "image"
                })
    doc.close()
    return all_blocks

def group_blocks_to_paragraphs(blocks: List[dict]) -> List[dict]:
    # No grouping: return only text and ocr blocks as-is, in order
    return [block for block in blocks if block["type"] in ("text", "ocr")]

def wrap_text_to_fit(page, text, fontname, font_size, max_width, max_height, min_font_size=5):
    # Try to wrap and fit text into the bounding box, reducing font size if needed
    lines = []
    words = text.split()
    while font_size >= min_font_size:
        lines = []
        current_line = words[0] if words else ''
        for word in words[1:]:
            test_line = current_line + ' ' + word
            w = fitz.get_text_length(test_line, fontname=fontname, fontsize=font_size)
            if w <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        line_height = font_size * 1.15
        total_height = len(lines) * line_height
        if total_height <= max_height:
            return lines, font_size
        font_size -= 1
    # If we get here, couldn't fit text even at min font size
    return lines, font_size

def batch_translate(texts: List[str], target_lang: str, batch_size=10, model="gpt-3.5-turbo") -> List[str]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        prompt = "".join([f"{idx+1}. {t}\n" for idx, t in enumerate(batch)])
        system_prompt = f"You are a professional translator. Translate the following text to {target_lang}. Return only the translations, numbered as in the input."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2048,
                temperature=0.2
            )
            content = response.choices[0].message.content
            # Parse numbered list
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            translations = []
            for l in lines:
                if l[0].isdigit() and l[1] in ".)":
                    translations.append(l[2:].strip())
                elif l[1:3] == ". ":
                    translations.append(l[3:].strip())
                else:
                    translations.append(l)
            if len(translations) != len(batch):
                print("[WARN] Mismatch in translation count, using original text for missing.")
                while len(translations) < len(batch):
                    translations.append(batch[len(translations)])
            results.extend(translations)
        except Exception as e:
            print(f"[ERROR] OpenAI translation failed: {e}")
            results.extend(batch)  # fallback: use original
        time.sleep(1)  # avoid rate limits
    return results

def create_translated_pdf_pymupdf(original_pdf_path: str, blocks: List[dict], translations: List[str], output_pdf_path: str):
    doc = fitz.open(original_pdf_path)
    for block, translated in zip(blocks, translations):
        if block["type"] == "text" or block["type"] == "ocr":
            page = doc[block["page"]]
            x0, y0, x1, y1 = block["bbox"]
            # Redact original text
            padding = 2
            redact_rect = fitz.Rect(x0-padding, y0-padding, x1+padding, y1+padding)
            page.add_redact_annot(redact_rect, text="", fill=(1, 1, 1))
    for page in doc:
        page.apply_redactions()
    for block, translated in zip(blocks, translations):
        if block["type"] == "text" or block["type"] == "ocr":
            page = doc[block["page"]]
            x0, y0, x1, y1 = block["bbox"]
            font_size = max(5, int(block.get("size", 12)))
            font_name = "helv"  # PyMuPDF built-in font
            # Remove unsupported characters
            safe_text = ''.join([c if ord(c) < 128 or (32 <= ord(c) <= 126) else '?' for c in translated])
            try:
                page.insert_text((x0+2, y0+font_size+2), safe_text, fontsize=font_size, color=(0,0,0), fontname=font_name)
            except Exception as e:
                print(f"[WARN] Could not insert text: {safe_text[:30]}... Error: {e}")
        elif block["type"] == "image":
            # Re-insert image if possible (not always possible with PyMuPDF, but we try)
            page = doc[block["page"]]
            x0, y0, x1, y1 = block["bbox"]
            # If image data is available, insert it
            if block.get("image"):
                try:
                    img_bytes = block["image"]
                    img = Image.open(io.BytesIO(img_bytes))
                    img_path = f"temp_img_{block['page']}.png"
                    img.save(img_path)
                    page.insert_image(fitz.Rect(x0, y0, x1, y1), filename=img_path)
                    os.remove(img_path)
                except Exception as e:
                    print(f"[WARN] Could not re-insert image: {e}")
    doc.save(output_pdf_path)
    doc.close()
    print(f"[PyMuPDF] Translated PDF saved to: {output_pdf_path}")

def detect_blocks_layoutparser(pdf_path: str, page_num: int = 0):
    # Render the first page to an image
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_np = np.array(img)
    # Use layoutparser's Detectron2 model
    model = lp.Detectron2LayoutModel(
        config_path='publaynet_config.yml',
        model_path='/Users/yogesh/.torch/iopath_cache/s/dgy9c10wykk4lq4/model_final.pth',
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5, "MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.5],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )
    layout = model.detect(img_np)
    blocks = []
    for lblock in layout:
        x0, y0, x1, y1 = map(int, lblock.block.coordinates)
        block_type = lblock.type
        # Map image coordinates to PDF coordinates (PyMuPDF origin is top-left, same as image)
        # Extract text from the PDF region
        rect = fitz.Rect(x0, y0, x1, y1)
        text = page.get_textbox(rect).strip()
        if not text:
            # Fallback to OCR
            crop = img.crop((x0, y0, x1, y1))
            text = pytesseract.image_to_string(crop).strip()
        blocks.append({
            "text": text,
            "bbox": (x0, y0, x1, y1),
            "size": 12,  # We'll use a default size for now
            "font": "helv",
            "page": page_num,
            "type": block_type
        })
    doc.close()
    return blocks

def main():
    parser = argparse.ArgumentParser(description="Translate a PDF and preserve layout using PyMuPDF and OpenAI.")
    parser.add_argument("input_pdf", help="Input PDF file")
    parser.add_argument("output_pdf", help="Output (translated) PDF file")
    parser.add_argument("--lang", default="en", help="Target language (default: en)")
    args = parser.parse_args()

    print(f"[INFO] Detecting blocks visually on first page using layoutparser ...")
    blocks = detect_blocks_layoutparser(args.input_pdf, page_num=0)
    block_texts = [b["text"] for b in blocks if b["type"] in ("Text", "Title", "List")]
    print(f"[INFO] Translating {len(block_texts)} visually detected blocks ...")
    translations = batch_translate(block_texts, args.lang)
    print(f"[INFO] Generating translated PDF (first page only) ...")
    doc = fitz.open(args.input_pdf)
    page = doc[0]
    # Redact all detected text blocks first
    text_blocks = [b for b in blocks if b["type"] in ("Text", "Title", "List")]
    for block in text_blocks:
        x0, y0, x1, y1 = block["bbox"]
        padding = 2
        redact_rect = fitz.Rect(x0-padding, y0-padding, x1+padding, y1+padding)
        page.add_redact_annot(redact_rect, text="", fill=(1, 1, 1))
    page.apply_redactions()
    # Insert translated text, block by block
    for block, translated in zip(text_blocks, translations):
        x0, y0, x1, y1 = block["bbox"]
        max_width = x1 - x0 - 4
        max_height = y1 - y0 - 4
        font_size = max(5, int(block.get("size", 12)))
        font_name = "helv"
        safe_text = ''.join([c if ord(c) < 128 or (32 <= ord(c) <= 126) else '?' for c in translated])
        lines, fitted_font_size = wrap_text_to_fit(page, safe_text, font_name, font_size, max_width, max_height)
        if fitted_font_size < 5:
            print(f"[WARN] Could not fit text in block at page 0 bbox {block['bbox']}")
        line_height = fitted_font_size * 1.15
        current_y = y0 + fitted_font_size + 2
        for line in lines:
            if current_y < y1 + 3:
                try:
                    page.insert_text((x0 + 2, current_y), line, fontsize=fitted_font_size, color=(0,0,0), fontname=font_name)
                except Exception as e:
                    print(f"[WARN] Could not insert text: {line[:30]}... Error: {e}")
            current_y += line_height
    doc.save(args.output_pdf)
    doc.close()
    print(f"[PyMuPDF] Translated PDF saved to: {args.output_pdf}")
    print("[DONE]")

if __name__ == "__main__":
    main() 