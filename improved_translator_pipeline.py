import fitz  # PyMuPDF
import re
import json
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import os
import io
from reportlab.lib.utils import ImageReader
import openai
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics import renderPDF
from reportlab.lib.colors import Color, red, blue, black
import logging

# --- Field Mapping ---
FIELD_MAP = {
    "Factuurnummer": "Invoice Number",
    "Factuurdatum": "Invoice Date",
    "Bestelnummer": "Order Number",
    "Besteldatum": "Order Date",
    "Bezorgwijze": "Delivery Method",
    "Levernummer": "Delivery Number",
    "Bezorgdatum": "Delivery Date",
    "Betaalwijze": "Payment Method",
    "Voorgaand document": "Previous Document",
    "Totaal": "Total",
    "Restbedrag": "Remaining Amount",
    # Add more as needed
}

ADDRESS_REGEX = re.compile(r"\d{4}\s?[A-Z]{2}\s[A-Za-z]+|[0-9]{9,}")

def map_field(field):
    return FIELD_MAP.get(field.strip(), field)

def should_translate(line):
    if ADDRESS_REGEX.search(line):
        return False
    return True

# --- Table Formatting ---
def format_table_as_text(table: dict) -> str:
    if not table.get("rows"):
        return ""
    text_parts = []
    if table["type"] == "product_table":
        text_parts.append("PRODUCT TABLE:")
        text_parts.append("Headers: " + " | ".join(table["headers"]))
        for row in table["rows"]:
            row_text = " | ".join(str(row.get(h, "")) for h in table["headers"])
            text_parts.append(row_text)
    elif table["type"] == "vat_table":
        text_parts.append("VAT CALCULATION TABLE:")
        text_parts.append("Headers: " + " | ".join(table["headers"]))
        for row in table["rows"]:
            row_text = " | ".join(str(row.get(h, "")) for h in table["headers"])
            text_parts.append(row_text)
    elif table["type"] == "summary_table":
        text_parts.append("SUMMARY TABLE:")
        for row in table["rows"]:
            text_parts.append(f"{row['label']}: {row['value']}")
    return "\n".join(text_parts)

# --- Main Pipeline ---
class ImprovedPDFTranslator:
    def __init__(self, target_lang='en'):
        self.target_lang = target_lang

    def is_nontranslatable(self, text):
        # Only digits, or digits with punctuation, or a single word with no letters
        if re.fullmatch(r"[0-9.,/_-]+", text.strip()):
            return True
        # Looks like a code (e.g., 261891936_1)
        if re.fullmatch(r"[A-Za-z0-9_/-]+", text.strip()) and not any(c.isalpha() for c in text):
            return True
        return False

    def translate_text(self, text):
        if not text.strip():
            return text
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": text}],
                max_tokens=256,
                temperature=0.2,
            ).choices[0].message.content.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def gpt_translate(self, text, source_lang='nl', target_lang='en'):
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = (
            f"Translate the following Dutch invoice text to English, using correct business and financial terminology. "
            f"Preserve the meaning and context, and do not translate names or numbers. Text: {text}"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def gpt_translate_batch(self, texts, source_lang='nl', target_lang='en'):
        """Translate multiple texts in a single GPT API call"""
        if not texts:
            return []
            
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Create a numbered list of texts for the prompt
        numbered_texts = "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))
        prompt = (
            f"Translate the following Dutch invoice texts to English, using correct business and financial terminology. "
            f"Preserve the meaning and context, and do not translate names or numbers. "
            f"Return ONLY a numbered list of translations in the same order as the input. "
            f"Texts to translate:\n{numbered_texts}"
        )
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,  # Increased for batch
                temperature=0.2,
            )
            
            # Parse the numbered list response
            result = response.choices[0].message.content.strip()
            translations = []
            
            # Extract translations from the numbered list
            for line in result.split('\n'):
                # Match lines like "1. translation" or "1) translation"
                match = re.match(r'^\d+[.)]\s*(.*)', line.strip())
                if match:
                    translations.append(match.group(1).strip())
                else:
                    # If parsing fails, add the original text
                    translations.append(texts[len(translations)])
            
            # Ensure we have the same number of translations as inputs
            while len(translations) < len(texts):
                translations.append(texts[len(translations)])
                
            return translations
            
        except Exception as e:
            print(f"Batch translation error: {e}")
            # Return original texts if translation fails
            return texts

    def map_to_reportlab_font(self, font_name):
        font_name = font_name.lower()
        if 'bold' in font_name and ('oblique' in font_name or 'italic' in font_name):
            return 'Helvetica-BoldOblique'
        elif 'bold' in font_name:
            return 'Helvetica-Bold'
        elif 'oblique' in font_name or 'italic' in font_name:
            return 'Helvetica-Oblique'
        else:
            return 'Helvetica'

    def log_drawings_for_debug(self, pdf_path, output_json="drawings_debug.json"): 
        import fitz
        import json
        doc = fitz.open(pdf_path)
        all_drawings = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            drawings = page.get_drawings()
            for drawing in drawings:
                drawing_copy = dict(drawing)
                # Convert colors to list for JSON
                if 'color' in drawing_copy and drawing_copy['color'] is not None:
                    drawing_copy['color'] = list(drawing_copy['color'])
                if 'fill' in drawing_copy and drawing_copy['fill'] is not None:
                    drawing_copy['fill'] = list(drawing_copy['fill'])
                # Convert items to list
                drawing_copy['items'] = [list(item) if isinstance(item, tuple) else item for item in drawing_copy.get('items',[])]
                drawing_copy['page'] = page_num
                all_drawings.append(drawing_copy)
        with open(output_json, "w") as f:
            json.dump(all_drawings, f, indent=2)
        print(f"Logged all drawing items to {output_json}")

    def create_colored_shapes_debug_pdf(self, original_pdf_path, output_pdf_path="shapes_only_colored.pdf"):
        import fitz
        from reportlab.pdfgen import canvas
        from reportlab.graphics.shapes import Drawing, Rect, Line, Path, Polygon, Circle, Ellipse
        from reportlab.graphics import renderPDF
        from reportlab.lib.colors import Color, black, red, blue
        from reportlab.lib.utils import ImageReader
        import io
        from PIL import Image
        doc = fitz.open(original_pdf_path)
        c = canvas.Canvas(output_pdf_path, pagesize=A4)
        width, height = A4
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            d = Drawing(width, height)
            drawings = page.get_drawings()
            print(f"[DEBUG] Page {page_num} has {len(drawings)} drawing items.")
            for drawing in drawings:
                stroke = drawing.get("color")
                fill = drawing.get("fill")
                stroke_color = Color(*stroke) if stroke else (Color(*fill) if fill else black)
                fill_color = Color(*fill) if fill is not None else None
                stroke_width = drawing.get("width") or 1
                for item in drawing["items"]:
                    item_type = item[0]
                    item_data = item[1]
                    print(f"[DEBUG] Page {page_num} Drawing item: {item_type}, Data: {item_data}")
                    # Draw a blue rectangle for every shape's bounding box if possible
                    if item_type in ["re", "rect"]:
                        x0, y0, x1, y1 = item_data
                        rect = Rect(x0, height - y1, x1-x0, y1-y0, strokeColor=stroke_color, fillColor=fill_color, strokeWidth=stroke_width)
                        d.add(rect)
                        # Add debug rectangle
                        debug_rect = Rect(x0, height - y1, x1-x0, y1-y0, strokeColor=blue, fillColor=None, strokeWidth=2)
                        d.add(debug_rect)
                    elif item_type == "l":
                        if not hasattr(self, '_rect_points'):
                            self._rect_points = []
                        self._rect_points.append((item_data[0], height - item_data[1]))
                        if len(self._rect_points) == 4:
                            x0 = min(p[0] for p in self._rect_points)
                            y0 = min(p[1] for p in self._rect_points)
                            x1 = max(p[0] for p in self._rect_points)
                            y1 = max(p[1] for p in self._rect_points)
                            rect = Rect(x0, y0, x1-x0, y1-y0, strokeColor=stroke_color, fillColor=fill_color, strokeWidth=stroke_width)
                            d.add(rect)
                            debug_rect = Rect(x0, y0, x1-x0, y1-y0, strokeColor=blue, fillColor=None, strokeWidth=2)
                            d.add(debug_rect)
                            self._rect_points = []
                        else:
                            if len(self._rect_points) == 2:
                                x0, y0 = self._rect_points[0]
                                x1, y1 = self._rect_points[1]
                                line = Line(x0, y0, x1, y1)
                                line.strokeColor = stroke_color
                                line.strokeWidth = stroke_width
                                d.add(line)
                                self._rect_points = []
                    elif item_type == "line":
                        x0, y0, x1, y1 = item_data
                        line = Line(x0, height - y0, x1, height - y1)
                        line.strokeColor = stroke_color
                        line.strokeWidth = stroke_width
                        d.add(line)
                        debug_rect = Rect(min(x0, x1), min(height - y0, height - y1), abs(x1-x0), abs(y1-y0), strokeColor=blue, fillColor=None, strokeWidth=2)
                        d.add(debug_rect)
                    elif item_type == "curve":
                        points = [(x, height - y) for (x, y) in item_data]
                        if len(points) >= 2:
                            p = Path(points, strokeColor=stroke_color, fillColor=fill_color, strokeWidth=stroke_width)
                            d.add(p)
                    elif item_type == "polyline":
                        points = [(x, height - y) for (x, y) in item_data]
                        if len(points) >= 2:
                            p = Polygon(points, strokeColor=stroke_color, fillColor=fill_color, strokeWidth=stroke_width)
                            d.add(p)
                    elif item_type == "circle":
                        x, y, r = item_data
                        c_ = Circle(x, height - y, r, strokeColor=stroke_color, fillColor=fill_color, strokeWidth=stroke_width)
                        d.add(c_)
                    elif item_type == "ellipse":
                        x, y, rx, ry = item_data
                        e = Ellipse(x-rx, height - y - ry, 2*rx, 2*ry, strokeColor=stroke_color, fillColor=fill_color, strokeWidth=stroke_width)
                        d.add(e)
                    else:
                        print(f"[DEBUG] Unhandled shape type: {item_type}")
            # Log and draw all images from get_images(full=True)
            image_list = page.get_images(full=True)
            print(f"[DEBUG] Page {page_num} has {len(image_list)} images from get_images(full=True)")
            for img_info in image_list:
                xref = img_info[0]
                print(f"[DEBUG] Image xref: {xref}")
            # Draw all images using get_image_info (if available)
            if hasattr(page, 'get_image_info'):
                img_infos = page.get_image_info()
                print(f"[DEBUG] Page {page_num} has {len(img_infos)} images from get_image_info()")
                for img_info in img_infos:
                    xref = img_info.get('xref')
                    bbox = img_info.get('bbox')
                    print(f"[DEBUG] get_image_info xref: {xref}, bbox: {bbox}")
                    if xref and bbox:
                        x0, y0, x1, y1 = bbox
                        # Draw a red rectangle for every image bbox
                        debug_rect = Rect(x0, height - y1, x1-x0, y1-y0, strokeColor=red, fillColor=None, strokeWidth=2)
                        d.add(debug_rect)
                        # Try to draw the image (if possible)
                        try:
                            base_image = doc.extract_image(xref)
                            img_bytes = base_image["image"]
                            img = Image.open(io.BytesIO(img_bytes))
                            c.drawImage(ImageReader(img), x0, height - y1, width=x1-x0, height=y1-y0, preserveAspectRatio=True)
                        except Exception as e:
                            print(f"[DEBUG] Could not render image xref {xref}: {e}")
            renderPDF.draw(d, c, 0, 0)
            c.showPage()
        c.save()
        doc.close()
        print(f"[DEBUG] Created {output_pdf_path} with only colored shapes/lines (and images).")

    def create_translated_pdf(self, blocks, output_pdf_path, original_pdf_path):
        import io
        from reportlab.lib.utils import ImageReader
        from reportlab.graphics.shapes import Drawing, Rect, Line, Path, Polygon, Circle, Ellipse
        from reportlab.graphics import renderPDF
        from reportlab.lib.colors import Color, black, gray, red
        from PIL import Image
        import fitz
        import logging
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        doc = fitz.open(original_pdf_path)
        c = canvas.Canvas(output_pdf_path, pagesize=A4)
        width, height = A4
        
        # Create debug visualization
        debug_canvas = canvas.Canvas("debug_shapes.pdf", pagesize=A4)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            d = Drawing(width, height)
            debug_d = Drawing(width, height)
            
            # Process all drawings
            try:
                drawings = page.get_drawings()
                logger.info(f"Page {page_num + 1} has {len(drawings)} drawings")
                
                for i, drawing in enumerate(drawings):
                    stroke = drawing.get("color")
                    fill = drawing.get("fill")
                    stroke_color = Color(*stroke) if stroke else (Color(*fill) if fill else black)
                    fill_color = Color(*fill) if fill is not None else None
                    stroke_width = drawing.get("width") or 1
                    for item in drawing["items"]:
                        item_type = item[0]
                        item_data = item[1]
                        logger.info(f"[DEBUG] Drawing item type: {item_type}, Used Stroke: {tuple(stroke_color.rgb()) if stroke_color else None}, Used Fill: {tuple(fill_color.rgb()) if fill_color else None}, Data: {item_data}")
                        if item_type == "l":
                            if not hasattr(self, '_rect_points'):
                                self._rect_points = []
                            self._rect_points.append((item_data[0], height - item_data[1]))
                            if len(self._rect_points) == 4:
                                x0 = min(p[0] for p in self._rect_points)
                                y0 = min(p[1] for p in self._rect_points)
                                x1 = max(p[0] for p in self._rect_points)
                                y1 = max(p[1] for p in self._rect_points)
                                rect = Rect(x0, y0, x1-x0, y1-y0, strokeColor=stroke_color, fillColor=fill_color, strokeWidth=stroke_width)
                                d.add(rect)
                                debug_d.add(rect)
                                self._rect_points = []
                            else:
                                if len(self._rect_points) == 2:
                                    x0, y0 = self._rect_points[0]
                                    x1, y1 = self._rect_points[1]
                                    line = Line(x0, y0, x1, y1)
                                    line.strokeColor = stroke_color
                                    line.strokeWidth = stroke_width
                                    d.add(line)
                                    debug_d.add(line)
                                    self._rect_points = []
                        elif item_type in ["re", "rect"]:
                            x0, y0, x1, y1 = item_data
                            rect = Rect(x0, height - y1, x1-x0, y1-y0, strokeColor=stroke_color, fillColor=fill_color, strokeWidth=stroke_width)
                            d.add(rect)
                            debug_d.add(rect)
                        elif item_type == "line":
                            x0, y0, x1, y1 = item_data
                            line = Line(x0, height - y0, x1, height - y1)
                            line.strokeColor = stroke_color
                            line.strokeWidth = stroke_width
                            d.add(line)
                            debug_d.add(line)
                        elif item_type == "curve":
                            points = [(x, height - y) for (x, y) in item_data]
                            if len(points) >= 2:
                                p = Path(points, strokeColor=stroke_color, fillColor=fill_color, strokeWidth=stroke_width)
                                d.add(p)
                                debug_d.add(p)
                        elif item_type == "polyline":
                            points = [(x, height - y) for (x, y) in item_data]
                            if len(points) >= 2:
                                p = Polygon(points, strokeColor=stroke_color, fillColor=fill_color, strokeWidth=stroke_width)
                                d.add(p)
                                debug_d.add(p)
                        elif item_type == "circle":
                            x, y, r = item_data
                            c_ = Circle(x, height - y, r, strokeColor=stroke_color, fillColor=fill_color, strokeWidth=stroke_width)
                            d.add(c_)
                            debug_d.add(c_)
                        elif item_type == "ellipse":
                            x, y, rx, ry = item_data
                            e = Ellipse(x-rx, height - y - ry, 2*rx, 2*ry, strokeColor=stroke_color, fillColor=fill_color, strokeWidth=stroke_width)
                            d.add(e)
                            debug_d.add(e)
                        else:
                            logger.warning(f"Unhandled shape type: {item_type}")
            except Exception as e:
                logger.error(f"Error processing drawings: {e}")
                if hasattr(self, '_rect_points'):
                    delattr(self, '_rect_points')
            # Draw all images using get_image_info (if available)
            if hasattr(page, 'get_image_info'):
                for img_info in page.get_image_info():
                    xref = img_info.get('xref')
                    bbox = img_info.get('bbox')
                    if xref and bbox:
                        x0, y0, x1, y1 = bbox
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        img = Image.open(io.BytesIO(img_bytes))
                        c.drawImage(ImageReader(img), x0, height - y1, width=x1-x0, height=y1-y0, preserveAspectRatio=True)
                        debug_rect = Rect(x0, height - y1, x1-x0, y1-y0, strokeColor=red, fillColor=None, strokeWidth=1)
                        debug_d.add(debug_rect)
                        logger.info(f"[DEBUG] drew image at ({x0},{y0},{x1},{y1})")
            renderPDF.draw(d, c, 0, 0)
            for block in blocks:
                if block["page"] != page_num:
                    continue
                x0, y0, x1, y1 = map(int, block["bbox"])
                font_size = int(block.get("size", 12))
                text = block["translated"]
                font_name = self.map_to_reportlab_font(block.get("font", "Helvetica"))
                c.setFillColorRGB(0, 0, 0)
                c.setFont(font_name, font_size)
                y = height - y0 - font_size
                c.drawString(x0, y, text)
            renderPDF.draw(debug_d, debug_canvas, 0, 0)
            debug_canvas.showPage()
            c.showPage()
        c.save()
        debug_canvas.save()
        doc.close()
        logger.info("PDF creation complete. Check debug_shapes.pdf for visualization of shapes and images.")

    def match_casing(self, original, translated):
        # Single word
        if len(original.split()) == 1 and len(translated.split()) == 1:
            orig = original.strip()
            trans = translated.strip()
            if orig.isupper():
                return trans.upper()
            elif orig.istitle():
                return trans.capitalize()
            elif orig.islower():
                return trans.lower()
            else:
                return trans
        # Multi-word: match block-level casing
        if original.isupper():
            return translated.upper()
        elif original.istitle():
            return translated.title()
        elif original.islower():
            return translated.lower()
        else:
            # fallback to word-by-word as before
            return self.match_casing_word_by_word(original, translated)

    def match_casing_word_by_word(self, original, translated):
        def apply_case(orig_word, trans_word):
            if orig_word.isupper():
                return trans_word.upper()
            elif orig_word.istitle():
                return trans_word.capitalize()
            elif orig_word.islower():
                return trans_word.lower()
            else:
                return trans_word
        orig_words = original.split()
        trans_words = translated.split()
        matched = []
        for i, trans_word in enumerate(trans_words):
            if i < len(orig_words):
                matched.append(apply_case(orig_words[i], trans_word))
            else:
                matched.append(trans_word)
        return ' '.join(matched)

    def is_gpt_explanation(self, text, original=None):
        t = text.lower()
        # Existing checks
        if (
            "does not need to be translated" in t or
            "does not contain any text to be translated" in t or
            "no dutch invoice text provided" in t or
            "does not provide any context or meaning to be translated" in t or
            "please provide a complete sentence or phrase" in t
        ):
            return True
        # If output is much longer than input and input is very short, treat as explanation/template
        if original and len(original.strip()) < 5 and len(text.strip()) > 30:
            return True
        # If output contains Dutch invoice keywords and original is short, treat as template
        if (
            ("factuurnummer" in t and "datum" in t and len(original.strip()) < 10)
            or "geachte klant" in t
            or "bedankt voor uw aankoop" in t
        ):
            return True
        # If translation is much longer than original (e.g., >3x)
        if original and len(text.strip()) > 3 * max(1, len(original.strip())):
            return True
        return False

    def visualize_text_blocks(self, blocks, original_pdf_path, output_path):
        """Generate a visualization of text block rectangles on the original PDF, with header-based table detection and local grid clustering."""
        import io
        from reportlab.lib.utils import ImageReader
        from reportlab.graphics.shapes import Drawing, Rect, String
        from reportlab.graphics import renderPDF
        from reportlab.lib.colors import Color, red, blue, green, orange, purple, brown, pink, yellow, black, gray
        import random
        
        doc = fitz.open(original_pdf_path)
        c = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4
        
        def get_block_key(block):
            return tuple(map(int, block["bbox"]))
        
        def find_header_rows(blocks, min_cells=2):
            # Find rows with multiple blocks, bold font, or all-caps
            # Return list of (header_blocks, y0)
            if not blocks:
                return []
            # Group blocks by y0 (row clustering)
            y0s = sorted(set(int(b["bbox"][1]) for b in blocks))
            header_rows = []
            for y in y0s:
                row_blocks = [b for b in blocks if abs(int(b["bbox"][1]) - y) < 10]
                if len(row_blocks) >= min_cells:
                    # Check for bold or all-caps
                    bold_count = sum('bold' in b.get('font','').lower() for b in row_blocks)
                    caps_count = sum(b['text'].isupper() for b in row_blocks if b['text'])
                    if bold_count > 0 or caps_count > 0 or len(row_blocks) >= 3:
                        header_rows.append((row_blocks, y))
            return header_rows
        
        def assign_blocks_to_table(header_blocks, all_blocks, y0_header, row_thresh=20, col_thresh=30):
            # Use header x0/x1 as column boundaries
            col_bounds = [(int(b["bbox"][0]), int(b["bbox"][2])) for b in header_blocks]
            col_bounds = sorted(col_bounds, key=lambda x: x[0])
            # Collect all blocks below header that align with these columns
            table_blocks = [b for b in all_blocks if int(b["bbox"][1]) >= y0_header - 5]
            # Cluster by row (y0)
            y0s = sorted(set(int(b["bbox"][1]) for b in table_blocks))
            row_anchors = []
            for y in y0s:
                if not row_anchors or abs(y - row_anchors[-1]) > row_thresh:
                    row_anchors.append(y)
            # Assign each block to closest (row, col) by overlap
            cell_map = {}
            for b in table_blocks:
                y0 = int(b["bbox"][1])
                x0 = int(b["bbox"][0])
                x1 = int(b["bbox"][2])
                # Find closest row
                row = min(range(len(row_anchors)), key=lambda i: abs(y0 - row_anchors[i]))
                # Find column with max overlap
                max_overlap = 0
                col = 0
                for i, (cb0, cb1) in enumerate(col_bounds):
                    overlap = max(0, min(x1, cb1) - max(x0, cb0))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        col = i
                cell_map[get_block_key(b)] = (row, col)
            # Compute bounding box
            xs = [int(b["bbox"][0]) for b in table_blocks] + [int(b["bbox"][2]) for b in table_blocks]
            ys = [int(b["bbox"][1]) for b in table_blocks] + [int(b["bbox"][3]) for b in table_blocks]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            return cell_map, row_anchors, col_bounds, bbox, table_blocks
        
        group_colors = [blue, green, red, orange, purple, brown, pink, yellow]
        def get_group_color(group):
            if group < len(group_colors):
                return group_colors[group]
            random.seed(group)
            return Color(random.random(), random.random(), random.random())
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            img_stream = io.BytesIO(img_bytes)
            c.drawImage(ImageReader(img_stream), 0, 0, width, height)
            d = Drawing(width, height)
            page_blocks = [b for b in blocks if b["page"] == page_num]
            # Header-based table detection
            header_rows = find_header_rows(page_blocks, min_cells=2)
            table_blocks = set()
            for header_blocks, y0_header in header_rows:
                cell_map, row_anchors, col_bounds, bbox, region_blocks = assign_blocks_to_table(header_blocks, page_blocks, y0_header, row_thresh=20, col_thresh=30)
                # Draw region rectangle
                d.add(Rect(bbox[0], height-bbox[3], bbox[2]-bbox[0], bbox[3]-bbox[1], strokeColor=orange, fillColor=None, strokeWidth=2))
                # Draw column boundaries as vertical lines
                for cb0, cb1 in col_bounds:
                    d.add(
                        renderPDF.Line(cb0, height-bbox[3], cb0, height-bbox[1], strokeColor=gray, strokeWidth=1)
                    )
                # Draw grid and label
                for b in region_blocks:
                    x0, y0, x1, y1 = map(int, b["bbox"])
                    font_size = int(b.get("size", 12))
                    block_key = get_block_key(b)
                    if block_key in cell_map:
                        row, col = cell_map[block_key]
                        color = get_group_color(col)
                        # Snap x0/x1 to column boundaries, add padding
                        pad = max(2, int(font_size * 0.15))
                        snap_x0 = col_bounds[col][0] + pad
                        snap_x1 = col_bounds[col][1] - pad
                        rect = Rect(
                            snap_x0,
                            height - y1 - pad,
                            snap_x1 - snap_x0,
                            y1 - y0 + 2*pad,
                            fillColor=Color(color.red, color.green, color.blue, alpha=0.08),
                            strokeColor=color,
                            strokeWidth=1
                        )
                        d.add(rect)
                        label = f"H({row+1},{col+1})"
                        text_obj = String(
                            snap_x0 + 2,
                            height - y0 + 5,
                            label,
                            fontSize=8,
                            fillColor=color
                        )
                        d.add(text_obj)
                        table_blocks.add(get_block_key(b))
            # For blocks not in any table region, do global clustering
            non_table_blocks = [b for b in page_blocks if get_block_key(b) not in table_blocks]
            # Use previous clustering for non-table blocks
            def cluster_by_row(blocks_on_page, threshold=15):
                if not blocks_on_page:
                    return {}, []
                y0s = sorted(set(int(b["bbox"][1]) for b in blocks_on_page))
                clusters = []
                for y in y0s:
                    found = False
                    for cluster in clusters:
                        if abs(cluster[0] - y) <= threshold:
                            cluster.append(y)
                            found = True
                            break
                    if not found:
                        clusters.append([y])
                cluster_anchors = [int(sum(cluster)/len(cluster)) for cluster in clusters]
                group_map = {}
                for block in blocks_on_page:
                    y0 = int(block["bbox"][1])
                    min_dist = float('inf')
                    group = 0
                    for i, anchor in enumerate(cluster_anchors):
                        dist = abs(y0 - anchor)
                        if dist < min_dist:
                            min_dist = dist
                            group = i
                    group_map[get_block_key(block)] = group
                return group_map, cluster_anchors
            def hybrid_cluster_by_column(blocks_on_page, threshold=50):
                if not blocks_on_page:
                    return {}, []
                anchors = []
                for block in blocks_on_page:
                    x0 = int(block["bbox"][0])
                    x1 = int(block["bbox"][2])
                    xc = int((x0 + x1) / 2)
                    anchors.extend([x0, x1, xc])
                anchors = sorted(set(anchors))
                clusters = []
                for x in anchors:
                    found = False
                    for cluster in clusters:
                        if abs(cluster[0] - x) <= threshold:
                            cluster.append(x)
                            found = True
                            break
                    if not found:
                        clusters.append([x])
                cluster_anchors = [int(sum(cluster)/len(cluster)) for cluster in clusters]
                group_map = {}
                for block in blocks_on_page:
                    x0 = int(block["bbox"][0])
                    x1 = int(block["bbox"][2])
                    xc = int((x0 + x1) / 2)
                    block_points = [x0, x1, xc]
                    min_dist = float('inf')
                    group = 0
                    for i, anchor in enumerate(cluster_anchors):
                        for pt in block_points:
                            dist = abs(pt - anchor)
                            if dist < min_dist:
                                min_dist = dist
                                group = i
                    group_map[get_block_key(block)] = group
                return group_map, cluster_anchors
            row_map, row_anchors = cluster_by_row(non_table_blocks, threshold=15)
            col_map, col_anchors = hybrid_cluster_by_column(non_table_blocks, threshold=50)
            for block in non_table_blocks:
                x0, y0, x1, y1 = map(int, block["bbox"])
                font_size = int(block.get("size", 12))
                block_key = get_block_key(block)
                row = row_map.get(block_key, 0)
                col = col_map.get(block_key, 0)
                color = get_group_color(col)
                padding = font_size * 0.2
                rect = Rect(
                    x0 - padding,
                    height - y1 - padding,
                    x1 - x0 + 2*padding,
                    y1 - y0 + 2*padding,
                    fillColor=Color(color.red, color.green, color.blue, alpha=0.08),
                    strokeColor=color,
                    strokeWidth=1
                )
                d.add(rect)
                label = f"({row+1},{col+1})"
                text_obj = String(
                    x0,
                    height - y0 + 5,
                    label,
                    fontSize=8,
                    fillColor=color
                )
                d.add(text_obj)
            renderPDF.draw(d, c, 0, 0)
            c.showPage()
        c.save()
        doc.close()

    def process_pdf(self, pdf_path, output_pdf_path, layout_img_path, translated_img_path, diff_img_path):
        doc = fitz.open(pdf_path)
        blocks = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_dict = page.get_text("dict")
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = "".join(span["text"] for span in line["spans"]).strip()
                        bbox = line["bbox"]
                        font = line["spans"][0].get("font", "Helvetica")
                        size = line["spans"][0].get("size", 12)
                        blocks.append({
                            "text": line_text,
                            "bbox": bbox,
                            "font": font,
                            "size": size,
                            "page": page_num
                        })
        doc.close()

        # --- Generate block visualization ---
        block_viz_path = "text_blocks_visualization.pdf"
        self.visualize_text_blocks(blocks, pdf_path, block_viz_path)
        print(f"Generated text block visualization: {block_viz_path}")

        # --- Translate and Map Fields ---
        translated_blocks = []
        
        # Group blocks into batches for translation
        BATCH_SIZE = 10  # Adjust based on your needs
        current_batch = []
        current_batch_indices = []
        
        for i, block in enumerate(blocks):
            text = block["text"]
            if not text.strip() or self.is_nontranslatable(text):
                # Handle non-translatable blocks immediately
                block["translated"] = text
                translated_blocks.append(block)
                continue
                
            current_batch.append(text)
            current_batch_indices.append(i)
            
            # Process batch when it reaches the size limit or at the end
            if len(current_batch) >= BATCH_SIZE or i == len(blocks) - 1:
                if current_batch:
                    print(f"Translating batch of {len(current_batch)} texts...")
                    try:
                        translations = self.gpt_translate_batch(current_batch)
                        
                        # Process each translation in the batch
                        for batch_idx, (orig_text, translated) in enumerate(zip(current_batch, translations)):
                            block_idx = current_batch_indices[batch_idx]
                            block = blocks[block_idx]
                            
                            # Check for GPT explanation/template responses
                            if self.is_gpt_explanation(translated, orig_text):
                                translated = orig_text
                                
                            # Match casing
                            translated = self.match_casing(orig_text, translated)
                            
                            # Store the translation
                            block["translated"] = translated
                            translated_blocks.append(block)
                            
                            print(f"Translated: {orig_text} -> {translated}")
                            
                    except Exception as e:
                        print(f"Batch translation failed: {e}")
                        # Fall back to individual translations for this batch
                        for batch_idx, orig_text in enumerate(current_batch):
                            block_idx = current_batch_indices[batch_idx]
                            block = blocks[block_idx]
                            try:
                                translated = self.gpt_translate(orig_text)
                                if self.is_gpt_explanation(translated, orig_text):
                                    translated = orig_text
                            except Exception as e:
                                print(f"Individual translation failed for: {orig_text}\nError: {e}")
                                translated = orig_text
                            translated = self.match_casing(orig_text, translated)
                            block["translated"] = translated
                            translated_blocks.append(block)
                    
                    # Clear the batch
                    current_batch = []
                    current_batch_indices = []

        # --- Save as JSON for inspection ---
        with open("translated_blocks.json", "w", encoding="utf-8") as f:
            json.dump(translated_blocks, f, ensure_ascii=False, indent=2)

        # --- Create Translated PDF (with lines/images) ---
        self.create_translated_pdf(translated_blocks, output_pdf_path, pdf_path)

        # --- Render Layout Images ---
        self.render_layout_image(blocks, layout_img_path)
        self.render_layout_image(translated_blocks, translated_img_path, use_translated=True)

        # --- Visual Diff ---
        percent_diff = self.visual_diff(layout_img_path, translated_img_path, diff_img_path)
        print(f"Visual diff: {percent_diff:.2f}% of pixels differ.")

        # --- Auto-correct if needed ---
        if percent_diff > 10.0:
            print("WARNING: Diff exceeds threshold! Consider adjusting font size or block positions.")

    def render_layout_image(self, blocks, out_path, use_translated=False):
        # Simple white background
        img = Image.new("RGB", (1200, 1700), "white")
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        for block in blocks:
            x0, y0, x1, y1 = map(int, block["bbox"])
            text = block["translated"] if use_translated else block["text"]
            font_size = int(block.get("size", 12))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            draw.rectangle([x0, y0, x1, y1], outline="gray", width=1)
            draw.text((x0+2, y0+2), text, fill="black", font=font)
        img.save(out_path)

    def visual_diff(self, img1_path, img2_path, diff_path):
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)
        np_img1 = np.array(img1)
        np_img2 = np.array(img2)
        diff = np.abs(np_img1.astype(int) - np_img2.astype(int)).astype(np.uint8)
        highlight = np.zeros_like(np_img1)
        highlight[..., 0] = diff.max(axis=2)
        highlight_img = Image.fromarray(highlight)
        blended = Image.blend(img1, highlight_img, alpha=0.5)
        blended.save(diff_path)
        red_pixels = np.sum(highlight[..., 0] > 32)
        total_pixels = highlight[..., 0].size
        percent_diff = 100 * red_pixels / total_pixels
        return percent_diff

    def visualize_page_elements(self, pdf_path, output_path):
        """Visualize all shapes, lines, and images from the first page of a PDF."""
        import io
        from reportlab.lib.utils import ImageReader
        from reportlab.graphics.shapes import Drawing, Rect, Line, Path
        from reportlab.graphics import renderPDF
        from reportlab.lib.colors import Color, black, gray
        from PIL import Image
        import fitz
        
        doc = fitz.open(pdf_path)
        c = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4
        page = doc.load_page(0)
        d = Drawing(width, height)
        # Draw all vector shapes/lines
        try:
            for drawing in page.get_drawings():
                stroke = drawing.get("color", (0, 0, 0))
                fill = drawing.get("fill")
                stroke_color = Color(*stroke) if stroke else black
                # Always draw a filled rect if fill is present (even if it is (1,1,1) or (0,0,0))
                fill_color = Color(*fill) if fill is not None else None
                for item in drawing["items"]:
                    if item[0] == "rect":
                        x0, y0, x1, y1 = item[1]
                        rect = Rect(x0, height-y1, x1-x0, y1-y0, strokeColor=stroke_color, fillColor=fill_color, strokeWidth=1)
                        d.add(rect)
                    elif item[0] == "line":
                        x0, y0, x1, y1 = item[1]
                        line = Line(x0, height-y0, x1, height-y1)
                        line.strokeColor = stroke_color
                        line.strokeWidth = 1
                        d.add(line)
                    elif item[0] == "curve":
                        # Assume item[1] is a list of (x,y) tuples (or a list of points) for the curve.
                        # Convert (x,y) to (x, height-y) for reportlab (flip y) and build a Path.
                        points = [(x, height - y) for (x, y) in item[1]]
                        if len(points) >= 2:
                            p = Path(points, strokeColor=stroke_color, fillColor=fill_color, strokeWidth=1)
                            d.add(p)
                    # (You can add more cases for other path/shape types if needed.)
        except Exception as e:
            print(f"Error extracting vector shapes: {e}")
        # Draw all images
        try:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img = Image.open(io.BytesIO(img_bytes))
                # For now, just place at (0,0) as a placeholder
                c.drawImage(ImageReader(img), 0, 0, width=img.width, height=img.height)
        except Exception as e:
            print(f"Error extracting images: {e}")
        renderPDF.draw(d, c, 0, 0)
        c.save()
        doc.close()

if __name__ == "__main__":
    pdf_path = "sample.pdf"
    output_pdf_path = "sample_translated.pdf"
    layout_img_path = "layout_visualization_page_1.png"
    translated_img_path = "translated_layout_page_1.png"
    diff_img_path = "diff_page_1.png"

    translator = ImprovedPDFTranslator(target_lang="en")
    translator.process_pdf(pdf_path, output_pdf_path, layout_img_path, translated_img_path, diff_img_path)
    print("Pipeline complete. Check the output images, diff, and translated PDF for results.") 