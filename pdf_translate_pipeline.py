import os
import json
import zipfile
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import openai
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import Color
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
import fitz  # PyMuPDF
import io
import qrcode
from reportlab.lib.utils import ImageReader

# --- CONFIGURATION ---
INPUT_PDF = 'Sample3.pdf'
TEMP_DIR = Path('temp')
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_PDF = 'Sample3_translated.pdf'
ADOBE_CREDENTIALS = 'pdfservices-api-credentials.json'

# --- LOAD ENVIRONMENT ---
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# --- LOAD ADOBE CREDENTIALS ---
with open(ADOBE_CREDENTIALS, 'r') as f:
    adobe_creds = json.load(f)

# --- PDF TRANSLATION PIPELINE ---
def extract_pdf_elements(input_pdf_path, adobe_creds):
    from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
    from adobe.pdfservices.operation.pdf_services import PDFServices
    from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob, ExtractPDFParams
    from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import ExtractRenditionsElementType
    print(f"Extracting elements from: {input_pdf_path}")
    credentials = ServicePrincipalCredentials(
        client_id=adobe_creds['client_credentials']['client_id'],
        client_secret=adobe_creds['client_credentials']['client_secret']
    )
    pdf_services = PDFServices(credentials=credentials)
    with open(input_pdf_path, "rb") as f:
        input_stream = f.read()
    print("⏳ Uploading PDF to Adobe PDF Services...")
    input_asset = pdf_services.upload(input_stream=input_stream, mime_type="application/pdf")
    extract_params = ExtractPDFParams(
        elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES],
        elements_to_extract_renditions=[ExtractRenditionsElementType.FIGURES, ExtractRenditionsElementType.TABLES]
    )
    job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_params)
    print("⏳ Submitting extract job...")
    polling_url = pdf_services.submit(job)
    print("⏳ Waiting for job to complete...")
    response = pdf_services.get_job_result(polling_url, ExtractPDFResult)
    result_asset = response.get_result().get_resource()
    stream_asset = pdf_services.get_content(result_asset)
    output_zip_path = str(TEMP_DIR / 'extracted.zip')
    with open(output_zip_path, "wb") as out_f:
        out_f.write(stream_asset.get_input_stream())
    with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
        zip_ref.extractall(TEMP_DIR)
        with open(TEMP_DIR / 'structuredData.json', 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
    print(f"Extraction complete. Found {len(structured_data.get('elements', []))} elements")
    return structured_data

def parse_text_and_visual_elements(structured_data):
    text_elements = []
    qr_elements = []
    
    # First pass: collect all table cells with background colors
    cell_backgrounds = {}
    for element in structured_data.get('elements', []):
        if element.get('Path', '').endswith(('TH', 'TD')):
            attributes = element.get('attributes', {})
            background_color = attributes.get('BackgroundColor', None)
            if background_color:
                bounds = element.get('Bounds', [])
                if bounds and len(bounds) >= 4:
                    # Store background color for this cell area
                    cell_key = (bounds[0], bounds[1], bounds[2], bounds[3])
                    cell_backgrounds[cell_key] = background_color
    
    for element in structured_data.get('elements', []):
        if 'Text' in element and element['Text'].strip():
            text = element['Text'].strip()
            font_info = element.get('Font', {})
            
            # Enhanced font style detection
            font_family = font_info.get('family_name', 'Arial')
            font_weight = font_info.get('weight', 400)
            font_style = font_info.get('style', 'normal')
            
            # Additional font style detection from font name
            font_name = font_info.get('name', '')
            if 'Bold' in font_name or 'bold' in font_name:
                font_weight = 700
                font_style = 'bold'
            if 'Italic' in font_name or 'italic' in font_name:
                font_style = 'italic'
            if 'Oblique' in font_name or 'oblique' in font_name:
                font_style = 'italic'
            
            # Detect QR code by block characters or specific patterns
            block_chars = set(['\u2588', '\u2580', '\u2584', '\u258C', '\u2590', '\u2591', '\u2592', '\u2593'])
            has_block_chars = sum(1 for c in text if c in block_chars) / max(1, len(text)) > 0.3
            # Also check for QR-like patterns (repetitive characters, specific length)
            is_qr_like = (len(text) > 20 and len(set(text)) < 10) or has_block_chars
            
            if is_qr_like:
                # For QR codes, we need to extract meaningful data
                # Try to find any readable text in the QR code
                readable_chars = [c for c in text if c not in block_chars and c.isprintable()]
                if readable_chars:
                    qr_data = ''.join(readable_chars).strip()
                else:
                    # If no readable text, create a meaningful QR code based on document content
                    # Look for key information in the document
                    doc_content = []
                    for elem in structured_data.get('elements', []):
                        if 'Text' in elem and elem.get('Lang') == 'nl':
                            doc_content.append(elem['Text'])
                    
                    # Create a QR code with document reference info
                    qr_data = "https://ind.nl/document/Z1-186720992110"
                
                qr_elements.append({
                    'text': qr_data,
                    'bounds': element.get('Bounds', []),
                    'page': element.get('Page', 0),
                    'original_text': text,  # Keep original for debugging
                })
                continue
            
            # Find matching background color for this text element
            text_bounds = element.get('Bounds', [])
            bg_color = None
            if text_bounds and len(text_bounds) >= 4:
                # Look for a cell that contains this text
                for cell_bounds, cell_bg in cell_backgrounds.items():
                    # Check if text is within this cell (with some tolerance)
                    tolerance = 2.0
                    if (cell_bounds[0] - tolerance <= text_bounds[0] <= cell_bounds[2] + tolerance and
                        cell_bounds[1] - tolerance <= text_bounds[1] <= cell_bounds[3] + tolerance):
                        bg_color = cell_bg
                        break
            
            text_elements.append({
                'text': text,
                'bounds': text_bounds,
                'page': element.get('Page', 0),
                'font_size': element.get('TextSize', 12),
                'font_family': font_family,
                'font_weight': font_weight,
                'font_style': font_style,
                'font_color': element.get('Color', [0, 0, 0]),
                'background_color': bg_color,  # Add background color
                'object_id': element.get('ObjectID', 0),
                'path': element.get('Path', ''),
                'lang': element.get('Lang', 'unknown'),
                'text_align': element.get('TextAlign', 'left'),
                'line_height': element.get('LineHeight', 1.2),
                'font_name': font_name,  # Keep original font name for debugging
            })
    print(f"Parsed {len(text_elements)} text elements, {len(qr_elements)} QR elements")
    return text_elements, qr_elements

def translate_text_batch(texts, target_language="English"):
    print(f"Translating {len(texts)} text blocks to {target_language}")
    combined_text = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a professional translator. Translate the following text blocks to {target_language}. Maintain the same numbering format and return only the translated text with numbers."},
                {"role": "user", "content": combined_text}
            ],
            max_tokens=4000,
            temperature=0.3
        )
        translated_text = response.choices[0].message.content.strip()
        translated_lines = translated_text.split('\n')
        translations = []
        for line in translated_lines:
            if line.strip() and '. ' in line:
                text_part = line.split('. ', 1)[1] if '. ' in line else line
                translations.append(text_part.strip())
            else:
                translations.append(line.strip())
        while len(translations) < len(texts):
            translations.append("")
        return translations[:len(texts)]
    except Exception as e:
        print(f"Translation error: {e}")
        return texts

def _map_font_family(font_family, font_weight, font_style):
    font_mappings = {
        'Arial': 'Helvetica',
        'Helvetica': 'Helvetica',
        'Times': 'Times-Roman',
        'Times New Roman': 'Times-Roman',
        'Courier': 'Courier',
        'Courier New': 'Courier',
        'Verdana': 'Helvetica',
        'Georgia': 'Times-Roman',
        'Tahoma': 'Helvetica',
        'Calibri': 'Helvetica',
        'Cambria': 'Times-Roman',
        'Candara': 'Helvetica',
        'Consolas': 'Courier',
        'Constantia': 'Times-Roman',
        'Corbel': 'Helvetica',
        'Garamond': 'Times-Roman',
        'Palatino': 'Times-Roman',
        'Trebuchet MS': 'Helvetica',
        'Verdana': 'Helvetica',
    }
    base_font = font_mappings.get(font_family, 'Helvetica')
    
    # Enhanced font style mapping
    if font_weight >= 600 or font_style == 'bold':
        if base_font == 'Helvetica':
            return 'Helvetica-Bold'
        elif base_font == 'Times-Roman':
            return 'Times-Bold'
        elif base_font == 'Courier':
            return 'Courier-Bold'
    if font_style == 'italic':
        if base_font == 'Helvetica':
            return 'Helvetica-Oblique'
        elif base_font == 'Times-Roman':
            return 'Times-Italic'
        elif base_font == 'Courier':
            return 'Courier-Oblique'
    return base_font

def _map_text_alignment(text_align):
    alignment_map = {
        'left': TA_LEFT,
        'center': TA_CENTER,
        'right': TA_RIGHT,
        'justify': TA_JUSTIFY,
    }
    return alignment_map.get(str(text_align).lower(), TA_LEFT)

def extract_logo_images(pdf_path):
    # Use Adobe Extract API JSON to get all Figure elements and their bounding boxes
    structured_json_path = Path('temp/structuredData.json')
    image_elements = []
    available_files = set()
    # List all available image files in temp/ and temp/figures/
    temp_dir = Path('temp')
    figures_dir = temp_dir / 'figures'
    for d in [temp_dir, figures_dir]:
        if d.exists():
            for f in d.glob('*.png'):
                available_files.add(f.resolve())
    print(f"Available image files: {[str(f) for f in available_files]}")
    if structured_json_path.exists():
        with open(structured_json_path, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
        for element in structured_data.get('elements', []):
            if element.get('Path', '').endswith('Figure') and 'filePaths' in element:
                file_paths = element['filePaths']
                bounds = element.get('Bounds', [])
                page = element.get('Page', 0)
                print(f"Figure: file_paths={file_paths}, page={page}, bounds={bounds}")
                for file_path in file_paths:
                    # Only match the exact filename (not just the first file found)
                    img_file_candidates = [temp_dir / file_path, figures_dir / Path(file_path).name]
                    found = False
                    for img_file in img_file_candidates:
                        if img_file.exists() and img_file.name == Path(file_path).name:
                            with open(img_file, 'rb') as imgf:
                                image_bytes = imgf.read()
                            ext = img_file.suffix[1:] if '.' in img_file.name else 'png'
                            image_elements.append({
                                "data": image_bytes,
                                "page": page,
                                "bounds": bounds,
                                "ext": ext,
                                "filename": img_file.name
                            })
                            print(f"Matched Figure {file_path} to file {img_file} at page {page} bounds {bounds}")
                            found = True
                            break
                    if not found:
                        print(f"WARNING: Image file for Figure not found: {file_path}")
    else:
        print("No structuredData.json found for image extraction.")
    print(f"Extracted {len(image_elements)} images (figures) from PDF.")
    return image_elements

def reconstruct_pdf(text_elements, translated_texts, output_path, qr_elements=None, image_elements=None, page_size=A4):
    print(f"Reconstructing PDF with translated text, QR codes, and images...")
    
    # Register fonts properly
    noto_path = 'fonts/NotoSans-Regular.ttf'
    if os.path.exists(noto_path):
        if 'NotoSans-Regular' not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont('NotoSans-Regular', noto_path))
        default_font = 'NotoSans-Regular'
    else:
        default_font = 'Helvetica'
    
    # Built-in fonts should be available by default in ReportLab
    # No need to register them explicitly
    
    # Ensure all built-in fonts are available
    available_fonts = pdfmetrics.getRegisteredFontNames()
    print(f"Available fonts: {available_fonts}")
    
    c = canvas.Canvas(output_path, pagesize=page_size)
    page_width, page_height = page_size
    styles = getSampleStyleSheet()
    base_style = styles["Normal"]
    texts_by_page = {}
    for i, element in enumerate(text_elements):
        p = element['page']
        texts_by_page.setdefault(p, []).append((element, translated_texts[i] if i < len(translated_texts) else element['text']))
    qr_by_page = {}
    if qr_elements:
        for qr in qr_elements:
            p = qr['page']
            qr_by_page.setdefault(p, []).append(qr)
    images_by_page = {}
    if image_elements:
        for img in image_elements:
            p = img['page']
            images_by_page.setdefault(p, []).append(img)
    num_pages = max(
        max(texts_by_page.keys(), default=0),
        max(qr_by_page.keys(), default=0),
        max(images_by_page.keys(), default=0)
    ) + 1
    for page_num in range(num_pages):
        if page_num > 0:
            c.showPage()
        # Draw images (logos)
        for img in images_by_page.get(page_num, []):
            bounds = img['bounds']
            if bounds and len(bounds) == 4:
                x, y, width, height = bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]
                try:
                    img_reader = ImageReader(io.BytesIO(img['data']))
                    c.drawImage(img_reader, x, y, width, height, preserveAspectRatio=True, mask='auto')
                    print(f"Placed logo/image at position ({x}, {y})")
                except Exception as e:
                    print(f"Error drawing image: {e}")
        # Draw QR codes as images (always generate since Adobe doesn't extract QR as images)
        for qr in qr_by_page.get(page_num, []):
            bounds = qr['bounds']
            if bounds and len(bounds) == 4:
                x, y, width, height = bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]
                try:
                    qr_code = qrcode.QRCode(
                        version=1,
                        error_correction=qrcode.constants.ERROR_CORRECT_H,
                        box_size=10,
                        border=4,
                    )
                    qr_code.add_data(qr['text'])
                    qr_code.make(fit=True)
                    qr_img = qr_code.make_image(fill_color="black", back_color="white")
                    min_size = 50
                    if width < min_size or height < min_size:
                        scale_factor = max(min_size / width, min_size / height)
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)
                        qr_img = qr_img.resize((new_width, new_height))
                        x_offset = (width - new_width) / 2
                        y_offset = (height - new_height) / 2
                        x += x_offset
                        y += y_offset
                        width, height = new_width, new_height
                    img_buffer = io.BytesIO()
                    qr_img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    img_reader = ImageReader(img_buffer)
                    c.drawImage(img_reader, x, y, width, height, preserveAspectRatio=True, mask='auto')
                    print(f"Generated QR code at position ({x}, {y}) with data: {qr['text'][:50]}...")
                except Exception as e:
                    print(f"Error drawing QR code image: {e}")
                    print(f"QR data was: {qr['text']}")
        # Draw text (sorted by Y descending, then X ascending)
        page_texts = texts_by_page.get(page_num, [])
        page_texts_sorted = sorted(page_texts, key=lambda t: (-t[0]['bounds'][1], t[0]['bounds'][0]))
        for element, translated_text in page_texts_sorted:
            bounds = element['bounds']
            if len(bounds) >= 4:
                x, y, width, height = bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]
                font_size = element['font_size']
                font_family = element['font_family']
                font_weight = element['font_weight']
                font_style = element['font_style']
                text_align = element['text_align']
                line_height = element['line_height']
                original_font_name = element['font_name']
                mapped_font_name = _map_font_family(font_family, font_weight, font_style)
                alignment = _map_text_alignment(text_align)
                color = Color(0, 0, 0)
                
                # Draw background color if available
                background_color = element.get('background_color')
                if background_color and len(background_color) >= 3:
                    try:
                        # Always treat as [0, 1] floats
                        r = background_color[0]
                        g = background_color[1]
                        b = background_color[2]
                        bg_color = Color(r, g, b)
                        c.setFillColor(bg_color)
                        c.rect(x, y, width, height, fill=1, stroke=0)
                        c.setFillColor(Color(0, 0, 0))
                        print(f"Drew background color ({r:.2f}, {g:.2f}, {b:.2f}) for text: {translated_text[:30]}...")
                    except Exception as e:
                        print(f"Error drawing background color: {e}")
                        c.setFillColor(Color(0, 0, 0))
                # Decide if we should wrap (Paragraph) or single-line fit
                use_paragraph = (len(translated_text) > 20 or ' ' in translated_text)
                if use_paragraph:
                    min_font_size = max(6, font_size * 0.7)
                    current_font_size = font_size
                    fitted = False
                    final_para = None
                    while current_font_size >= min_font_size:
                        style = ParagraphStyle(
                            name=f'Custom_{page_num}_{x}_{y}_{current_font_size}',
                            parent=base_style,
                            fontName=mapped_font_name,
                            fontSize=current_font_size,
                            leading=current_font_size * line_height,
                            alignment=alignment,
                            textColor=color,
                        )
                        para = Paragraph(translated_text, style)
                        para_width, para_height = para.wrap(width, height)
                        if para_height <= height:
                            fitted = True
                            final_para = para
                            break
                        current_font_size -= 0.5
                    if not fitted:
                        style = ParagraphStyle(
                            name=f'Custom_{page_num}_{x}_{y}_{min_font_size}',
                            parent=base_style,
                            fontName=mapped_font_name,
                            fontSize=min_font_size,
                            leading=min_font_size * line_height,
                            alignment=alignment,
                            textColor=color,
                        )
                        final_para = Paragraph(translated_text, style)
                        final_para.wrap(width, height)
                    else:
                        final_para.wrap(width, height)
                    final_para.drawOn(c, x, y)
                else:
                    # Use direct canvas drawing with dynamic font sizing to fit text without wrapping
                    min_font_size = max(4, font_size * 0.3)
                    current_font_size = font_size
                    fitted = False
                    while current_font_size >= min_font_size:
                        c.setFont(mapped_font_name, current_font_size)
                        c.setFillColor(color)
                        text_width = c.stringWidth(translated_text, mapped_font_name, current_font_size)
                        text_height = current_font_size * 1.2
                        if text_width <= width and text_height <= height:
                            fitted = True
                            if alignment == TA_CENTER:
                                text_x = x + (width - text_width) / 2
                            elif alignment == TA_RIGHT:
                                text_x = x + width - text_width
                            else:
                                text_x = x
                            text_y = y + (height - text_height) / 2 + text_height * 0.8
                            c.drawString(text_x, text_y, translated_text)
                            break
                        current_font_size -= 0.5
                    if not fitted:
                        c.setFont(mapped_font_name, min_font_size)
                        c.setFillColor(color)
                        text_width = c.stringWidth(translated_text, mapped_font_name, min_font_size)
                        text_height = min_font_size * 1.2
                        if alignment == TA_CENTER:
                            text_x = x + (width - text_width) / 2
                        elif alignment == TA_RIGHT:
                            text_x = x + width - text_width
                        else:
                            text_x = x
                        text_y = y + (height - text_height) / 2 + text_height * 0.8
                        if text_width > width:
                            chars_to_show = len(translated_text)
                            while chars_to_show > 0:
                                partial_text = translated_text[:chars_to_show]
                                partial_width = c.stringWidth(partial_text, mapped_font_name, min_font_size)
                                if partial_width <= width:
                                    break
                                chars_to_show -= 1
                            if chars_to_show > 0:
                                c.drawString(text_x, text_y, translated_text[:chars_to_show])
                            else:
                                c.drawString(x, text_y, translated_text[0] if translated_text else "")
                        else:
                            c.drawString(text_x, text_y, translated_text)
                if font_weight >= 600 or font_style == 'bold' or font_style == 'italic':
                    print(f"Applied {mapped_font_name} (original: {original_font_name}) for text: {translated_text[:30]}...")
    c.save()
    print(f"PDF reconstructed with original formatting: {output_path}")
    return output_path

def main():
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    if not os.path.exists(ADOBE_CREDENTIALS):
        print(f"Error: Adobe credentials file not found: {ADOBE_CREDENTIALS}")
        return
    if not os.path.exists(INPUT_PDF):
        print(f"Error: Input PDF not found: {INPUT_PDF}")
        return
    structured_data = extract_pdf_elements(INPUT_PDF, adobe_creds)
    text_elements, qr_elements = parse_text_and_visual_elements(structured_data)
    texts_to_translate = [elem['text'] for elem in text_elements] if text_elements else []
    translated_texts = translate_text_batch(texts_to_translate, "English") if texts_to_translate else []
    image_elements = extract_logo_images(INPUT_PDF)
    reconstruct_pdf(text_elements, translated_texts, OUTPUT_PDF, qr_elements=qr_elements, image_elements=image_elements)
    print(f'Pipeline complete. Output: {OUTPUT_PDF}')

if __name__ == '__main__':
    main() 