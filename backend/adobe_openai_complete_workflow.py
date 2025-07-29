#!/usr/bin/env python3
"""
Complete PDF Translation Workflow using Adobe PDF Services SDK + OpenAI
Extracts PDF elements, translates text, and reconstructs PDF with translated content
"""

import json
import os
import zipfile
import tempfile
import time
from typing import List, Dict, Any, Tuple
import openai
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import black, white
import requests
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
import fitz  # PyMuPDF
from PIL import Image
import shutil
import qrcode
from reportlab.lib import colors
import sys

# Adobe PDF Services SDK imports
from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob, ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import ExtractRenditionsElementType

class PDFTranslationWorkflow:
    def __init__(self, adobe_credentials_path: str, openai_api_key: str):
        """Initialize the workflow with credentials"""
        self.adobe_credentials_path = adobe_credentials_path
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Load Adobe credentials
        with open(adobe_credentials_path, 'r') as f:
            self.adobe_creds = json.load(f)
    
    def extract_pdf_elements(self, input_pdf_path: str) -> Dict[str, Any]:
        """Extract PDF elements using Adobe PDF Services SDK"""
        print(f"Extracting elements from: {input_pdf_path}")
        
        # Initialize credentials
        credentials = ServicePrincipalCredentials(
            client_id=self.adobe_creds['client_credentials']['client_id'],
            client_secret=self.adobe_creds['client_credentials']['client_secret']
        )
        
        # Create PDF Services instance
        pdf_services = PDFServices(credentials=credentials)
        
        # Upload the PDF as an Asset
        with open(input_pdf_path, "rb") as f:
            input_stream = f.read()
        
        print("⏳ Uploading PDF to Adobe PDF Services...")
        input_asset = pdf_services.upload(input_stream=input_stream, mime_type="application/pdf")
        
        # Set extraction parameters
        extract_params = ExtractPDFParams(
            elements_to_extract=[
                ExtractElementType.TEXT,
                ExtractElementType.TABLES
            ],
            elements_to_extract_renditions=[
                ExtractRenditionsElementType.FIGURES,
                ExtractRenditionsElementType.TABLES
            ]
        )
        
        # Create and submit the extract job
        job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_params)
        print("⏳ Submitting extract job...")
        polling_url = pdf_services.submit(job)
        
        # Poll for result
        print("⏳ Waiting for job to complete...")
        response = pdf_services.get_job_result(polling_url, ExtractPDFResult)
        
        # Save the result ZIP
        result_asset = response.get_result().get_resource()
        stream_asset = pdf_services.get_content(result_asset)
        
        output_zip_path = input_pdf_path.replace('.pdf', '_extracted.zip')
        with open(output_zip_path, "wb") as out_f:
            out_f.write(stream_asset.get_input_stream())
        
        # Extract and parse JSON
        with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
            with open('structuredData.json', 'r', encoding='utf-8') as f:
                structured_data = json.load(f)
        
        print(f"Extraction complete. Found {len(structured_data.get('elements', []))} elements")
        return structured_data
    
    def parse_text_elements(self, structured_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse text elements from structured data and sort by visual reading order"""
        text_elements = []
        qr_elements = []  # Store QR codes separately as visual elements
        
        for element in structured_data.get('elements', []):
            if 'Text' in element and element['Text'].strip():
                text = element['Text'].strip()
                
                # Check if this is a QR code or barcode (should be treated as visual element)
                if self._is_qr_or_barcode(text):
                    print(f"[QR/Barcode] Detected visual element: {text[:50]}... (preserving as visual)")
                    # Store QR code as a visual element to be rendered
                    qr_element = {
                        'text': text,
                        'bounds': element.get('Bounds', []),
                        'page': element.get('Page', 0),
                        'font_size': element.get('TextSize', 12),
                        'font_family': element.get('Font', {}).get('family_name', 'Courier'),
                        'font_weight': element.get('Font', {}).get('weight', 400),
                        'font_style': element.get('Font', {}).get('style', 'normal'),
                        'font_color': element.get('Color', [0, 0, 0]),
                        'object_id': element.get('ObjectID', 0),
                        'path': element.get('Path', ''),
                        'lang': element.get('Lang', 'unknown'),
                        'text_align': element.get('TextAlign', 'left'),
                        'line_height': element.get('LineHeight', 1.2),
                        'is_qr_code': True  # Mark as QR code
                    }
                    qr_elements.append(qr_element)
                    continue  # Skip QR codes from translation
                
                # Extract font information
                font_info = element.get('Font', {})
                text_element = {
                    'text': text,
                    'bounds': element.get('Bounds', []),
                    'page': element.get('Page', 0),
                    'font_size': element.get('TextSize', 12),
                    'font_family': font_info.get('family_name', 'Arial'),
                    'font_weight': font_info.get('weight', 400),
                    'font_style': font_info.get('style', 'normal'),  # normal, italic, etc.
                    'font_color': element.get('Color', [0, 0, 0]),  # RGB color
                    'object_id': element.get('ObjectID', 0),
                    'path': element.get('Path', ''),
                    'lang': element.get('Lang', 'unknown'),
                    'text_align': element.get('TextAlign', 'left'),  # left, center, right, justify
                    'line_height': element.get('LineHeight', 1.2),  # line spacing multiplier
                }
                text_elements.append(text_element)
        
        # Sort by visual reading order: page, then by typical document layout
        def sort_key(elem):
            b = elem['bounds']
            text = elem['text'].lower()
            
            if len(b) >= 4:
                # Round Y to nearest 5 points to group similar Y positions
                y_group = round(b[1] / 5) * 5
                
                # Determine reading order priority based on content and position
                priority = 0
                
                # Address information (typically top-left, read first) - be more specific
                if any(word in text for word in ['straat', 'amsterdam', 'postcode']) or ('heer' in text and 'mevrouw' in text and len(text) < 50 and 'geachte' not in text.lower()):
                    priority = 1
                # Bank header (typically top-right, read second)
                elif any(word in text for word in ['abn amro', 'bank', 'abnamro.nl']):
                    priority = 2
                # Table headers and labels (read third)
                elif any(word in text for word in ['behandeld', 'muntsoort', 'afdeling', 'leningnummer', 'datum']):
                    priority = 3
                # Main content (read last) - including greetings
                else:
                    priority = 4
                
                return (elem['page'], priority, -y_group, b[0])
            return (elem['page'], 0, 0, 0)
        
        text_elements.sort(key=sort_key)
        print(f"Parsed {len(text_elements)} text elements (sorted by visual reading order)")
        print(f"Found {len(qr_elements)} QR code elements to preserve as visual")
        
        # Store QR elements for later rendering
        self.qr_elements = qr_elements
        
        return text_elements
    
    def _is_qr_or_barcode(self, text: str) -> bool:
        """Detect if text is a QR code or barcode that should be treated as a visual element"""
        # QR codes are often rendered as Unicode block characters
        block_chars = ['\u2588', '\u2580', '\u2584', '\u258C', '\u2590', '\u2591', '\u2592', '\u2593']
        
        # Check if text contains mostly block characters (QR code pattern)
        if len(text) > 10:  # QR codes are typically longer
            block_char_count = sum(1 for char in text if char in block_chars)
            if block_char_count / len(text) > 0.7:  # More than 70% block characters
                return True
        
        # Check for barcode patterns (repetitive characters)
        if len(text) > 20:
            # Look for repetitive patterns typical in barcodes
            char_counts = {}
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # If a few characters dominate the text, it might be a barcode
            total_chars = len(text)
            max_char_count = max(char_counts.values())
            if max_char_count / total_chars > 0.5:  # One character appears more than 50% of the time
                return True
        
        # Check for specific QR code indicators
        qr_indicators = ['qr', 'code', 'barcode', 'matrix']
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in qr_indicators):
            return True
        
        return False
    
    def parse_image_elements(self, structured_data: Dict[str, Any], extraction_dir: str = '.') -> List[Dict[str, Any]]:
        """Parse image elements from structured data and renditions, and ensure images are in the per-PDF directory."""
        image_elements = []
        for element in structured_data.get('elements', []):
            file_paths = element.get('filePaths')
            # Use Bounds if present, else fallback to attributes.BBox
            bounds = element.get('Bounds') or (element.get('attributes', {}).get('BBox'))
            page = element.get('Page', 0)
            if file_paths and bounds:
                for file_path in file_paths:
                    src_path = file_path
                    if src_path.startswith('figures/'):
                        filename = os.path.basename(src_path)
                        dest_path = os.path.join(extraction_dir, filename)
                        if not os.path.exists(dest_path):
                            try:
                                shutil.copy2(src_path, dest_path)
                                print(f"[AdobeImage] Copied {src_path} -> {dest_path}")
                            except Exception as e:
                                print(f"[AdobeImage] Failed to copy {src_path} -> {dest_path}: {e}")
                        else:
                            print(f"[AdobeImage] Exists in {dest_path}")
                        image_element = {
                            'bounds': bounds,
                            'page': page,
                            'path': dest_path,
                            'object_id': element.get('ObjectID', 0),
                            'filename': filename
                        }
                        image_elements.append(image_element)
                        print(f"[AdobeImage] Registered: {filename} page={page} bounds={bounds}")
        # Also check for any Image elements in the structured data (legacy)
        for element in structured_data.get('elements', []):
            if 'Image' in element:
                image_info = element['Image']
                bounds = element.get('Bounds', [])
                page = element.get('Page', 0)
                image_path = None
                if 'path' in image_info:
                    potential_paths = [
                        os.path.join(extraction_dir, image_info['path']),
                        os.path.join(extraction_dir, 'images', image_info['path']),
                        os.path.join(extraction_dir, 'images', os.path.basename(image_info['path']))
                    ]
                    for path in potential_paths:
                        if os.path.exists(path):
                            image_path = path
                            break
                if image_path:
                    image_element = {
                        'bounds': bounds,
                        'page': page,
                        'path': image_path,
                        'object_id': element.get('ObjectID', 0),
                        'image_info': image_info
                    }
                    image_elements.append(image_element)
        print(f"Parsed {len(image_elements)} image elements (Adobe + legacy)")
        return image_elements
    
    def translate_text_batch(self, texts: List[str], target_language: str = "English") -> List[str]:
        """Translate a batch of texts using OpenAI"""
        print(f"Translating {len(texts)} text blocks to {target_language}")
        
        # Combine texts for batch translation
        combined_text = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
        
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional translator. Translate the following text blocks to {target_language}. Maintain the same numbering format and return only the translated text with numbers."
                    },
                    {
                        "role": "user",
                        "content": combined_text
                    }
                ],
                max_tokens=4000,
                temperature=0.3
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # Parse the numbered responses
            translated_lines = translated_text.split('\n')
            translations = []
            
            for line in translated_lines:
                if line.strip() and '. ' in line:
                    # Extract text after the number
                    text_part = line.split('. ', 1)[1] if '. ' in line else line
                    translations.append(text_part.strip())
                else:
                    translations.append(line.strip())
            
            # Ensure we have the same number of translations as inputs
            while len(translations) < len(texts):
                translations.append("")
            
            return translations[:len(texts)]
            
        except Exception as e:
            print(f"Translation error: {e}")
            # Return original texts if translation fails
            return texts
    
    def parse_tables(self, structured_data: Dict[str, Any], translated_texts: Dict[int, str]) -> list:
        tables = []
        # Find all table elements
        for element in structured_data.get('elements', []):
            if element.get('Path', '').endswith('/Table'):
                table_info = {
                    'object_id': element.get('ObjectID'),
                    'page': element.get('Page', 0),
                    'bbox': element.get('attributes', {}).get('BBox', []),
                    'num_rows': element.get('attributes', {}).get('NumRow', 0),
                    'num_cols': element.get('attributes', {}).get('NumCol', 0),
                    'cells': []
                }
                tables.append(table_info)
        # Build a lookup for /P children by parent path
        path_to_p = {}
        for element in structured_data.get('elements', []):
            path = element.get('Path', '')
            if path.endswith('/P'):
                parent_path = path.rsplit('/P', 1)[0]
                path_to_p[parent_path] = element
        # Find all table cell elements (TH/TD)
        for element in structured_data.get('elements', []):
            path = element.get('Path', '')
            if '/Table/TR/' in path and (path.endswith('/TD') or path.endswith('/TH')):
                attrs = element.get('attributes', {})
                row = attrs.get('RowIndex')
                col = attrs.get('ColIndex')
                obj_id = element.get('ObjectID')
                # Prefer /P child for text and formatting
                p_elem = path_to_p.get(path)
                text = ''
                if p_elem:
                    text = p_elem.get('Text', '').strip()
                    if p_elem.get('ObjectID') in translated_texts:
                        text = translated_texts[p_elem.get('ObjectID')]
                else:
                    text = element.get('Text', '').strip()
                    if obj_id in translated_texts:
                        text = translated_texts[obj_id]
                cell = {
                    'row': row,
                    'col': col,
                    'text': text,
                    'bbox': attrs.get('BBox', []),
                    'is_header': 'TH' in path,
                    'object_id': p_elem.get('ObjectID') if p_elem else obj_id
                }
                # Find the parent table by bbox containment
                for table in tables:
                    tbbox = table['bbox']
                    cbbox = cell['bbox']
                    if tbbox and cbbox and tbbox[0] <= cbbox[0] <= tbbox[2] and tbbox[1] <= cbbox[1] <= tbbox[3]:
                        table['cells'].append(cell)
                        break
        return tables

    def reconstruct_pdf(self, text_elements: List[Dict[str, Any]], 
                       translated_texts: List[str], 
                       output_path: str,
                       page_size: Tuple[float, float] = A4,
                       image_elements: List[Dict[str, Any]] = None,
                       structured_data: Dict[str, Any] = None) -> str:
        """Reconstruct PDF with translated text and images, using best available bounds for each image."""
        print(f"Reconstructing PDF with translated text and original formatting...")
        
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.colors import Color
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.utils import ImageReader
        import os
        
        # Use A4 if not specified
        if page_size is None:
            page_size = A4
        
        # Register fonts
        self.register_fonts()
        
        # Create PDF
        canvas_obj = canvas.Canvas(output_path, pagesize=page_size)
        page_width, page_height = page_size
        styles = getSampleStyleSheet()
        base_style = styles["Normal"]

        # Group images and text by page
        images_by_page = {}
        if image_elements:
            for img in image_elements:
                p = img.get('page', 0)
                images_by_page.setdefault(p, []).append(img)
        texts_by_page = {}
        for i, element in enumerate(text_elements):
            p = element['page']
            texts_by_page.setdefault(p, []).append((element, translated_texts[i] if i < len(translated_texts) else element['text']))
        
        # Group QR codes by page
        qr_codes_by_page = {}
        if hasattr(self, 'qr_elements'):
            for qr_element in self.qr_elements:
                p = qr_element['page']
                qr_codes_by_page.setdefault(p, []).append(qr_element)
        
        # Build a mapping from ObjectID to translated text for table cell lookup
        objid_to_trans = {e['object_id']: t for e, t in zip(text_elements, translated_texts) if 'object_id' in e}
        # Parse tables
        tables = self.parse_tables(structured_data, objid_to_trans) if structured_data else []
        tables_by_page = {}
        for table in tables:
            p = table['page']
            tables_by_page.setdefault(p, []).append(table)
        # Track cell object IDs to skip from text rendering
        table_cell_objids = set()
        for table in tables:
            for cell in table['cells']:
                table_cell_objids.add(cell.get('object_id'))
        
        num_pages = max(
            max(images_by_page.keys(), default=-1),
            max(texts_by_page.keys(), default=-1),
            max(qr_codes_by_page.keys(), default=-1),
            max(tables_by_page.keys(), default=-1)
        ) + 1
        
        for page_num in range(num_pages):
            if page_num > 0:
                canvas_obj.showPage()
            
            # Draw all images for this page
            for img_element in images_by_page.get(page_num, []):
                img_path = img_element.get('path', '')
                bounds_pymupdf = img_element.get('bounds', None)
                bounds_adobe = img_element.get('adobe_bounds', None)
                use_bbox = False
                x = y = width = height = None
                # Check if bbox is valid (not full page)
                if bounds_pymupdf and len(bounds_pymupdf) == 4:
                    bx, by, bw, bh = bounds_pymupdf
                    if not (abs(bx) < 2 and abs(by) < 2 and abs(bw - page_width) < 2 and abs(bh - page_height) < 2):
                        x, y, width, height = bx, by, bw - bx, bh - by
                        use_bbox = True
                        print(f"  Placed image (PyMuPDF bbox): {os.path.basename(img_path)} at ({x:.1f}, {y:.1f}) size ({width:.1f}x{height:.1f})")
                if not use_bbox and bounds_adobe and len(bounds_adobe) == 4:
                    bx, by, bw, bh = bounds_adobe
                    if not (abs(bx) < 2 and abs(by) < 2 and abs(bw - page_width) < 2 and abs(bh - page_height) < 2):
                        x, y, width, height = bx, by, bw - bx, bh - by
                        use_bbox = True
                        print(f"  Placed image (Adobe bbox): {os.path.basename(img_path)} at ({x:.1f}, {y:.1f}) size ({width:.1f}x{height:.1f})")
                if use_bbox:
                    try:
                        canvas_obj.drawImage(img_path, x, y, width, height)
                    except Exception as e:
                        print(f"  Error placing image {img_path}: {e}")
                else:
                    print(f"  WARNING: No valid bbox for image {os.path.basename(img_path)}. Skipping placement.")
            
            # Draw all QR codes for this page (render as real QR image)
            for qr_element in qr_codes_by_page.get(page_num, []):
                bounds = qr_element['bounds']
                if len(bounds) >= 4:
                    x, y, width, height = bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]
                    qr_text = qr_element['text']
                    # Try to use the text as payload if not just block chars
                    block_chars = set(['', '\u2588', '\u2580', '\u2584', '\u258C', '\u2590', '\u2591', '\u2592', '\u2593'])
                    payload = qr_text
                    # If mostly block chars, use a placeholder
                    if sum(1 for c in qr_text if c in block_chars) / max(1, len(qr_text)) > 0.7:
                        payload = 'QR_CODE'
                    # Generate QR code image
                    qr_img = qrcode.make(payload)
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
                        qr_img.save(tmp_img.name)
                        tmp_img_path = tmp_img.name
                    try:
                        canvas_obj.drawImage(tmp_img_path, x, y, width, height, preserveAspectRatio=True, mask='auto')
                        print(f"  Rendered QR code IMAGE at ({x:.1f}, {y:.1f}) size ({width:.1f}x{height:.1f}) payload: {payload}")
                    except Exception as e:
                        print(f"  Error rendering QR code image: {e}")
            
            # Draw all tables for this page
            for table in tables_by_page.get(page_num, []):
                if not table['bbox'] or not table['cells']:
                    continue
                x0, y0, x1, y1 = table['bbox']
                width, height = x1 - x0, y1 - y0
                nrows, ncols = table['num_rows'], table['num_cols']
                # Build 2D data array with Paragraphs
                data = [["" for _ in range(ncols)] for _ in range(nrows)]
                for cell in table['cells']:
                    r, c = cell['row'], cell['col']
                    if r is not None and c is not None:
                        # Try to get font info from text_elements
                        font_size = 9
                        font_name = "Helvetica"
                        alignment = 1  # Center
                        for e in text_elements:
                            if e.get('object_id') == cell.get('object_id'):
                                font_size = e.get('font_size', 9)
                                font_name = self._map_font_family(e.get('font_family', 'Helvetica'), e.get('font_weight', 400), e.get('font_style', 'normal'))
                                alignment = self._map_text_alignment(e.get('text_align', 'center'))
                                break
                        style = ParagraphStyle(
                            name=f'TableCell_{r}_{c}',
                            parent=base_style,
                            fontName=font_name,
                            fontSize=font_size,
                            alignment=alignment,
                            leading=font_size * 1.2,
                        )
                        para = Paragraph(cell['text'], style)
                        data[r][c] = para
                # Create and style the table
                tbl = Table(data, colWidths=width/ncols, rowHeights=height/nrows)
                tbl.setStyle(TableStyle([
                    ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ]))
                tbl.wrapOn(canvas_obj, width, height)
                tbl.drawOn(canvas_obj, x0, y0)
                print(f"  Rendered table at ({x0:.1f}, {y0:.1f}) size ({width:.1f}x{height:.1f})")
            
            # Draw all text for this page
            for element, translated_text in texts_by_page.get(page_num, []):
                # Skip if this text is part of a table cell
                if element.get('object_id') in table_cell_objids:
                    continue
                bounds = element['bounds']
                if len(bounds) >= 4:
                    x, y, width, height = bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]
                    font_size = element['font_size']
                    font_family = element['font_family']
                    font_weight = element['font_weight']
                    font_style = element['font_style']
                    text_align = element['text_align']
                    line_height = element['line_height']
                    font_name = self._map_font_family(font_family, font_weight, font_style)
                    if font_weight >= 600:
                        print(f"  Bold text detected: '{translated_text[:30]}...' -> {font_name}")
                    alignment = self._map_text_alignment(text_align)
                    color = Color(0, 0, 0)
                    # Try to fit text in the original box, reducing font size if needed
                    min_font_size = max(6, font_size * 0.7)  # Don't go below 70% of original or 6pt
                    current_font_size = font_size
                    fitted = False
                    final_para = None
                    while current_font_size >= min_font_size:
                        style = ParagraphStyle(
                            name=f'Custom_{page_num}_{x}_{y}_{current_font_size}',
                            parent=base_style,
                            fontName=font_name,
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
                        current_font_size -= 0.5  # Reduce font size in small steps
                    if not fitted:
                        # If it still doesn't fit, use the minimum font size
                        style = ParagraphStyle(
                            name=f'Custom_{page_num}_{x}_{y}_{min_font_size}',
                            parent=base_style,
                            fontName=font_name,
                            fontSize=min_font_size,
                            leading=min_font_size * line_height,
                            alignment=alignment,
                            textColor=color,
                        )
                        final_para = Paragraph(translated_text, style)
                        final_para.wrap(width, height)
                    else:
                        # Ensure wrap is called on the final_para before drawing
                        final_para.wrap(width, height)
                    final_para.drawOn(canvas_obj, x, y)
        canvas_obj.save()
        print(f"PDF reconstructed with original formatting (black text only): {output_path}")
        return output_path
    
    def _map_font_family(self, font_family: str, font_weight: int, font_style: str) -> str:
        """Map font family, weight, and style to ReportLab font names"""
        # Common font mappings
        font_mappings = {
            'Arial': 'Helvetica',
            'Helvetica': 'Helvetica',
            'Times': 'Times-Roman',
            'Times New Roman': 'Times-Roman',
            'Courier': 'Courier',
            'Courier New': 'Courier',
            'Verdana': 'Helvetica',  # Fallback to Helvetica
            'Georgia': 'Times-Roman',  # Fallback to Times
            'Tahoma': 'Helvetica',  # Fallback to Helvetica
            # Custom fonts mapping
            'ABNAMRO Sans': 'Helvetica',  # Custom sans-serif font
            'ABNAMRO': 'Helvetica',  # Custom sans-serif font
        }
        
        # Get base font name
        base_font = font_mappings.get(font_family, 'Helvetica')
        
        # Apply weight and style - check for semi-bold (600) and bold (700+)
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
    
    def _map_text_alignment(self, text_align: str) -> int:
        """Map text alignment to ReportLab alignment constants"""
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
        
        alignment_map = {
            'left': TA_LEFT,
            'center': TA_CENTER,
            'right': TA_RIGHT,
            'justify': TA_JUSTIFY,
        }
        return alignment_map.get(text_align.lower(), TA_LEFT)
    
    def _create_color(self, color_rgb: List[float]) -> str:
        """Create ReportLab color from RGB values"""
        from reportlab.lib.colors import Color
        
        if len(color_rgb) >= 3:
            r, g, b = color_rgb[0], color_rgb[1], color_rgb[2]
            # Normalize RGB values (0-1 range)
            if r > 1 or g > 1 or b > 1:
                r, g, b = r/255, g/255, b/255
            return Color(r, g, b)
        return Color(0, 0, 0)  # Default to black
    
    def extract_color_information(self, input_pdf_path: str, text_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract color information from the original PDF and match it with text elements"""
        print("Extracting color information from original PDF...")
        
        try:
            # Open the original PDF
            doc = fitz.open(input_pdf_path)
            
            # Dictionary to store color information by page and position
            color_info = {}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text blocks with color information
                text_dict = page.get_text("dict")
                
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                # Get color information
                                color = span.get("color", 0)  # Default to black
                                bbox = span["bbox"]  # (x0, y0, x1, y1)
                                
                                # Store color info by position
                                key = (page_num, bbox[0], bbox[1], bbox[2], bbox[3])
                                color_info[key] = color
            
            # Match color information with text elements
            for element in text_elements:
                bounds = element['bounds']
                page = element['page']
                
                if len(bounds) >= 4:
                    # Find matching color by position
                    element_key = (page, bounds[0], bounds[1], bounds[2], bounds[3])
                    
                    # Look for exact match or closest match
                    matched_color = None
                    min_distance = float('inf')
                    
                    for color_key, color in color_info.items():
                        if color_key[0] == page:  # Same page
                            # Calculate distance between bounding boxes
                            distance = abs(color_key[1] - bounds[0]) + abs(color_key[2] - bounds[1])
                            if distance < min_distance:
                                min_distance = distance
                                matched_color = color
                    
                    # Convert color to RGB
                    if matched_color is not None:
                        # Handle different color formats
                        if isinstance(matched_color, (list, tuple)):
                            # RGB values already provided
                            if len(matched_color) >= 3:
                                element['font_color'] = list(matched_color[:3])
                            else:
                                element['font_color'] = [0, 0, 0]
                        elif isinstance(matched_color, (int, float)):
                            # Integer RGB value (e.g., 16711680 for red)
                            if matched_color > 0:
                                # Convert integer RGB to normalized RGB
                                r = (matched_color >> 16) & 0xFF
                                g = (matched_color >> 8) & 0xFF
                                b = matched_color & 0xFF
                                element['font_color'] = [r/255, g/255, b/255]
                            else:
                                element['font_color'] = [0, 0, 0]  # Black
                        else:
                            element['font_color'] = [0, 0, 0]
                    else:
                        element['font_color'] = [0, 0, 0]  # Default to black
            
            doc.close()
            print(f"Color information extracted and matched for {len(text_elements)} elements")
            return text_elements
            
        except ImportError:
            print("PyMuPDF not available, using default colors")
            for element in text_elements:
                element['font_color'] = [0, 0, 0]  # Default to black
            return text_elements
        except Exception as e:
            print(f"Error extracting color information: {e}")
            for element in text_elements:
                element['font_color'] = [0, 0, 0]  # Default to black
            return text_elements
    
    def register_fonts(self):
        """Register custom fonts for ReportLab"""
        try:
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            
            # Register Noto fonts if available
            font_paths = {
                'NotoSans-Regular': 'fonts/NotoSans-Regular.ttf',
                'NotoSans-Bold': 'fonts/NotoSans-Bold.ttf',
                'NotoSansDevanagari-Regular': 'fonts/NotoSansDevanagari-Regular.ttf',
                'NotoSansDevanagari-Bold': 'fonts/NotoSansDevanagari-Bold.ttf',
                'NotoSansArabic-Regular': 'fonts/NotoSansArabic-Regular.ttf',
                'NotoSansArabic-Bold': 'fonts/NotoSansArabic-Bold.ttf'
            }
            
            for font_name, font_path in font_paths.items():
                try:
                    if os.path.exists(font_path):
                        pdfmetrics.registerFont(TTFont(font_name, font_path))
                        print(f"Registered font: {font_name}")
                except Exception as e:
                    print(f"Failed to register font {font_name}: {e}")
                    
        except ImportError:
            print("ReportLab not available for font registration")
        except Exception as e:
            print(f"Error registering fonts: {e}")
    
    def run_complete_workflow(self, input_pdf_path: str, output_pdf_path: str, 
                            target_language: str = "English") -> str:
        """Run the complete PDF translation workflow"""
        print("=== Starting Complete PDF Translation Workflow ===")
        
        # Step 1: Extract PDF elements
        structured_data = self.extract_pdf_elements(input_pdf_path)
        
        # Step 2: Parse text elements
        text_elements = self.parse_text_elements(structured_data)
        
        # --- Use unique image extraction directory per PDF ---
        pdf_stem = os.path.splitext(os.path.basename(input_pdf_path))[0]
        extraction_dir = os.path.join("figures", pdf_stem)
        # Clean the directory before use
        if os.path.exists(extraction_dir):
            shutil.rmtree(extraction_dir)
        os.makedirs(extraction_dir, exist_ok=True)
        
        # Step 3: Parse image elements from Adobe extraction (use extraction_dir)
        adobe_image_elements = self.parse_image_elements(structured_data, extraction_dir=extraction_dir)
        
        # Step 4: Extract all images from PDF using PyMuPDF (for logos, etc) into extraction_dir
        pymupdf_image_elements = self.extract_all_images_pymupdf(input_pdf_path, output_dir=extraction_dir)
        
        # Step 5: Merge image elements, deduplicating by file path and coordinates
        def image_key(img):
            return (os.path.basename(img.get('path', '')), img.get('page', -1))
        seen = {}
        # Index Adobe images by key for bounds fallback
        adobe_index = {image_key(img): img for img in adobe_image_elements}
        merged_images = []
        for img in pymupdf_image_elements:
            k = image_key(img)
            if k not in seen and img.get('path') and os.path.exists(img['path']):
                seen[k] = True
                # Attach Adobe bounds if available
                if k in adobe_index and 'bounds' in adobe_index[k]:
                    img['adobe_bounds'] = adobe_index[k]['bounds']
                merged_images.append(img)
        # Add any Adobe images not already included
        for img in adobe_image_elements:
            k = image_key(img)
            if k not in seen and img.get('path') and os.path.exists(img['path']):
                seen[k] = True
                merged_images.append(img)
        print(f"Total unique images for placement: {len(merged_images)}")
        
        if not text_elements and not merged_images:
            print("No text or image elements found")
            return input_pdf_path
        
        # Step 6: Extract color information from original PDF
        if text_elements:
            text_elements = self.extract_color_information(input_pdf_path, text_elements)
        
        # Step 7: Extract texts for translation
        texts_to_translate = [elem['text'] for elem in text_elements] if text_elements else []
        
        # Step 8: Translate texts
        translated_texts = self.translate_text_batch(texts_to_translate, target_language) if texts_to_translate else []
        
        # Step 9: Reconstruct PDF with both text and images
        final_pdf_path = self.reconstruct_pdf(text_elements, translated_texts, output_pdf_path, image_elements=merged_images, structured_data=structured_data)
        
        print("=== Workflow Complete ===")
        return final_pdf_path
    
    def extract_all_images_pymupdf(self, input_pdf_path: str, output_dir: str = "figures_pymupdf") -> list:
        """Extract all images from each page using PyMuPDF, save to output_dir, and return list of dicts with path, bounds, and page."""
        os.makedirs(output_dir, exist_ok=True)
        doc = fitz.open(input_pdf_path)
        image_elements = []
        for page_num, page in enumerate(doc):
            img_list = page.get_images(full=True)
            # Get all image blocks with bbox and xref
            text_dict = page.get_text("dict")
            image_blocks = [block for block in text_dict.get("blocks", []) if block.get("type", 0) == 1]
            for img_index, img in enumerate(img_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                img_path = os.path.join(output_dir, f"page{page_num+1}_img{img_index+1}.{ext}")
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                # Find bbox for this xref
                bbox = None
                for block in image_blocks:
                    if block.get("image", None) == xref:
                        bbox = block.get("bbox", None)
                        break
                image_elements.append({
                    "path": img_path,
                    "page": page_num,
                    "bounds": bbox,
                    "xref": xref
                })
        doc.close()
        print(f"[PyMuPDF] Extracted {len(image_elements)} images from PDF.")
        return image_elements

def main():
    """Main function to run the workflow"""
    # Configuration
    adobe_credentials_path = "pdfservices-api-credentials.json"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Parse command line arguments
    input_pdf = "sample2.pdf"
    output_pdf = "translated_sample2.pdf"
    
    if len(sys.argv) >= 2:
        input_pdf = sys.argv[1]
    if len(sys.argv) >= 3:
        output_pdf = sys.argv[2]
    
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    if not os.path.exists(adobe_credentials_path):
        print(f"Error: Adobe credentials file not found: {adobe_credentials_path}")
        return
    
    if not os.path.exists(input_pdf):
        print(f"Error: Input PDF not found: {input_pdf}")
        return
    
    # Create workflow instance
    workflow = PDFTranslationWorkflow(adobe_credentials_path, openai_api_key)
    
    # Run complete workflow
    try:
        result_path = workflow.run_complete_workflow(input_pdf, output_pdf, "English")
        print(f"Translation complete! Output saved to: {result_path}")
    except Exception as e:
        print(f"Workflow failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 