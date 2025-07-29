#!/usr/bin/env python3
"""
Adobe PDF Services Solution for Complex Scripts
Handles Hindi, Arabic, Thai, and other scripts that don't work well with ReportLab
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Adobe PDF Services imports
try:
    from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
    from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
    from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
    from adobe.pdfservices.operation.io.stream_asset import StreamAsset
    from adobe.pdfservices.operation.pdf_services import PDFServices
    from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
    from adobe.pdfservices.operation.pdfjobs.jobs.html_to_pdf_job import HTMLtoPDFJob
    from adobe.pdfservices.operation.pdfjobs.params.html_to_pdf.html_to_pdf_params import HTMLtoPDFParams
    from adobe.pdfservices.operation.pdfjobs.params.html_to_pdf.page_layout import PageLayout
    from adobe.pdfservices.operation.pdfjobs.result.html_to_pdf_result import HTMLtoPDFResult
    ADOBE_AVAILABLE = True
except ImportError:
    ADOBE_AVAILABLE = False
    print("⚠️ Adobe PDF Services SDK not available")

# ReportLab imports for fallback
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# PyMuPDF for text extraction
import fitz

# Complex script detection
COMPLEX_SCRIPTS = {
    'devanagari': ['hi', 'ne', 'mr', 'sa'],  # Hindi, Nepali, Marathi, Sanskrit
    'arabic': ['ar', 'fa', 'ur', 'ps'],      # Arabic, Persian, Urdu, Pashto
    'thai': ['th', 'lo'],                    # Thai, Lao
    'khmer': ['km'],                         # Khmer
    'myanmar': ['my'],                       # Burmese
    'hebrew': ['he', 'yi'],                  # Hebrew, Yiddish
    'georgian': ['ka'],                      # Georgian
    'ethiopic': ['am'],                      # Amharic
    'cyrillic': ['ru', 'uk', 'bg', 'mk', 'sr']  # Russian, Ukrainian, etc.
}

def detect_complex_script(text: str) -> Optional[str]:
    """
    Detect if text contains complex scripts that need special handling
    """
    if not text:
        return None
    
    # Unicode ranges for complex scripts
    script_ranges = {
        'devanagari': (0x0900, 0x097F),      # Devanagari
        'arabic': (0x0600, 0x06FF),          # Arabic
        'thai': (0x0E00, 0x0E7F),            # Thai
        'khmer': (0x1780, 0x17FF),           # Khmer
        'myanmar': (0x1000, 0x109F),         # Myanmar
        'hebrew': (0x0590, 0x05FF),          # Hebrew
        'georgian': (0x10A0, 0x10FF),        # Georgian
        'ethiopic': (0x1200, 0x137F),        # Ethiopic
        'cyrillic': (0x0400, 0x04FF),        # Cyrillic
    }
    
    for script, (start, end) in script_ranges.items():
        for char in text:
            if start <= ord(char) <= end:
                return script
    
    return None

def needs_adobe_processing(text: str) -> bool:
    """
    Determine if text needs Adobe PDF Services processing
    """
    complex_script = detect_complex_script(text)
    return complex_script is not None

def setup_adobe_credentials() -> Optional[PDFServices]:
    """
    Setup Adobe PDF Services credentials
    """
    if not ADOBE_AVAILABLE:
        print("❌ Adobe PDF Services SDK not available")
        return None
    
    try:
        client_id = os.getenv('PDF_SERVICES_CLIENT_ID')
        client_secret = os.getenv('PDF_SERVICES_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            print("❌ Adobe credentials not set")
            print("   Set PDF_SERVICES_CLIENT_ID and PDF_SERVICES_CLIENT_SECRET")
            return None
        
        credentials = ServicePrincipalCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        
        pdf_services = PDFServices(credentials=credentials)
        print("✅ Adobe PDF Services connected")
        return pdf_services
        
    except Exception as e:
        print(f"❌ Adobe setup failed: {e}")
        return None

def create_html_template_for_complex_script(pages_data: List[List[Dict]], script_type: str) -> str:
    """
    Create HTML template optimized for complex scripts
    """
    html_parts = []
    
    # HTML header with proper encoding and fonts
    html_parts.append("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Complex Script PDF</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;700&display=swap');
        
        body {
            font-family: 'Noto Sans', sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.4;
        }
        
        .page {
            width: 210mm;
            min-height: 297mm;
            margin: 0 auto;
            padding: 20mm;
            box-sizing: border-box;
            position: relative;
        }
        
        .text-element {
            position: absolute;
            white-space: pre;
        }
        
        /* Script-specific font settings */
        .devanagari { font-family: 'Noto Sans Devanagari', sans-serif; }
        .arabic { font-family: 'Noto Sans Arabic', sans-serif; direction: rtl; }
        .thai { font-family: 'Noto Sans Thai', sans-serif; }
        
        /* Font size mapping */
        .font-6 { font-size: 6pt; }
        .font-8 { font-size: 8pt; }
        .font-10 { font-size: 10pt; }
        .font-12 { font-size: 12pt; }
        .font-14 { font-size: 14pt; }
        .font-16 { font-size: 16pt; }
        .font-18 { font-size: 18pt; }
        .font-20 { font-size: 20pt; }
        .font-22 { font-size: 22pt; }
        .font-24 { font-size: 24pt; }
        
        .bold { font-weight: bold; }
        .italic { font-style: italic; }
    </style>
</head>
<body>
""")
    
    # Process each page
    for page_num, page_elements in enumerate(pages_data):
        if page_num > 0:
            html_parts.append('<div style="page-break-before: always;"></div>')
        
        html_parts.append(f'<div class="page" id="page-{page_num + 1}">')
        
        # Sort elements by vertical position (top to bottom)
        page_elements.sort(key=lambda x: -x['bbox'][1])
        
        for element in page_elements:
            text = element['text']
            bbox = element['bbox']
            font_info = element['font']
            
            # Position calculation (convert from PDF coordinates to CSS)
            x = bbox[0]
            y = 297 - bbox[1]  # Flip Y coordinate
            
            # Font size
            font_size = font_info['size']
            if font_size < 6:
                font_size = 6
            elif font_size > 24:
                font_size = 24
            
            # Font class
            font_class = f"font-{int(font_size)}"
            
            # Script class
            script_class = ""
            detected_script = detect_complex_script(text)
            if detected_script:
                script_class = f" {detected_script}"
            
            # Bold/italic
            style_class = ""
            flags = font_info.get('flags', 0)
            if flags & 2**4:  # Bold flag
                style_class += " bold"
            if flags & 2**1:  # Italic flag
                style_class += " italic"
            
            # Create text element
            html_parts.append(f'''
                <div class="text-element{script_class}{style_class} {font_class}" 
                     style="left: {x}mm; top: {y}mm;">
                    {text}
                </div>
            ''')
        
        html_parts.append('</div>')
    
    # Close HTML
    html_parts.append("""
</body>
</html>
""")
    
    return '\n'.join(html_parts)

def create_complex_script_pdf_adobe(pages_data: List[List[Dict]], output_path: str, script_type: str) -> bool:
    """
    Create PDF using Adobe PDF Services for complex scripts
    """
    if not ADOBE_AVAILABLE:
        print("❌ Adobe PDF Services not available")
        return False
    
    pdf_services = setup_adobe_credentials()
    if not pdf_services:
        print("❌ Adobe credentials not available")
        return False
    
    print(f"🏗️ Creating {script_type} PDF with Adobe PDF Services...")
    
    try:
        # Create HTML template
        html_content = create_html_template_for_complex_script(pages_data, script_type)
        
        # Create temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_html:
            temp_html.write(html_content)
            temp_html_path = temp_html.name
        
        print(f"📝 Created HTML template: {temp_html_path}")
        
        # Upload HTML to Adobe PDF Services
        with open(temp_html_path, 'rb') as file:
            input_stream = file.read()
        
        input_asset = pdf_services.upload(
            input_stream=input_stream,
            mime_type=PDFServicesMediaType.HTML
        )
        
        # Create parameters for the job
        html_to_pdf_params = HTMLtoPDFParams(
            page_layout=PageLayout.A4_PORTRAIT
        )
        
        # Create HTML to PDF job
        html_to_pdf_job = HTMLtoPDFJob(
            input_asset=input_asset,
            html_to_pdf_params=html_to_pdf_params
        )
        
        # Submit job and get result
        print("⚙️ Processing with Adobe PDF Services...")
        location = pdf_services.submit(html_to_pdf_job)
        pdf_services_response = pdf_services.get_job_result(location, HTMLtoPDFResult)
        
        # Get result asset
        result_asset = pdf_services_response.get_result().get_asset()
        
        # Download and save the result
        stream_asset = pdf_services.get_content(result_asset)
        
        with open(output_path, "wb") as file:
            file.write(stream_asset.get_input_stream())
        
        # Clean up temp file
        os.unlink(temp_html_path)
        
        file_size = os.path.getsize(output_path) / 1024
        print(f"✅ Adobe PDF created: {output_path} ({file_size:.1f} KB)")
        return True
        
    except Exception as e:
        print(f"❌ Adobe PDF conversion failed: {e}")
        # Clean up temp file if it exists
        if 'temp_html_path' in locals() and os.path.exists(temp_html_path):
            os.unlink(temp_html_path)
        return False

def create_complex_script_pdf_hybrid(pages_data: List[List[Dict]], output_path: str) -> str:
    """
    Hybrid approach: Try Adobe PDF Services first, fallback to ReportLab
    """
    print("🎯 HYBRID COMPLEX SCRIPT PDF CREATION")
    print("=" * 50)
    
    # Check if any page contains complex scripts
    complex_script_found = False
    detected_script = None
    
    for page_elements in pages_data:
        for element in page_elements:
            text = element['text']
            script = detect_complex_script(text)
            if script:
                complex_script_found = True
                detected_script = script
                break
        if complex_script_found:
            break
    
    if not complex_script_found:
        print("📝 No complex scripts detected, using ReportLab...")
        return create_clean_text_pdf_reportlab(pages_data, output_path)
    
    print(f"🔤 Complex script detected: {detected_script}")
    
    # Try Adobe PDF Services first
    if ADOBE_AVAILABLE:
        print("🌐 Attempting Adobe PDF Services conversion...")
        success = create_complex_script_pdf_adobe(pages_data, output_path, detected_script)
        
        if success:
            print("✅ Adobe PDF Services conversion successful!")
            return output_path
        else:
            print("⚠️ Adobe conversion failed, falling back to ReportLab...")
    
    # Fallback to ReportLab with warning
    print("⚠️ Using ReportLab fallback (may have rendering issues)")
    return create_clean_text_pdf_reportlab(pages_data, output_path)

def create_clean_text_pdf_reportlab(pages_data: List[List[Dict]], output_path: str) -> str:
    """
    Fallback ReportLab implementation (original method)
    """
    print(f"🔨 Creating PDF with ReportLab (fallback)")
    print(f"📝 Output: {output_path}")
    
    # Create PDF
    c = canvas.Canvas(output_path, pagesize=A4)
    page_width, page_height = A4
    
    for page_num, page_elements in enumerate(pages_data):
        print(f"   📄 Page {page_num + 1}: {len(page_elements)} text elements")
        
        if not page_elements:
            continue
        
        # Sort elements by vertical position (top to bottom)
        page_elements.sort(key=lambda x: -x['bbox'][1])
        
        for element in page_elements:
            text = element['text']
            bbox = element['bbox']
            font_info = element['font']
            
            # Position coordinates
            x = bbox[0]
            y = page_height - bbox[1]  # Flip Y coordinate for ReportLab
            
            # Font size
            font_size = font_info['size']
            if font_size < 6:
                font_size = 6
            elif font_size > 24:
                font_size = 24
            
            # Font selection
            font_name = "Helvetica"
            flags = font_info.get('flags', 0)
            if flags & 2**4:  # Bold flag
                font_name = "Helvetica-Bold"
            
            c.setFont(font_name, font_size)
            c.setFillColor(black)
            
            # Draw text
            try:
                c.drawString(x, y, text)
            except:
                # Handle encoding issues
                safe_text = text.encode('utf-8', 'ignore').decode('utf-8')
                c.drawString(x, y, safe_text)
        
        # New page if not the last page
        if page_num < len(pages_data) - 1:
            c.showPage()
    
    c.save()
    
    file_size = os.path.getsize(output_path) / 1024
    print(f"✅ Created ReportLab PDF: {file_size:.1f} KB")
    
    return output_path

def extract_text_with_positions(pdf_path: str):
    """
    Extract text with positioning from a PDF using PyMuPDF
    """
    print(f"🔍 EXTRACTING TEXT WITH POSITIONS")
    print(f"📄 Input: {pdf_path}")
    print("-" * 40)
    
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        all_pages_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get text with detailed position information
            text_dict = page.get_text("dict")
            
            page_data = []
            for block in text_dict["blocks"]:
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:  # Only add non-empty text
                                bbox = span["bbox"]  # [x0, y0, x1, y1]
                                font_info = {
                                    'size': span["size"],
                                    'font': span["font"],
                                    'flags': span["flags"]  # Bold, italic, etc.
                                }
                                
                                page_data.append({
                                    'text': text,
                                    'bbox': bbox,
                                    'font': font_info
                                })
            
            all_pages_data.append(page_data)
            print(f"   📄 Page {page_num + 1}: {len(page_data)} text elements")
        
        doc.close()
        print(f"✅ Extracted text from {len(all_pages_data)} pages")
        return all_pages_data
        
    except Exception as e:
        print(f"❌ Text extraction error: {e}")
        raise

def create_complex_script_pdf(adobe_ocr_pdf: str, output_path: str = None) -> str:
    """
    Convert Adobe OCR PDF to clean text-based PDF with complex script support
    """
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(adobe_ocr_pdf))[0]
        output_path = f"{base_name}_complex_script.pdf"
    
    print("🚀 COMPLEX SCRIPT PDF CONVERSION")
    print("=" * 50)
    print(f"📂 Input: {adobe_ocr_pdf}")
    print(f"📝 Output: {output_path}")
    print()
    
    # Step 1: Extract text with positions
    pages_data = extract_text_with_positions(adobe_ocr_pdf)
    
    # Step 2: Create PDF with complex script support
    result_path = create_complex_script_pdf_hybrid(pages_data, output_path)
    
    print(f"\n🎉 CONVERSION COMPLETE!")
    print(f"📁 Result: {result_path}")
    
    # Check if complex scripts were detected
    complex_scripts_found = []
    for page_elements in pages_data:
        for element in page_elements:
            script = detect_complex_script(element['text'])
            if script and script not in complex_scripts_found:
                complex_scripts_found.append(script)
    
    if complex_scripts_found:
        print(f"🔤 Complex scripts detected: {', '.join(complex_scripts_found)}")
        print(f"✨ Professional rendering with Adobe PDF Services")
    else:
        print(f"📝 Standard text processing with ReportLab")
    
    return result_path

def main():
    """Main function"""
    
    # Check if we have the Adobe OCR result
    adobe_ocr_result = "scanned_adobe_ocr.pdf"
    
    if not os.path.exists(adobe_ocr_result):
        print("❌ Adobe OCR result not found!")
        print("   Run: python adobe_ocr_service.py")
        print("   Then run this script again.")
        return
    
    print("🎯 Complex Script PDF Conversion")
    print("=" * 50)
    print("This script handles Hindi, Arabic, Thai, and other complex scripts")
    print("that don't render properly in ReportLab.")
    print()
    
    result = create_complex_script_pdf(adobe_ocr_result)
    print(f"\n🚀 Opening result...")
    os.system(f"open '{result}'")

if __name__ == "__main__":
    main() 