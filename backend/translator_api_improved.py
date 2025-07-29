#!/usr/bin/env python3
"""
Improved Translation API with better Unicode support using WeasyPrint
This addresses ReportLab's limitations with complex scripts
"""

import os
import tempfile
import json
import base64
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import httpx
from dotenv import load_dotenv

# Import the existing functionality
from translator_api import AdvancedPDFLayoutParser, SimpleTextBlock

load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

def extract_and_encode_visual_elements(pdf_content: bytes, page_num: int = 0) -> List[Dict]:
    """Extract visual elements and encode them as base64 for HTML embedding"""
    visual_elements = []
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_file.write(pdf_content)
        tmp_file.flush()
        
        try:
            doc = fitz.open(tmp_file.name)
            page = doc[page_num]
            
            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_base64 = base64.b64encode(img_data).decode()
                        
                        # Get image rectangle
                        img_rect = page.get_image_rects(img)[0] if page.get_image_rects(img) else None
                        if img_rect:
                            visual_elements.append({
                                'type': 'image',
                                'data': f'data:image/png;base64,{img_base64}',
                                'bbox': tuple(img_rect),
                                'index': img_index
                            })
                    
                    pix = None
                except Exception as e:
                    print(f"Error extracting image {img_index}: {e}")
            
            # Extract vector graphics as images
            drawings = page.get_drawings()
            for draw_index, drawing in enumerate(drawings):
                try:
                    # Get the drawing bounds
                    bounds = drawing.get('rect', None)
                    if bounds and bounds.width > 10 and bounds.height > 10:
                        # Create a new document with just this drawing
                        temp_doc = fitz.open()
                        temp_page = temp_doc.new_page(width=bounds.width, height=bounds.height)
                        
                        # Render the drawing area
                        clip_rect = fitz.Rect(bounds)
                        pix = page.get_pixmap(clip=clip_rect, matrix=fitz.Matrix(2, 2))
                        
                        if pix.width > 0 and pix.height > 0:
                            img_data = pix.tobytes("png")
                            img_base64 = base64.b64encode(img_data).decode()
                            
                            visual_elements.append({
                                'type': 'vector',
                                'data': f'data:image/png;base64,{img_base64}',
                                'bbox': tuple(bounds),
                                'index': draw_index
                            })
                        
                        pix = None
                        temp_doc.close()
                        
                except Exception as e:
                    print(f"Error extracting vector drawing {draw_index}: {e}")
            
            # Extract text blocks to find QR codes
            text_blocks = page.get_text("dict")
            for block in text_blocks.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text.startswith("http") or "www." in text:
                                # This might be a QR code data
                                bbox = span.get("bbox", [0, 0, 0, 0])
                                visual_elements.append({
                                    'type': 'qr_code',
                                    'data': text,
                                    'bbox': tuple(bbox),
                                    'url': text
                                })
            
            doc.close()
            
        finally:
            os.unlink(tmp_file.name)
    
    return visual_elements

def create_html_from_blocks(blocks_with_translations: list, visual_elements: list, page_width: float, page_height: float) -> str:
    """Create HTML representation of the PDF with visual elements and better Unicode rendering"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;700&family=Noto+Sans+Arabic:wght@400;700&family=Noto+Sans+SC:wght@400;700&display=swap');
            
            body {{
                margin: 0;
                padding: 20px;
                font-family: 'Noto Sans Devanagari', 'Noto Sans Arabic', 'Noto Sans SC', Arial, sans-serif;
                font-size: 12px;
                line-height: 1.4;
                position: relative;
                width: {page_width}px;
                height: {page_height}px;
                background: white;
            }}
            
            .text-block {{
                position: absolute;
                overflow: hidden;
                word-wrap: break-word;
            }}
            
            .visual-element {{
                position: absolute;
                z-index: 1;
            }}
            
            .qr-code {{
                position: absolute;
                border: 2px solid #333;
                background: white;
                padding: 5px;
                font-size: 8px;
                text-align: center;
                z-index: 2;
            }}
            
            .devanagari {{
                font-family: 'Noto Sans Devanagari', Arial, sans-serif;
                direction: ltr;
                text-align: left;
            }}
            
            .arabic {{
                font-family: 'Noto Sans Arabic', Arial, sans-serif;
                direction: rtl;
                text-align: right;
            }}
            
            .cjk {{
                font-family: 'Noto Sans SC', Arial, sans-serif;
                direction: ltr;
                text-align: left;
            }}
            
            .bold {{
                font-weight: 700;
            }}
            
            .italic {{
                font-style: italic;
            }}
        </style>
    </head>
    <body>
    """
    
    # Add visual elements first (so they appear behind text)
    for element in visual_elements:
        if element['type'] in ['image', 'vector']:
            x0, y0, x1, y1 = element['bbox']
            width = x1 - x0
            height = y1 - y0
            
            html_content += f"""
            <img class="visual-element" 
                 src="{element['data']}" 
                 style="left: {x0}px; top: {y0}px; width: {width}px; height: {height}px;">
            """
        
        elif element['type'] == 'qr_code':
            x0, y0, x1, y1 = element['bbox']
            width = max(40, x1 - x0)
            height = max(40, y1 - y0)
            
            html_content += f"""
            <div class="qr-code" 
                 style="left: {x0}px; top: {y0}px; width: {width}px; height: {height}px;">
                QR<br>{element['data'][:20]}...
            </div>
            """
    
    # Add text blocks
    for block, page_num, translated_text in blocks_with_translations:
        if page_num == 0:  # Only handle first page for now
            x0, y0, x1, y1 = block.bbox
            width = x1 - x0
            height = y1 - y0
            
            # Skip QR code text blocks as we handle them separately
            if block.type == "qr_code":
                continue
            
            # Determine script class
            script_class = "devanagari"  # Default for Hindi
            if any(ord(c) >= 0x0600 and ord(c) <= 0x06FF for c in translated_text):
                script_class = "arabic"
            elif any(ord(c) >= 0x4E00 and ord(c) <= 0x9FFF for c in translated_text):
                script_class = "cjk"
            
            # Font styling
            style_classes = [script_class]
            if getattr(block, 'bold', False):
                style_classes.append('bold')
            if getattr(block, 'italic', False):
                style_classes.append('italic')
            
            font_size = max(8, int(getattr(block, 'size', 12)))
            
            html_content += f"""
            <div class="text-block {' '.join(style_classes)}" 
                 style="left: {x0}px; top: {y0}px; width: {width}px; height: {height}px; font-size: {font_size}px; z-index: 10;">
                {translated_text}
            </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content

async def create_pdf_with_weasyprint(html_content: str, output_path: str) -> bool:
    """Create PDF using WeasyPrint for better Unicode support"""
    try:
        # Check if WeasyPrint is available
        import weasyprint
        
        # Create PDF from HTML
        html_doc = weasyprint.HTML(string=html_content)
        html_doc.write_pdf(output_path)
        
        return True
    except ImportError:
        print("WeasyPrint not available. Install with: pip install weasyprint")
        return False
    except Exception as e:
        print(f"Error creating PDF with WeasyPrint: {e}")
        return False

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/translate-pdf-improved")
async def translate_pdf_improved(
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("en")
):
    """Improved PDF translation with better Unicode support and visual elements"""
    try:
        # Read PDF content
        pdf_content = await file.read()
        
        # Create parser
        parser = AdvancedPDFLayoutParser()
        
        # Extract blocks and visual elements
        blocks_with_pages = parser.extract_blocks_from_pdf(pdf_content)
        
        # Extract visual elements with base64 encoding for HTML
        visual_elements_encoded = extract_and_encode_visual_elements(pdf_content, 0)
        
        print(f"Extracted {len(blocks_with_pages)} initial text blocks")
        print(f"Extracted {len(visual_elements_encoded)} visual elements")
        
        # Remove duplicates
        unique_blocks_with_pages = parser.remove_duplicates(blocks_with_pages)
        print(f"After deduplication: {len(unique_blocks_with_pages)} text blocks")
        
        # Translate blocks
        blocks_with_translations = []
        for block, page_num in unique_blocks_with_pages:
            if block.type == "qr_code":
                blocks_with_translations.append((block, page_num, "[QR Code]"))
            else:
                translated_text = await parser.translate_text_openai(block.text, target_lang)
                blocks_with_translations.append((block, page_num, translated_text))
                print(f"Translated: '{block.text[:50]}...' -> '{translated_text[:50]}...'")
        
        # Get page dimensions
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(pdf_content)
            tmp_file.flush()
            
            doc = fitz.open(tmp_file.name)
            page = doc[0]
            page_width = page.rect.width
            page_height = page.rect.height
            doc.close()
            os.unlink(tmp_file.name)
        
        # Create HTML content with visual elements
        html_content = create_html_from_blocks(blocks_with_translations, visual_elements_encoded, page_width, page_height)
        
        # Try WeasyPrint first, fall back to ReportLab
        output_path = f"translated_{target_lang}_{file.filename}"
        
        if await create_pdf_with_weasyprint(html_content, output_path):
            print(f"Created PDF with WeasyPrint (better Unicode support) - includes {len(visual_elements_encoded)} visual elements")
        else:
            # Fall back to original method
            from translator_api import create_advanced_pdf_with_visuals
            visual_elements_original = parser.extract_visual_elements(pdf_content, 0)
            output_pdf_bytes = await create_advanced_pdf_with_visuals(
                pdf_content, blocks_with_translations, visual_elements_original
            )
            
            with open(output_path, 'wb') as f:
                f.write(output_pdf_bytes)
            print("Created PDF with ReportLab (fallback)")
        
        # Return the PDF
        response = FileResponse(
            path=output_path,
            media_type='application/pdf',
            filename=f"improved_translated_{file.filename}"
        )
        
        return response
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Improved Translation API with Better Unicode Support"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 