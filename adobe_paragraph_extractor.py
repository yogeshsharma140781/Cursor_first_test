#!/usr/bin/env python3
"""
Adobe Extract API Paragraph Grouping
Use Adobe Extract API to group text into meaningful paragraphs and create formatted PDF
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import zipfile
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)

try:
    from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
    from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
    from adobe.pdfservices.operation.pdf_services import PDFServices
    from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
    from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
    from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
    from adobe.pdfservices.operation.pdfjobs.jobs.html_to_pdf_job import HTMLtoPDFJob
    from adobe.pdfservices.operation.pdfjobs.result.html_to_pdf_result import HTMLtoPDFResult
    from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
    from adobe.pdfservices.operation.io.stream_asset import StreamAsset
    ADOBE_SDK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Adobe PDF Services SDK not available: {e}")
    ADOBE_SDK_AVAILABLE = False

class AdobeParagraphExtractor:
    """Adobe Extract API for intelligent paragraph grouping"""
    
    def __init__(self, credentials_path: str = "backend/pdfservices-api-credentials.json"):
        """Initialize Adobe Extract API processor"""
        
        self.credentials = None
        self.pdf_services = None
        self.credentials_path = credentials_path
        
        self.setup_adobe_credentials()
    
    def setup_adobe_credentials(self):
        """Setup Adobe PDF Services credentials"""
        
        print("🔑 Setting up Adobe PDF Services credentials...")
        
        if not ADOBE_SDK_AVAILABLE:
            print("❌ Adobe PDF Services SDK not available")
            return False
        
        try:
            # Load credentials from file
            if os.path.exists(self.credentials_path):
                with open(self.credentials_path, 'r') as f:
                    creds_data = json.load(f)
                
                client_id = creds_data.get('client_credentials', {}).get('client_id')
                client_secret = creds_data.get('client_credentials', {}).get('client_secret')
                
                if client_id and client_secret:
                    self.credentials = ServicePrincipalCredentials(
                        client_id=client_id,
                        client_secret=client_secret
                    )
                    
                    # Create PDF Services instance
                    self.pdf_services = PDFServices(credentials=self.credentials)
                    print("✅ Adobe PDF Services initialized successfully")
                    return True
            
            print("❌ Adobe credentials not found")
            return False
            
        except Exception as e:
            print(f"❌ Error setting up Adobe credentials: {e}")
            return False
    
    def extract_structured_content(self, pdf_path: str) -> Dict[str, Any]:
        """Extract structured content using Adobe Extract API"""
        
        if not self.pdf_services:
            print("❌ Adobe PDF Services not initialized")
            return None
        
        try:
            print(f"📄 Extracting structured content from {pdf_path}...")
            
            # Check if PDF file exists
            if not os.path.exists(pdf_path):
                print(f"❌ PDF file not found: {pdf_path}")
                return None
            
            # Upload PDF to Adobe
            with open(pdf_path, 'rb') as file:
                input_stream = file.read()
            
            input_asset = self.pdf_services.upload(
                input_stream=input_stream,
                mime_type=PDFServicesMediaType.PDF
            )
            
            # Create extract job with text focus
            extract_pdf_params = ExtractPDFParams(
                elements_to_extract=[
                    ExtractElementType.TEXT
                ]
            )
            extract_pdf_job = ExtractPDFJob(
                input_asset=input_asset, 
                extract_pdf_params=extract_pdf_params
            )
            
            # Submit job
            print("🚀 Submitting Adobe Extract job...")
            location = self.pdf_services.submit(extract_pdf_job)
            
            # Get result
            print("⏳ Waiting for extraction completion...")
            pdf_services_response = self.pdf_services.get_job_result(location, ExtractPDFResult)
            result_asset = pdf_services_response.get_result().get_resource()
            stream_asset = self.pdf_services.get_content(result_asset)
            
            # Save extraction result
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            extraction_output = f"adobe_extract_paragraphs_{timestamp}.zip"
            
            with open(extraction_output, "wb") as file:
                file.write(stream_asset.get_input_stream())
            
            print(f"✅ Adobe extraction completed: {extraction_output}")
            
            # Extract and parse JSON
            extract_dir = f"adobe_extract_temp_{timestamp}"
            with zipfile.ZipFile(extraction_output, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Load structured data
            json_file = os.path.join(extract_dir, "structuredData.json")
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    extraction_data = json.load(f)
                
                print(f"✅ Loaded structured data with {len(extraction_data.get('elements', []))} elements")
                
                # Clean up
                os.remove(extraction_output)
                
                return extraction_data
            else:
                print("❌ No structured data found in Adobe extraction")
                return None
                
        except Exception as e:
            print(f"❌ Adobe extraction error: {e}")
            return None
    
    def group_elements_into_paragraphs(self, extraction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Group Adobe elements into meaningful paragraphs"""
        
        if not extraction_data or 'elements' not in extraction_data:
            print("❌ No elements found in extraction data")
            return []
        
        elements = extraction_data['elements']
        text_elements = [elem for elem in elements if elem.get('Text')]
        
        print(f"📝 Processing {len(text_elements)} text elements into paragraphs...")
        
        # Sort elements by position (top to bottom, left to right)
        sorted_elements = sorted(text_elements, key=lambda x: (
            x.get('Bounds', [0, 0, 0, 0])[1],  # Y position (top)
            x.get('Bounds', [0, 0, 0, 0])[0]   # X position (left)
        ))
        
        paragraphs = []
        current_paragraph = []
        
        # Parameters for paragraph grouping
        line_height_threshold = 30   # Maximum line height difference
        paragraph_gap_threshold = 20  # Minimum gap between paragraphs
        indent_threshold = 20        # Minimum indentation to consider
        
        previous_bottom = None
        previous_left = None
        
        for i, elem in enumerate(sorted_elements):
            bounds = elem.get('Bounds', [0, 0, 0, 0])
            left, top, width, height = bounds
            bottom = top + height
            text = elem.get('Text', '').strip()
            
            # Skip empty text
            if not text:
                continue
            
            # Determine if this starts a new paragraph
            start_new_paragraph = False
            
            if previous_bottom is None:
                # First element
                start_new_paragraph = True
            else:
                # Calculate gap from previous line
                gap = top - previous_bottom
                
                # Check for paragraph break indicators
                if gap > paragraph_gap_threshold:
                    start_new_paragraph = True
                elif previous_left is not None and abs(left - previous_left) > indent_threshold:
                    # Significant indentation change
                    start_new_paragraph = True
                elif text.endswith('.') and gap > line_height_threshold / 2:
                    # End of sentence with moderate gap
                    start_new_paragraph = True
                elif len(text) < 50 and gap > line_height_threshold / 3:
                    # Short line with small gap (might be title/header)
                    start_new_paragraph = True
            
            # Start new paragraph if needed
            if start_new_paragraph and current_paragraph:
                paragraphs.append(self.create_paragraph_from_elements(current_paragraph))
                current_paragraph = []
            
            # Add element to current paragraph
            current_paragraph.append(elem)
            
            # Update tracking variables
            previous_bottom = bottom
            previous_left = left
        
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(self.create_paragraph_from_elements(current_paragraph))
        
        print(f"✅ Created {len(paragraphs)} paragraphs from {len(text_elements)} text elements")
        
        return paragraphs
    
    def create_paragraph_from_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a paragraph from a list of elements"""
        
        if not elements:
            return {}
        
        # Combine text with proper spacing
        text_parts = []
        for elem in elements:
            text = elem.get('Text', '').strip()
            if text:
                text_parts.append(text)
        
        # Join with spaces, but be smart about it
        paragraph_text = ' '.join(text_parts)
        
        # Clean up common OCR artifacts
        paragraph_text = self.clean_paragraph_text(paragraph_text)
        
        # Calculate bounding box for entire paragraph
        all_bounds = [elem.get('Bounds', [0, 0, 0, 0]) for elem in elements]
        min_x = min(bounds[0] for bounds in all_bounds)
        min_y = min(bounds[1] for bounds in all_bounds)
        max_x = max(bounds[0] + bounds[2] for bounds in all_bounds)
        max_y = max(bounds[1] + bounds[3] for bounds in all_bounds)
        
        # Determine paragraph type
        paragraph_type = self.classify_paragraph(paragraph_text, elements)
        
        return {
            'text': paragraph_text,
            'bounds': [min_x, min_y, max_x - min_x, max_y - min_y],
            'element_count': len(elements),
            'type': paragraph_type,
            'line_count': len(elements),
            'word_count': len(paragraph_text.split()),
            'original_elements': elements
        }
    
    def clean_paragraph_text(self, text: str) -> str:
        """Clean and normalize paragraph text"""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR spacing issues
        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace(' ;', ';')
        text = text.replace(' :', ':')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')
        
        # Fix quotes
        text = text.replace(' "', '"')
        text = text.replace('" ', '"')
        
        return text
    
    def classify_paragraph(self, text: str, elements: List[Dict[str, Any]]) -> str:
        """Classify paragraph type based on content and structure"""
        
        # Check for headers/titles
        if len(elements) == 1 and len(text) < 100:
            return 'header'
        
        # Check for addresses
        if any(keyword in text.lower() for keyword in ['straat', 'laan', 'weg', 'amsterdam', 'heerlen']):
            return 'address'
        
        # Check for dates
        if any(keyword in text.lower() for keyword in ['datum', 'date', '2023', '2024', '2025']):
            return 'date'
        
        # Check for signatures/closings
        if any(keyword in text.lower() for keyword in ['hoogachtend', 'vriendelijk', 'groet']):
            return 'signature'
        
        # Check for reference numbers
        if any(keyword in text.lower() for keyword in ['nummer', 'referentie', 'kenmerk']):
            return 'reference'
        
        # Default to body text
        return 'body'
    
    def create_formatted_html(self, paragraphs: List[Dict[str, Any]], title: str = "Structured Document") -> str:
        """Create well-formatted HTML from paragraphs"""
        
        print(f"🎨 Creating formatted HTML from {len(paragraphs)} paragraphs...")
        
        html_content = f"""<!DOCTYPE html>
<html lang="nl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #ffffff;
            color: #333;
        }}
        
        .document-header {{
            text-align: center;
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 30px;
            margin-bottom: 40px;
        }}
        
        .document-title {{
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .document-subtitle {{
            font-size: 14px;
            color: #7f8c8d;
            font-style: italic;
        }}
        
        .paragraph {{
            margin-bottom: 20px;
            text-align: justify;
        }}
        
        .paragraph.header {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 30px;
            margin-bottom: 15px;
            text-align: left;
        }}
        
        .paragraph.address {{
            font-style: italic;
            color: #555;
            margin-bottom: 25px;
            line-height: 1.4;
        }}
        
        .paragraph.date {{
            color: #666;
            font-size: 14px;
            margin-bottom: 25px;
        }}
        
        .paragraph.reference {{
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin-bottom: 25px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }}
        
        .paragraph.signature {{
            margin-top: 30px;
            font-style: italic;
            color: #555;
        }}
        
        .paragraph.body {{
            font-size: 16px;
            line-height: 1.7;
            margin-bottom: 20px;
        }}
        
        .paragraph-meta {{
            font-size: 11px;
            color: #95a5a6;
            margin-bottom: 5px;
            font-family: 'Arial', sans-serif;
        }}
        
        .footer {{
            margin-top: 50px;
            text-align: center;
            font-size: 12px;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
            padding-top: 30px;
        }}
        
        .stats {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        
        .stats-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        
        .stat-item {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="document-header">
        <div class="document-title">{title}</div>
        <div class="document-subtitle">Intelligently structured using Adobe Extract API</div>
    </div>
    
    <div class="stats">
        <div class="stats-title">Document Analysis</div>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{len(paragraphs)}</div>
                <div class="stat-label">Paragraphs</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{sum(p.get('word_count', 0) for p in paragraphs)}</div>
                <div class="stat-label">Words</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{sum(p.get('line_count', 0) for p in paragraphs)}</div>
                <div class="stat-label">Lines</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(set(p.get('type', 'body') for p in paragraphs))}</div>
                <div class="stat-label">Types</div>
            </div>
        </div>
    </div>
    
    <div class="content">
"""
        
        # Add paragraphs
        for i, paragraph in enumerate(paragraphs):
            paragraph_type = paragraph.get('type', 'body')
            word_count = paragraph.get('word_count', 0)
            line_count = paragraph.get('line_count', 0)
            
            html_content += f"""
        <div class="paragraph {paragraph_type}">
            <div class="paragraph-meta">
                Paragraph {i+1} | Type: {paragraph_type.title()} | Words: {word_count} | Lines: {line_count}
            </div>
            {paragraph['text']}
        </div>
"""
        
        # Add footer
        html_content += f"""
    </div>
    
    <div class="footer">
        <p><strong>Document processed on:</strong> {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
        <p><strong>Processing method:</strong> Adobe Extract API with intelligent paragraph grouping</p>
        <p><strong>Total paragraphs:</strong> {len(paragraphs)} | <strong>Total words:</strong> {sum(p.get('word_count', 0) for p in paragraphs)}</p>
    </div>
</body>
</html>"""
        
        return html_content
    
    def create_pdf_from_html(self, html_content: str, output_pdf_path: str) -> bool:
        """Create PDF from HTML using Adobe PDF Services"""
        
        if not self.pdf_services:
            print("❌ Adobe PDF Services not initialized")
            return False
        
        try:
            print(f"📄 Creating PDF from HTML...")
            
            # Create temporary HTML file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(html_content)
                temp_html = temp_file.name
            
            # Upload HTML to Adobe
            with open(temp_html, 'rb') as file:
                input_stream = file.read()
            
            asset = self.pdf_services.upload(
                input_stream=input_stream,
                mime_type=PDFServicesMediaType.HTML
            )
            
            # Create HTML to PDF job
            html_to_pdf_job = HTMLtoPDFJob(input_asset=asset)
            
            # Submit job
            print("🚀 Submitting HTML to PDF job...")
            location = self.pdf_services.submit(html_to_pdf_job)
            
            # Get result
            print("⏳ Waiting for PDF generation...")
            pdf_services_response = self.pdf_services.get_job_result(location, HTMLtoPDFResult)
            result_asset = pdf_services_response.get_result().get_asset()
            stream_asset = self.pdf_services.get_content(result_asset)
            
            # Save PDF
            with open(output_pdf_path, "wb") as file:
                file.write(stream_asset.get_input_stream())
            
            # Clean up temp file
            os.unlink(temp_html)
            
            file_size = os.path.getsize(output_pdf_path)
            print(f"✅ PDF created successfully: {output_pdf_path} ({file_size:,} bytes)")
            
            return True
            
        except Exception as e:
            print(f"❌ PDF creation error: {e}")
            return False
    
    def process_document_with_paragraphs(self, input_pdf: str = "scanned_ADOBE_OCR_DUTCH.pdf", 
                                       output_prefix: str = "structured_paragraphs"):
        """Complete paragraph extraction and formatting workflow"""
        
        print("🚀 ADOBE PARAGRAPH EXTRACTION WORKFLOW")
        print("=" * 60)
        
        if not self.pdf_services:
            print("❌ Adobe PDF Services not initialized")
            return False
        
        # Step 1: Extract structured content
        print("\n📄 STEP 1: EXTRACTING STRUCTURED CONTENT")
        extraction_data = self.extract_structured_content(input_pdf)
        if not extraction_data:
            print("❌ Content extraction failed")
            return False
        
        # Step 2: Group elements into paragraphs
        print("\n📝 STEP 2: GROUPING ELEMENTS INTO PARAGRAPHS")
        paragraphs = self.group_elements_into_paragraphs(extraction_data)
        if not paragraphs:
            print("❌ Paragraph grouping failed")
            return False
        
        # Step 3: Create formatted HTML
        print("\n🎨 STEP 3: CREATING FORMATTED HTML")
        html_content = self.create_formatted_html(paragraphs, "Dutch Government Document - Structured")
        
        # Save HTML
        html_output = f"{output_prefix}.html"
        with open(html_output, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✅ HTML saved: {html_output}")
        
        # Step 4: Create PDF
        print("\n📄 STEP 4: CREATING STRUCTURED PDF")
        pdf_output = f"{output_prefix}.pdf"
        pdf_success = self.create_pdf_from_html(html_content, pdf_output)
        
        # Summary
        print(f"\n🎉 PARAGRAPH EXTRACTION COMPLETE!")
        print(f"   📄 Input PDF: {input_pdf}")
        print(f"   📝 Extracted: {len(paragraphs)} paragraphs")
        print(f"   📄 HTML Output: {html_output}")
        if pdf_success:
            print(f"   📄 PDF Output: {pdf_output}")
        
        # Paragraph breakdown
        print(f"\n📊 PARAGRAPH BREAKDOWN:")
        type_counts = {}
        for p in paragraphs:
            p_type = p.get('type', 'body')
            type_counts[p_type] = type_counts.get(p_type, 0) + 1
        
        for p_type, count in type_counts.items():
            print(f"   {p_type.title()}: {count} paragraphs")
        
        return pdf_success

def main():
    """Main execution function"""
    
    try:
        # Initialize paragraph extractor
        extractor = AdobeParagraphExtractor()
        
        # Process document with paragraph grouping
        success = extractor.process_document_with_paragraphs()
        
        if success:
            print("\n✅ Adobe paragraph extraction completed successfully!")
        else:
            print("\n❌ Paragraph extraction failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 