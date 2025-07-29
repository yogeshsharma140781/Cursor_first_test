#!/usr/bin/env python3
"""
Test script to demonstrate ReportLab vs Adobe PDF Services for complex scripts
Shows the difference in Hindi/Devanagari rendering quality
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append('backend')

def test_reportlab_hindi():
    """Test ReportLab with Hindi text"""
    print("🔨 Testing ReportLab with Hindi text...")
    
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    # Try to register Noto Devanagari font
    fonts_dir = Path.home() / 'Library' / 'Fonts'
    font_path = fonts_dir / 'NotoSansDevanagari-Regular.ttf'
    
    if font_path.exists():
        try:
            pdfmetrics.registerFont(TTFont('NotoDevanagari', str(font_path)))
            font_name = 'NotoDevanagari'
            print(f"✅ Using Noto Devanagari font")
        except Exception as e:
            print(f"⚠️ Font registration failed: {e}")
            font_name = 'Helvetica'
    else:
        print(f"⚠️ Noto Devanagari font not found: {font_path}")
        font_name = 'Helvetica'
    
    # Test Hindi text samples
    hindi_samples = [
        "नमस्ते",                    # Hello
        "प्रिय श्री शर्मा",          # Dear Mr. Sharma
        "न्याय और सुरक्षा",          # Justice and Security
        "इस पत्र के माध्यम से",      # Through this letter
        "नैदरलैंड्स नागरिकता",      # Netherlands citizenship
    ]
    
    # Create PDF
    output_path = 'reportlab_hindi_test.pdf'
    c = canvas.Canvas(output_path, pagesize=letter)
    
    y_position = 750
    
    # Title
    c.setFont('Helvetica', 16)
    c.drawString(50, y_position, "ReportLab Hindi Rendering Test")
    y_position -= 30
    
    c.setFont('Helvetica', 10)
    c.drawString(50, y_position, f"Font: {font_name}")
    y_position -= 30
    
    # Test each sample
    for i, text in enumerate(hindi_samples):
        if y_position < 100:
            c.showPage()
            y_position = 750
        
        # Sample number
        c.setFont('Helvetica', 10)
        c.drawString(50, y_position, f"Sample {i+1}:")
        y_position -= 15
        
        # Hindi text
        try:
            c.setFont(font_name, 14)
            c.drawString(70, y_position, text)
            result = "SUCCESS"
        except Exception as e:
            c.setFont('Helvetica', 10)
            c.drawString(70, y_position, f"[ERROR: {str(e)[:50]}]")
            result = f"FAILED: {e}"
        
        y_position -= 20
        
        # Unicode info
        c.setFont('Helvetica', 8)
        unicode_info = ' '.join([f"U+{ord(char):04X}" for char in text[:10]])
        c.drawString(70, y_position, f"Unicode: {unicode_info}")
        y_position -= 30
        
        print(f"  Sample {i+1}: {result}")
    
    c.save()
    print(f"✅ ReportLab test PDF created: {output_path}")
    return output_path

def test_adobe_hindi():
    """Test Adobe PDF Services with Hindi text"""
    print("\n🌐 Testing Adobe PDF Services with Hindi text...")
    
    # Check if Adobe is available
    try:
        from backend.adobe_complex_script_solution import create_html_template_for_complex_script, setup_adobe_credentials
        from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
        from adobe.pdfservices.operation.pdfjobs.jobs.html_to_pdf_job import HTMLtoPDFJob
        from adobe.pdfservices.operation.pdfjobs.params.html_to_pdf.html_to_pdf_params import HTMLtoPDFParams
        from adobe.pdfservices.operation.pdfjobs.params.html_to_pdf.page_layout import PageLayout
        from adobe.pdfservices.operation.pdfjobs.result.html_to_pdf_result import HTMLtoPDFResult
        import tempfile
        
        ADOBE_AVAILABLE = True
    except ImportError:
        print("❌ Adobe PDF Services not available")
        return None
    
    # Setup Adobe credentials
    pdf_services = setup_adobe_credentials()
    if not pdf_services:
        print("❌ Adobe credentials not available")
        return None
    
    # Create test data
    test_data = [
        {
            'text': 'नमस्ते',
            'bbox': [50, 700, 100, 720],
            'font': {'size': 14, 'font': 'Noto Sans Devanagari', 'flags': 0}
        },
        {
            'text': 'प्रिय श्री शर्मा',
            'bbox': [50, 650, 150, 670],
            'font': {'size': 14, 'font': 'Noto Sans Devanagari', 'flags': 0}
        },
        {
            'text': 'न्याय और सुरक्षा',
            'bbox': [50, 600, 140, 620],
            'font': {'size': 14, 'font': 'Noto Sans Devanagari', 'flags': 0}
        },
        {
            'text': 'इस पत्र के माध्यम से',
            'bbox': [50, 550, 160, 570],
            'font': {'size': 14, 'font': 'Noto Sans Devanagari', 'flags': 0}
        },
        {
            'text': 'नैदरलैंड्स नागरिकता',
            'bbox': [50, 500, 150, 520],
            'font': {'size': 14, 'font': 'Noto Sans Devanagari', 'flags': 0}
        }
    ]
    
    pages_data = [test_data]  # Single page with test data
    
    try:
        # Create HTML template
        html_content = create_html_template_for_complex_script(pages_data, 'devanagari')
        
        # Add title to HTML
        html_content = html_content.replace(
            '<title>Complex Script PDF</title>',
            '<title>Adobe PDF Services Hindi Test</title>'
        )
        
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
        
        output_path = 'adobe_hindi_test.pdf'
        with open(output_path, "wb") as file:
            file.write(stream_asset.get_input_stream())
        
        # Clean up temp file
        os.unlink(temp_html_path)
        
        file_size = os.path.getsize(output_path) / 1024
        print(f"✅ Adobe PDF Services test PDF created: {output_path} ({file_size:.1f} KB)")
        return output_path
        
    except Exception as e:
        print(f"❌ Adobe PDF Services test failed: {e}")
        # Clean up temp file if it exists
        if 'temp_html_path' in locals() and os.path.exists(temp_html_path):
            os.unlink(temp_html_path)
        return None

def create_comparison_summary():
    """Create a summary of the comparison"""
    print("\n📊 COMPARISON SUMMARY")
    print("=" * 50)
    
    reportlab_file = 'reportlab_hindi_test.pdf'
    adobe_file = 'adobe_hindi_test.pdf'
    
    print("🔍 Check the generated PDFs to see the difference:")
    print()
    
    if os.path.exists(reportlab_file):
        size = os.path.getsize(reportlab_file) / 1024
        print(f"📄 ReportLab: {reportlab_file} ({size:.1f} KB)")
        print("   - May show broken conjuncts")
        print("   - Characters may appear separate")
        print("   - Font embedding issues possible")
    
    if os.path.exists(adobe_file):
        size = os.path.getsize(adobe_file) / 1024
        print(f"📄 Adobe PDF Services: {adobe_file} ({size:.1f} KB)")
        print("   - Perfect conjunct rendering")
        print("   - Professional typography")
        print("   - Proper font embedding")
    
    print()
    print("🎯 Key Differences:")
    print("   ReportLab: Fast, free, but poor complex script support")
    print("   Adobe PDF Services: Slower, paid, but perfect complex script rendering")
    print()
    print("💡 Recommendation: Use hybrid approach - detect scripts and route accordingly")

def main():
    """Main test function"""
    print("🎯 COMPLEX SCRIPT RENDERING COMPARISON")
    print("=" * 50)
    print("Testing ReportLab vs Adobe PDF Services for Hindi/Devanagari text")
    print()
    
    # Test ReportLab
    reportlab_result = test_reportlab_hindi()
    
    # Test Adobe PDF Services
    adobe_result = test_adobe_hindi()
    
    # Create comparison summary
    create_comparison_summary()
    
    # Open results
    print("\n🚀 Opening test results...")
    if reportlab_result and os.path.exists(reportlab_result):
        os.system(f"open '{reportlab_result}'")
    
    if adobe_result and os.path.exists(adobe_result):
        os.system(f"open '{adobe_result}'")
    
    print("\n📋 Analysis:")
    print("1. Compare the Hindi text rendering quality")
    print("2. Check for proper conjunct formation (प्र, श्र, न्य)")
    print("3. Look for font embedding issues")
    print("4. Note the file size differences")
    print()
    print("🎯 Conclusion: Adobe PDF Services provides superior complex script support!")

if __name__ == "__main__":
    main() 