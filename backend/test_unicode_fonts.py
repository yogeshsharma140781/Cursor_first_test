#!/usr/bin/env python3
"""
Test script to verify Unicode font support for PDF translation
Tests different scripts and languages to ensure proper rendering
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import os
import sys

# Add current directory to path to import translator_api
sys.path.append(os.path.dirname(__file__))

try:
    from translator_api import (
        register_unicode_fonts, 
        get_best_font_for_language,
        detect_script_from_text,
        ensure_fonts_registered
    )
except ImportError as e:
    print(f"Error importing translator_api: {e}")
    print("Make sure you're running this from the backend directory")
    sys.exit(1)

def test_unicode_fonts():
    """Test Unicode font registration and rendering"""
    print("Testing Unicode Font Support")
    print("=" * 40)
    
    # Ensure fonts are registered
    ensure_fonts_registered()
    
    # Get list of registered fonts
    registered_fonts = pdfmetrics.getRegisteredFontNames()
    print(f"Total registered fonts: {len(registered_fonts)}")
    
    # Test different languages and scripts
    test_cases = [
        {
            'lang': 'en',
            'text': 'Hello World! This is English text.',
            'expected_script': 'default'
        },
        {
            'lang': 'hi',
            'text': 'नमस्ते दुनिया! यह हिंदी पाठ है।',
            'expected_script': 'devanagari'
        },
        {
            'lang': 'ja',
            'text': 'こんにちは世界！これは日本語のテキストです。',
            'expected_script': 'cjk'
        },
        {
            'lang': 'zh',
            'text': '你好世界！这是中文文本。',
            'expected_script': 'cjk'
        },
        {
            'lang': 'ar',
            'text': 'مرحبا بالعالم! هذا نص عربي.',
            'expected_script': 'arabic'
        },
        {
            'lang': 'ko',
            'text': '안녕하세요 세계! 이것은 한국어 텍스트입니다.',
            'expected_script': 'cjk'
        },
        {
            'lang': 'ru',
            'text': 'Привет мир! Это русский текст.',
            'expected_script': 'cyrillic'
        }
    ]
    
    print("\nTesting script detection:")
    print("-" * 30)
    
    for case in test_cases:
        detected_script = detect_script_from_text(case['text'])
        status = "✓" if detected_script == case['expected_script'] else "✗"
        print(f"{status} {case['lang']:3} -> {detected_script:12} (expected: {case['expected_script']})")
    
    print("\nTesting font selection:")
    print("-" * 30)
    
    for case in test_cases:
        font_name = get_best_font_for_language(case['lang'], case['text'])
        available = "✓" if font_name in registered_fonts else "✗"
        print(f"{available} {case['lang']:3} -> {font_name:25} ({'available' if available == '✓' else 'fallback'})")
    
    # Create a test PDF with Unicode text
    print("\nGenerating test PDF...")
    create_test_pdf(test_cases)
    
    print("\nFont test completed!")

def create_test_pdf(test_cases):
    """Create a test PDF with Unicode text samples"""
    output_path = "unicode_font_test.pdf"
    
    try:
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Unicode Font Test - PDF Translation Support")
        
        y_position = height - 100
        
        for i, case in enumerate(test_cases):
            if y_position < 100:  # Start new page if needed
                c.showPage()
                y_position = height - 50
            
            # Language label
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_position, f"{case['lang'].upper()}:")
            
            # Get appropriate font for this language
            font_name = get_best_font_for_language(case['lang'], case['text'])
            
            try:
                c.setFont(font_name, 14)
                c.drawString(100, y_position - 20, case['text'])
                status = "Rendered successfully"
                status_color = (0, 0.7, 0)  # Green
            except Exception as e:
                # Fallback to Helvetica if Unicode font fails
                c.setFont("Helvetica", 14)
                # Replace non-ASCII characters with placeholders for fallback
                fallback_text = ''.join(c if ord(c) < 128 else '?' for c in case['text'])
                c.drawString(100, y_position - 20, fallback_text)
                status = f"Fallback used: {str(e)[:50]}"
                status_color = (0.7, 0, 0)  # Red
            
            # Status message
            c.setFillColorRGB(*status_color)
            c.setFont("Helvetica", 8)
            c.drawString(100, y_position - 35, f"Font: {font_name} - {status}")
            c.setFillColorRGB(0, 0, 0)  # Reset to black
            
            y_position -= 80
        
        c.save()
        print(f"Test PDF created: {output_path}")
        
    except Exception as e:
        print(f"Error creating test PDF: {e}")

def main():
    """Main function"""
    try:
        test_unicode_fonts()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 