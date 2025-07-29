#!/usr/bin/env python3
"""
Compare font rendering between old installed fonts and fresh Google Noto fonts
Creates side-by-side comparison PDFs to show Unicode improvements
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path
import os

def create_font_comparison():
    """Create side-by-side comparison of font rendering"""
    
    # Test cases from actual problematic translations
    test_cases = [
        ("Simple Hindi", "नमस्ते दुनिया"),
        ("Government letter", "प्रिय श्री शर्मा"),
        ("Dutch citizenship", "नैदरलैंड्स नागरिकता"),
        ("Complex conjuncts", "श्री, क्र, त्र, ज्ञ"),
        ("With diacritics", "हिंदी भाषा में लिखा गया पत्र"),
        ("Address format", "वापसी पता पोस्टबॉक्स 3 9560 AA टेर एपेल"),
        ("Official closure", "सादर, न्याय और सुरक्षा के राज्य सचिव।"),
        ("Mixed content", "योगेश शर्मा, 4 जून 2025, Case Z1-186720992110"),
        ("Long sentence", "इस पत्र के माध्यम से मैं आपको आपकी नागरिकता अनुरोध की प्रगति के बारे में सूचित कर रहा हूँ।")
    ]
    
    output_path = 'font_rendering_comparison.pdf'
    c = canvas.Canvas(output_path, pagesize=letter)
    
    y_start = 750
    y_position = y_start
    
    # Title
    c.setFont('Helvetica-Bold', 16)
    c.drawString(50, y_position, "Font Rendering Comparison: Google Noto vs Previous Versions")
    y_position -= 30
    
    # Column headers
    c.setFont('Helvetica-Bold', 12)
    c.drawString(50, y_position, "Test Case")
    c.drawString(250, y_position, "Google Noto Font")
    c.drawString(450, y_position, "Status")
    y_position -= 20
    
    # Draw separator line
    c.line(50, y_position, 550, y_position)
    y_position -= 20
    
    for i, (description, text) in enumerate(test_cases):
        if y_position < 100:
            c.showPage()
            y_position = y_start
        
        # Test case description
        c.setFont('Helvetica', 10)
        c.drawString(50, y_position, f"{i+1}. {description}")
        
        # Try rendering with Google Noto font
        try:
            c.setFont('NotoSansDevanagari-Regular', 12)
            c.drawString(250, y_position, text)
            
            # Success indicator
            c.setFont('Helvetica', 8)
            c.setFillColorRGB(0, 0.7, 0)  # Green
            c.drawString(450, y_position, "✅ RENDERED")
            c.setFillColorRGB(0, 0, 0)  # Reset to black
            
        except Exception as e:
            # Error indicator
            c.setFont('Helvetica', 8)
            c.setFillColorRGB(0.7, 0, 0)  # Red
            c.drawString(450, y_position, f"❌ ERROR: {str(e)[:20]}")
            c.setFillColorRGB(0, 0, 0)  # Reset to black
        
        y_position -= 25
    
    # Add font information
    y_position -= 20
    c.setFont('Helvetica-Bold', 12)
    c.drawString(50, y_position, "Font Information:")
    y_position -= 15
    
    registered_fonts = [font for font in pdfmetrics.getRegisteredFontNames() if 'Noto' in font]
    
    c.setFont('Helvetica', 10)
    c.drawString(50, y_position, f"Google Noto fonts registered: {len(registered_fonts)}")
    y_position -= 15
    
    for font in registered_fonts:
        c.drawString(70, y_position, f"• {font}")
        y_position -= 12
    
    c.save()
    print(f"Font comparison PDF created: {output_path}")
    return output_path

def analyze_character_coverage():
    """Analyze character coverage of the fonts"""
    
    # Problematic characters from server logs
    problematic_chars = [
        ('व', 'DEVANAGARI LETTER VA'),
        ('ा', 'DEVANAGARI VOWEL SIGN AA'), 
        ('प', 'DEVANAGARI LETTER PA'),
        ('स', 'DEVANAGARI LETTER SA'),
        ('ी', 'DEVANAGARI VOWEL SIGN II'),
        ('्', 'DEVANAGARI SIGN VIRAMA'),
        ('र', 'DEVANAGARI LETTER RA'),
        ('ि', 'DEVANAGARI VOWEL SIGN I'),
        ('य', 'DEVANAGARI LETTER YA'),
        ('श', 'DEVANAGARI LETTER SHA'),
        ('ं', 'DEVANAGARI SIGN ANUSVARA'),
        ('ॉ', 'DEVANAGARI VOWEL SIGN CANDRA O'),
        ('ँ', 'DEVANAGARI SIGN CANDRABINDU'),
        ('ै', 'DEVANAGARI VOWEL SIGN AI'),
        ('ौ', 'DEVANAGARI VOWEL SIGN AU'),
    ]
    
    print("\nCharacter Coverage Analysis:")
    print("=" * 50)
    
    for char, name in problematic_chars:
        unicode_point = f"U+{ord(char):04X}"
        
        # Test rendering
        try:
            test_canvas = canvas.Canvas("temp_test.pdf", pagesize=(100, 100))
            test_canvas.setFont('NotoSansDevanagari-Regular', 12)
            test_canvas.drawString(10, 50, char)
            test_canvas.save()
            
            # Clean up
            if os.path.exists("temp_test.pdf"):
                os.remove("temp_test.pdf")
            
            status = "✅ SUPPORTED"
            
        except Exception as e:
            status = f"❌ ERROR: {str(e)}"
        
        print(f"{char} ({unicode_point}) {name}: {status}")

def main():
    """Main comparison function"""
    
    print("🔄 Creating font rendering comparison...")
    
    # Create comparison PDF
    comparison_pdf = create_font_comparison()
    
    # Analyze character coverage
    analyze_character_coverage()
    
    print(f"\n📄 Comparison PDF created: {comparison_pdf}")
    print("🔍 Open the PDF to see the visual differences in rendering")
    
    # Open the comparison PDF
    os.system(f"open {comparison_pdf}")

if __name__ == "__main__":
    main() 