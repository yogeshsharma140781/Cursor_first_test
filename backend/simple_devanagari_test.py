#!/usr/bin/env python3

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path
import os

def create_simple_test():
    """Create a simple PDF with just Devanagari text to test rendering"""
    
    # Font paths
    fonts_dir = Path.home() / 'Library' / 'Fonts'
    
    # Register Noto Devanagari fonts
    fonts_to_test = [
        ('NotoDevanagari-Regular', 'NotoSansDevanagari-Regular.ttf'),
        ('NotoDevanagari-Bold', 'NotoSansDevanagari-Bold.ttf'),
        ('NotoSans-Regular', 'NotoSans-Regular.ttf'),
    ]
    
    registered_fonts = []
    
    for font_name, font_file in fonts_to_test:
        font_path = fonts_dir / font_file
        if font_path.exists():
            try:
                pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                registered_fonts.append(font_name)
                print(f"✅ Registered: {font_name}")
            except Exception as e:
                print(f"❌ Failed to register {font_name}: {e}")
        else:
            print(f"⚠️  Font not found: {font_path}")
    
    if not registered_fonts:
        print("❌ No fonts registered! Cannot create test PDF.")
        return
    
    # Create test PDF
    c = canvas.Canvas('simple_devanagari_test.pdf', pagesize=letter)
    
    # Test texts with increasing complexity
    test_cases = [
        ("English Test", "NotoSans-Regular", "This should always work"),
        ("Simple Devanagari", "NotoDevanagari-Regular", "नमस्ते"),
        ("Complex Conjuncts", "NotoDevanagari-Bold", "प्रिय श्री शर्मा"),
        ("Full Sentence", "NotoDevanagari-Regular", "इस पत्र के माध्यम से मैं आपको सूचित करता हूँ।"),
        ("Government Terms", "NotoDevanagari-Bold", "न्याय और सुरक्षा के राज्य सचिव"),
    ]
    
    y_position = 700
    
    # Title
    c.setFont("NotoSans-Regular", 16)
    c.drawString(50, y_position, "Devanagari Font Rendering Test")
    y_position -= 40
    
    # Test each case
    for title, font_name, text in test_cases:
        if font_name in registered_fonts:
            # Title
            c.setFont("NotoSans-Regular", 12)
            c.drawString(50, y_position, f"{title}:")
            y_position -= 20
            
            # Test text
            c.setFont(font_name, 14)
            c.drawString(70, y_position, text)
            y_position -= 30
            
            # Unicode representation
            c.setFont("NotoSans-Regular", 8)
            unicode_repr = " ".join([f"U+{ord(char):04X}" for char in text[:20]])
            c.drawString(70, y_position, f"Unicode: {unicode_repr}")
            y_position -= 40
        else:
            c.setFont("NotoSans-Regular", 12)
            c.drawString(50, y_position, f"{title}: FONT NOT AVAILABLE")
            y_position -= 30
    
    c.save()
    print(f"\n✅ Created: simple_devanagari_test.pdf")
    print("\nNext steps:")
    print("1. Open 'simple_devanagari_test.pdf' in different PDF viewers:")
    print("   - macOS Preview")
    print("   - Adobe Acrobat Reader")
    print("   - Chrome browser")
    print("   - Firefox browser")
    print("\n2. Report which viewers show:")
    print("   ✅ Proper Devanagari characters")
    print("   ❌ Boxes (□) instead of characters")
    print("\nThis will help identify if it's a viewer issue or font embedding issue.")

if __name__ == "__main__":
    create_simple_test() 