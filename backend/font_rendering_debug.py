#!/usr/bin/env python3

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path
import unicodedata
import os

def debug_font_rendering():
    """Debug font rendering issues with Devanagari text"""
    
    # Font paths
    fonts_dir = Path.home() / 'Library' / 'Fonts'
    
    # Test text - the problematic phrase
    test_text = "प्रिय श्री शर्मा"
    
    print("=== FONT RENDERING DEBUG ===")
    print(f"Test text: {test_text}")
    print(f"Unicode points: {[hex(ord(c)) for c in test_text]}")
    print()
    
    # Check Unicode normalization
    print("=== UNICODE NORMALIZATION ===")
    nfc_text = unicodedata.normalize('NFC', test_text)
    nfd_text = unicodedata.normalize('NFD', test_text)
    print(f"Original: {test_text} (len: {len(test_text)})")
    print(f"NFC:      {nfc_text} (len: {len(nfc_text)})")
    print(f"NFD:      {nfd_text} (len: {len(nfd_text)})")
    print(f"NFC == Original: {nfc_text == test_text}")
    print()
    
    # Register fonts with different methods
    fonts_to_test = [
        ('NotoDevanagari-Regular', 'NotoSansDevanagari-Regular.ttf'),
        ('NotoDevanagari-Bold', 'NotoSansDevanagari-Bold.ttf'),
        ('SystemDevanagari', '/System/Library/Fonts/Kohinoor.ttc'),
    ]
    
    registered_fonts = []
    
    print("=== FONT REGISTRATION ===")
    for font_name, font_file in fonts_to_test:
        if font_file.startswith('/'):
            font_path = font_file
        else:
            font_path = str(fonts_dir / font_file)  # Convert to string
            
        if os.path.exists(font_path):
            try:
                # Try different registration methods
                if font_path.endswith('.ttc'):
                    # Skip TTC files as they need special handling
                    print(f"⚠️  Skipping TTC file: {font_path} (not supported)")
                    continue
                else:
                    # For TTF files
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                    registered_fonts.append(font_name)
                    print(f"✅ Registered: {font_name}")
                    
            except Exception as e:
                print(f"❌ Failed to register {font_name}: {e}")
        else:
            print(f"⚠️  Font not found: {font_path}")
    
    print()
    print(f"Total registered fonts: {len(registered_fonts)}")
    
    if not registered_fonts:
        print("❌ No fonts registered! Cannot create test PDF.")
        return
    
    # Create test PDF with different rendering approaches
    c = canvas.Canvas('font_rendering_debug.pdf', pagesize=letter)
    
    y_position = 750
    
    # Title
    c.setFont("Helvetica", 16)
    c.drawString(50, y_position, "Devanagari Font Rendering Debug")
    y_position -= 30
    
    # Test each font
    for font_name in registered_fonts:
        try:
            c.setFont("Helvetica", 10)
            c.drawString(50, y_position, f"Font: {font_name}")
            y_position -= 15
            
            # Test different text variants
            test_variants = [
                ("Original", test_text),
                ("NFC Normalized", nfc_text),
                ("NFD Normalized", nfd_text),
                ("Manual Unicode", "\\u092a\\u094d\\u0930\\u093f\\u092f \\u0936\\u094d\\u0930\\u0940 \\u0936\\u0930\\u094d\\u092e\\u093e"),
            ]
            
            for variant_name, variant_text in test_variants:
                try:
                    # Convert manual unicode if needed
                    if variant_text.startswith('\\u'):
                        variant_text = variant_text.encode().decode('unicode_escape')
                    
                    c.setFont(font_name, 14)
                    c.drawString(70, y_position, f"{variant_name}: {variant_text}")
                    y_position -= 20
                    
                    # Test font metrics
                    from reportlab.pdfbase.pdfmetrics import stringWidth
                    width = stringWidth(variant_text, font_name, 14)
                    c.setFont("Helvetica", 8)
                    c.drawString(70, y_position, f"Width: {width:.2f} pts")
                    y_position -= 15
                    
                except Exception as e:
                    c.setFont("Helvetica", 8)
                    c.drawString(70, y_position, f"ERROR: {str(e)}")
                    y_position -= 15
            
            y_position -= 10
            
        except Exception as e:
            c.setFont("Helvetica", 8)
            c.drawString(50, y_position, f"Font {font_name} failed: {str(e)}")
            y_position -= 20
    
    # Add character analysis
    y_position -= 20
    c.setFont("Helvetica", 12)
    c.drawString(50, y_position, "Character Analysis:")
    y_position -= 20
    
    for i, char in enumerate(test_text):
        c.setFont("Helvetica", 8)
        char_info = f"Pos {i}: '{char}' U+{ord(char):04X} ({unicodedata.name(char, 'UNKNOWN')})"
        c.drawString(50, y_position, char_info)
        y_position -= 12
    
    c.save()
    
    print()
    print("✅ Created: font_rendering_debug.pdf")
    print()
    print("NEXT STEPS:")
    print("1. Open font_rendering_debug.pdf")
    print("2. Compare how different fonts render the same text")
    print("3. Check if any font shows correct glyph shapes")
    print("4. Note which normalization method works best")
    print()
    print("EXPECTED BEHAVIOR:")
    print("- All variants should show the same visual result")
    print("- Characters should form proper conjuncts (प्र, श्र)")
    print("- If shapes are wrong, it's a font shaping issue")

if __name__ == "__main__":
    debug_font_rendering() 