#!/usr/bin/env python3

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import subprocess

def find_system_devanagari_fonts():
    """Find all Devanagari fonts on the system"""
    system_fonts = []
    
    # Common system locations for fonts
    font_paths = [
        "/System/Library/Fonts/",
        "/System/Library/Fonts/Supplemental/",
        "/Library/Fonts/",
        str(os.path.expanduser("~/Library/Fonts/")),
    ]
    
    # Look for Devanagari fonts
    devanagari_keywords = ['devanagari', 'hindi', 'kohinoor', 'shree', 'mangal', 'noto']
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                for filename in os.listdir(font_path):
                    if filename.lower().endswith(('.ttf', '.otf')):
                        for keyword in devanagari_keywords:
                            if keyword in filename.lower():
                                system_fonts.append({
                                    'name': filename,
                                    'path': os.path.join(font_path, filename),
                                    'keyword': keyword
                                })
                                break
            except PermissionError:
                continue
    
    return system_fonts

def test_system_fonts():
    """Test different system fonts for Devanagari rendering"""
    
    # Find system fonts
    system_fonts = find_system_devanagari_fonts()
    
    print("=== SYSTEM DEVANAGARI FONTS TEST ===")
    print(f"Found {len(system_fonts)} potential Devanagari fonts:")
    for font in system_fonts:
        print(f"  - {font['name']} ({font['keyword']})")
    print()
    
    if not system_fonts:
        print("❌ No system Devanagari fonts found")
        return
    
    # Test text
    test_text = "प्रिय श्री शर्मा"
    
    # Create PDF
    c = canvas.Canvas('system_fonts_test.pdf', pagesize=letter)
    
    y_position = 750
    
    # Title
    c.setFont("Helvetica", 16)
    c.drawString(50, y_position, "System Devanagari Fonts Test")
    y_position -= 30
    
    c.setFont("Helvetica", 10)
    c.drawString(50, y_position, f"Test text: {test_text}")
    y_position -= 30
    
    registered_count = 0
    
    # Test each font
    for i, font_info in enumerate(system_fonts):
        font_name = f"SystemFont{i}"
        font_path = font_info['path']
        
        try:
            # Try to register the font
            pdfmetrics.registerFont(TTFont(font_name, font_path))
            registered_count += 1
            
            # Font info
            c.setFont("Helvetica", 10)
            c.drawString(50, y_position, f"Font: {font_info['name']}")
            y_position -= 15
            
            # Test the font
            c.setFont(font_name, 14)
            c.drawString(70, y_position, test_text)
            y_position -= 25
            
            # Test metrics
            from reportlab.pdfbase.pdfmetrics import stringWidth
            width = stringWidth(test_text, font_name, 14)
            c.setFont("Helvetica", 8)
            c.drawString(70, y_position, f"Width: {width:.2f} pts")
            y_position -= 20
            
            print(f"✅ Successfully tested: {font_info['name']}")
            
        except Exception as e:
            c.setFont("Helvetica", 8)
            c.drawString(50, y_position, f"Failed: {font_info['name']} - {str(e)[:50]}")
            y_position -= 15
            print(f"❌ Failed to register: {font_info['name']} - {e}")
        
        if y_position < 100:
            c.showPage()
            y_position = 750
    
    c.save()
    
    print(f"\n✅ Created: system_fonts_test.pdf")
    print(f"Successfully registered {registered_count}/{len(system_fonts)} fonts")
    print("\nNEXT STEPS:")
    print("1. Open system_fonts_test.pdf")
    print("2. Compare rendering quality of different fonts")
    print("3. Identify which font renders conjuncts best")
    print("4. Update translator_api.py to use the best font")

def get_font_info_via_fc_list():
    """Get font information using fc-list if available"""
    try:
        # Try to get Devanagari fonts using fc-list
        result = subprocess.run(['fc-list', ':lang=hi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n=== FC-LIST DEVANAGARI FONTS ===")
            fonts = result.stdout.strip().split('\n')
            for font in fonts[:10]:  # Show first 10
                print(f"  {font}")
            print(f"... and {max(0, len(fonts) - 10)} more")
        else:
            print("fc-list not available (normal on macOS)")
    except FileNotFoundError:
        print("fc-list not available (normal on macOS)")

if __name__ == "__main__":
    test_system_fonts()
    get_font_info_via_fc_list() 