#!/usr/bin/env python3

"""
Final solution for Devanagari rendering issues in ReportLab
This tries multiple approaches to get proper conjunct rendering
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path
import os
import unicodedata

def try_register_system_devanagari_fonts():
    """Try to register system Devanagari fonts that might work better"""
    
    registered_fonts = []
    
    # Try to extract fonts from TTC files using different methods
    system_font_attempts = [
        # Try to find extracted/converted versions of system fonts
        {
            'name': 'DevanagariMT-Regular',
            'paths': [
                '/System/Library/Fonts/Supplemental/DevanagariMT.ttc',
                '/Library/Fonts/DevanagariMT.ttf',  # If manually extracted
            ]
        },
        {
            'name': 'Kohinoor-Regular', 
            'paths': [
                '/System/Library/Fonts/Kohinoor.ttc',
                '/Library/Fonts/Kohinoor.ttf',  # If manually extracted
            ]
        },
        {
            'name': 'ShreeDevanagari-Regular',
            'paths': [
                '/System/Library/Fonts/Supplemental/Shree714.ttc',
                '/Library/Fonts/Shree714.ttf',  # If manually extracted
            ]
        }
    ]
    
    print("=== TRYING SYSTEM DEVANAGARI FONTS ===")
    
    for font_info in system_font_attempts:
        font_name = font_info['name']
        
        for font_path in font_info['paths']:
            if os.path.exists(font_path):
                try:
                    if font_path.endswith('.ttc'):
                        # Try different subfont indices for TTC files
                        for subfont_idx in range(5):
                            try:
                                subfont_name = f"{font_name}-{subfont_idx}"
                                pdfmetrics.registerFont(TTFont(subfont_name, font_path, subfontIndex=subfont_idx))
                                registered_fonts.append(subfont_name)
                                print(f"✅ Registered: {subfont_name} from {font_path}")
                                break
                            except Exception as e:
                                if subfont_idx == 0:
                                    print(f"❌ TTC registration failed for {font_name}: {e}")
                                continue
                    else:
                        # Regular TTF file
                        pdfmetrics.registerFont(TTFont(font_name, font_path))
                        registered_fonts.append(font_name)
                        print(f"✅ Registered: {font_name} from {font_path}")
                        break
                        
                except Exception as e:
                    print(f"❌ Failed to register {font_name} from {font_path}: {e}")
                    continue
    
    # Also register our Noto fonts as fallback
    fonts_dir = Path.home() / 'Library' / 'Fonts'
    noto_fonts = [
        ('NotoDevanagari-Regular', 'NotoSansDevanagari-Regular.ttf'),
        ('NotoDevanagari-Bold', 'NotoSansDevanagari-Bold.ttf'),
    ]
    
    for font_name, font_file in noto_fonts:
        font_path = str(fonts_dir / font_file)
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                registered_fonts.append(font_name)
                print(f"✅ Registered: {font_name} (Noto fallback)")
            except Exception as e:
                print(f"❌ Failed to register {font_name}: {e}")
    
    return registered_fonts

def create_comprehensive_test():
    """Create a comprehensive test of all available Devanagari fonts"""
    
    # Register fonts
    available_fonts = try_register_system_devanagari_fonts()
    
    if not available_fonts:
        print("❌ No Devanagari fonts could be registered!")
        return
    
    print(f"\n✅ Successfully registered {len(available_fonts)} fonts")
    
    # Test texts with increasing complexity
    test_cases = [
        {
            'title': 'Simple Characters',
            'text': 'प र श म',
            'description': 'Individual characters (should always work)'
        },
        {
            'title': 'Basic Conjuncts', 
            'text': 'प्र श्र',
            'description': 'Common conjuncts (प्र = प + ् + र)'
        },
        {
            'title': 'Full Phrase',
            'text': 'प्रिय श्री शर्मा',
            'description': 'Complete phrase with conjuncts'
        },
        {
            'title': 'Complex Text',
            'text': 'न्याय और सुरक्षा के राज्य सचिव',
            'description': 'Government terminology with multiple conjuncts'
        }
    ]
    
    # Create PDF
    c = canvas.Canvas('comprehensive_devanagari_test.pdf', pagesize=letter)
    
    y_position = 750
    
    # Title
    c.setFont("Helvetica", 16)
    c.drawString(50, y_position, "Comprehensive Devanagari Font Test")
    y_position -= 30
    
    c.setFont("Helvetica", 10)
    c.drawString(50, y_position, f"Testing {len(available_fonts)} fonts with {len(test_cases)} test cases")
    y_position -= 30
    
    # Test each font with each test case
    for font_name in available_fonts:
        if y_position < 200:
            c.showPage()
            y_position = 750
        
        # Font header
        c.setFont("Helvetica", 12)
        c.drawString(50, y_position, f"Font: {font_name}")
        y_position -= 20
        
        # Test each case
        for test_case in test_cases:
            try:
                # Test case title
                c.setFont("Helvetica", 9)
                c.drawString(70, y_position, f"{test_case['title']}:")
                y_position -= 12
                
                # Test text
                c.setFont(font_name, 14)
                c.drawString(90, y_position, test_case['text'])
                y_position -= 15
                
                # Unicode info
                c.setFont("Helvetica", 7)
                unicode_info = ' '.join([f"U+{ord(char):04X}" for char in test_case['text'][:10]])
                c.drawString(90, y_position, unicode_info)
                y_position -= 15
                
            except Exception as e:
                c.setFont("Helvetica", 8)
                c.drawString(90, y_position, f"ERROR: {str(e)[:60]}")
                y_position -= 15
        
        y_position -= 10
    
    # Add analysis page
    c.showPage()
    y_position = 750
    
    c.setFont("Helvetica", 16)
    c.drawString(50, y_position, "Analysis & Recommendations")
    y_position -= 40
    
    analysis_text = [
        "WHAT TO LOOK FOR:",
        "",
        "1. CORRECT RENDERING:",
        "   प्रिय = प + ् + र + ि + य (conjunct प्र + vowel ि + य)",
        "   श्री = श + ् + र + ी (conjunct श्र + vowel ी)", 
        "",
        "2. INCORRECT RENDERING:",
        "   - Separated characters: प ् र ि य",
        "   - Wrong shapes: misformed conjuncts",
        "   - Missing diacriticals: missing vowel marks",
        "",
        "3. FONT EVALUATION:",
        "   - Best font = clearest conjuncts + proper vowel positioning",
        "   - Acceptable = readable even if not perfect",
        "   - Poor = separated characters or wrong shapes",
        "",
        "NEXT STEPS:",
        "1. Identify the best-rendering font from this test",
        "2. Update translator_api.py to prioritize that font",
        "3. Set up proper fallback chain",
        "4. Test with real document translation"
    ]
    
    for line in analysis_text:
        if line.startswith(('WHAT', 'NEXT')):
            c.setFont("Helvetica-Bold", 12)
        elif line.startswith(('1.', '2.', '3.', '4.')):
            c.setFont("Helvetica-Bold", 10)
        elif line.startswith('   -'):
            c.setFont("Helvetica", 9)
        else:
            c.setFont("Helvetica", 10)
        
        c.drawString(50, y_position, line)
        y_position -= 15
    
    c.save()
    
    print(f"\n✅ Created: comprehensive_devanagari_test.pdf")
    print("\nNEXT STEPS:")
    print("1. Open comprehensive_devanagari_test.pdf")
    print("2. Compare all fonts side-by-side")
    print("3. Identify which font renders conjuncts best")
    print("4. Report back which font looks correct")

def provide_manual_fix_instructions():
    """Provide instructions for manual font extraction if needed"""
    
    print("\n=== MANUAL FONT EXTRACTION (If Needed) ===")
    print()
    print("If system fonts show better rendering, you can extract them:")
    print()
    print("1. EXTRACT FONTS FROM TTC FILES:")
    print("   brew install fonttools  # Install font tools")
    print("   ttx -s /System/Library/Fonts/Kohinoor.ttc  # List subfonts")
    print("   ttx -t cmap -s /System/Library/Fonts/Kohinoor.ttc  # Extract specific subfont")
    print()
    print("2. OR USE ONLINE CONVERTERS:")
    print("   - Upload TTC files to online TTC-to-TTF converters")
    print("   - Download individual TTF files")
    print("   - Place in ~/Library/Fonts/")
    print()
    print("3. OR INSTALL ADDITIONAL FONTS:")
    print("   - Download Mangal.ttf (Windows Devanagari font)")
    print("   - Download other specialized Devanagari fonts")
    print("   - Install in ~/Library/Fonts/")

if __name__ == "__main__":
    print("=== FINAL DEVANAGARI RENDERING SOLUTION ===")
    print()
    
    create_comprehensive_test()
    provide_manual_fix_instructions() 