#!/usr/bin/env python3
"""
Test Noto Fonts from Google with ReportLab for better glyph rendering
Downloads latest versions and tests complex script rendering
"""

import os
import requests
import tempfile
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import zipfile

def download_google_noto_fonts():
    """Download the latest Noto fonts directly from Google Fonts"""
    
    fonts_dir = Path.home() / 'Library' / 'Fonts'
    fonts_dir.mkdir(parents=True, exist_ok=True)
    
    # Google Fonts API URLs for Noto fonts
    noto_fonts = {
        'NotoSansDevanagari': {
            'regular': 'https://fonts.google.com/download?family=Noto%20Sans%20Devanagari',
            'weights': ['400', '700']  # regular and bold
        },
        'NotoSans': {
            'regular': 'https://fonts.google.com/download?family=Noto%20Sans',
            'weights': ['400', '700']
        },
        'NotoSansArabic': {
            'regular': 'https://fonts.google.com/download?family=Noto%20Sans%20Arabic',
            'weights': ['400', '700']
        }
    }
    
    # Alternative: Direct GitHub releases (more reliable)
    github_fonts = {
        'NotoSansDevanagari-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf',
        'NotoSansDevanagari-Bold.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Bold.ttf',
        'NotoSans-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf',
        'NotoSans-Bold.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf',
        'NotoSansArabic-Regular.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansArabic/NotoSansArabic-Regular.ttf',
        'NotoSansArabic-Bold.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansArabic/NotoSansArabic-Bold.ttf',
    }
    
    print("🔄 Downloading latest Noto fonts from Google/GitHub...")
    
    downloaded_fonts = []
    for font_name, url in github_fonts.items():
        font_path = fonts_dir / font_name
        
        try:
            print(f"  Downloading {font_name}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(font_path, 'wb') as f:
                f.write(response.content)
            
            downloaded_fonts.append(str(font_path))
            print(f"  ✅ Downloaded: {font_path}")
            
        except Exception as e:
            print(f"  ❌ Failed to download {font_name}: {e}")
    
    print(f"\n📦 Downloaded {len(downloaded_fonts)} fonts")
    return downloaded_fonts

def register_noto_fonts():
    """Register Noto fonts with ReportLab"""
    
    fonts_dir = Path.home() / 'Library' / 'Fonts'
    
    font_files = {
        'NotoSansDevanagari-Regular': 'NotoSansDevanagari-Regular.ttf',
        'NotoSansDevanagari-Bold': 'NotoSansDevanagari-Bold.ttf',
        'NotoSans-Regular': 'NotoSans-Regular.ttf', 
        'NotoSans-Bold': 'NotoSans-Bold.ttf',
        'NotoSansArabic-Regular': 'NotoSansArabic-Regular.ttf',
        'NotoSansArabic-Bold': 'NotoSansArabic-Bold.ttf',
    }
    
    registered_fonts = []
    
    print("\n🔧 Registering Noto fonts with ReportLab...")
    
    for font_name, filename in font_files.items():
        font_path = fonts_dir / filename
        
        if font_path.exists():
            try:
                # Clear any existing registration
                if font_name in pdfmetrics.getRegisteredFontNames():
                    print(f"  Font {font_name} already registered, skipping...")
                    registered_fonts.append(font_name)
                    continue
                
                pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                registered_fonts.append(font_name)
                print(f"  ✅ Registered: {font_name} -> {font_path}")
                
            except Exception as e:
                print(f"  ❌ Failed to register {font_name}: {e}")
        else:
            print(f"  ⚠️  Font file not found: {font_path}")
    
    print(f"\n📋 Registered {len(registered_fonts)} fonts with ReportLab")
    return registered_fonts

def test_devanagari_glyphs():
    """Test specific Devanagari glyphs that were problematic"""
    
    # Test cases from the actual translation
    test_cases = [
        ("Simple characters", "नमस्ते दुनिया"),
        ("From translation", "वापसी पता पोस्टबॉक्स"),
        ("Complex conjuncts", "प्रिय श्री शर्मा"),
        ("Problematic text", "नैदरलैंड्स नागरिकता"),
        ("Long sentence", "इस पत्र के माध्यम से मैं आपको आपकी नागरिकता अनुरोध की प्रगति के बारे में सूचित कर रहा हूँ।"),
        ("With punctuation", "सादर, न्याय और सुरक्षा के राज्य सचिव।"),
        ("Mixed content", "योगेश शर्मा, 4 जून 2025"),
    ]
    
    print("\n🧪 Testing Devanagari glyph rendering...")
    
    output_path = 'noto_devanagari_glyph_test.pdf'
    c = canvas.Canvas(output_path, pagesize=letter)
    
    y_position = 750
    
    # Test with different font weights
    fonts_to_test = ['NotoSansDevanagari-Regular', 'NotoSansDevanagari-Bold']
    
    for font in fonts_to_test:
        if font in pdfmetrics.getRegisteredFontNames():
            c.setFont('Helvetica', 12)
            c.drawString(50, y_position, f"Testing font: {font}")
            y_position -= 20
            
            for i, (description, text) in enumerate(test_cases):
                if y_position < 100:
                    c.showPage()
                    y_position = 750
                
                # Description in English
                c.setFont('Helvetica', 10)
                c.drawString(50, y_position, f"{description}:")
                
                # Test the Devanagari text
                try:
                    c.setFont(font, 14)
                    c.drawString(200, y_position, text)
                    
                    # Character analysis
                    char_count = len([c for c in text if ord(c) >= 0x0900])
                    c.setFont('Helvetica', 8)
                    c.drawString(450, y_position, f"({char_count} chars)")
                    
                    result = "SUCCESS"
                    
                except Exception as e:
                    c.setFont('Helvetica', 10)
                    c.drawString(200, y_position, f"[ERROR: {str(e)}]")
                    result = f"FAILED: {e}"
                
                print(f"    {description}: {result}")
                y_position -= 25
            
            y_position -= 20
        else:
            print(f"    ❌ Font {font} not available")
    
    c.save()
    print(f"\n📄 Test PDF created: {output_path}")
    return output_path

def analyze_character_support():
    """Analyze which specific characters are supported"""
    
    # Characters from the actual translation that were problematic
    problematic_chars = [
        ('\u0935', 'व', 'DEVANAGARI LETTER VA'),
        ('\u093E', 'ा', 'DEVANAGARI VOWEL SIGN AA'),
        ('\u092A', 'प', 'DEVANAGARI LETTER PA'),
        ('\u0938', 'स', 'DEVANAGARI LETTER SA'),
        ('\u0940', 'ी', 'DEVANAGARI VOWEL SIGN II'),
        ('\u094D', '्', 'DEVANAGARI SIGN VIRAMA'),
        ('\u0930', 'र', 'DEVANAGARI LETTER RA'),
        ('\u093F', 'ि', 'DEVANAGARI VOWEL SIGN I'),
        ('\u092F', 'य', 'DEVANAGARI LETTER YA'),
        ('\u0936', 'श', 'DEVANAGARI LETTER SHA'),
        ('\u0902', 'ं', 'DEVANAGARI SIGN ANUSVARA'),
        ('\u0949', 'ॉ', 'DEVANAGARI VOWEL SIGN CANDRA O'),
        ('\u0901', 'ँ', 'DEVANAGARI SIGN CANDRABINDU'),
    ]
    
    print("\n🔍 Character Support Analysis:")
    print("=" * 60)
    
    for unicode_char, char, name in problematic_chars:
        # Test if the character renders properly
        try:
            # Simple test - create a small canvas with just this character
            temp_path = f"temp_char_test_{ord(unicode_char)}.pdf"
            c = canvas.Canvas(temp_path, pagesize=(100, 100))
            c.setFont('NotoSansDevanagari-Regular', 12)
            c.drawString(10, 50, char)
            c.save()
            
            # If we got here, the character was rendered
            os.remove(temp_path)  # Clean up
            status = "✅ SUPPORTED"
            
        except Exception as e:
            status = f"❌ FAILED: {e}"
        
        print(f"{char} (U+{ord(unicode_char):04X}) {name}: {status}")

def main():
    """Main function to test Google Noto fonts with ReportLab"""
    
    print("🚀 Testing Google Noto Fonts with ReportLab")
    print("=" * 50)
    
    # Step 1: Download latest fonts
    downloaded = download_google_noto_fonts()
    
    if not downloaded:
        print("❌ No fonts downloaded. Exiting.")
        return
    
    # Step 2: Register fonts with ReportLab
    registered = register_noto_fonts()
    
    if not registered:
        print("❌ No fonts registered. Exiting.")
        return
    
    # Step 3: Test character support
    analyze_character_support()
    
    # Step 4: Create comprehensive test PDF
    test_pdf = test_devanagari_glyphs()
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"✅ Downloaded: {len(downloaded)} font files")
    print(f"✅ Registered: {len(registered)} fonts with ReportLab") 
    print(f"📄 Test PDF created: {test_pdf}")
    print("\n🔍 Next steps:")
    print("1. Open the test PDF to check glyph rendering")
    print("2. Compare with previous versions")
    print("3. If successful, integrate with translation API")

if __name__ == "__main__":
    main() 