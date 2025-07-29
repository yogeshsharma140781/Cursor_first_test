#!/usr/bin/env python3
"""
Enhanced Devanagari font support with better character handling
This script addresses complex script rendering issues in ReportLab
"""

import os
import requests
from pathlib import Path
import unicodedata
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def download_better_devanagari_fonts():
    """Download fonts with better Devanagari support"""
    fonts_dir = Path.home() / 'Library' / 'Fonts'
    fonts_dir.mkdir(parents=True, exist_ok=True)
    
    # Better Devanagari fonts with comprehensive character support
    better_fonts = [
        {
            'name': 'Mangal Regular',
            'filename': 'mangal.ttf',
            'url': 'https://github.com/microsoft/cascadia-code/releases/download/v2111.01/CascadiaCode-2111.01.zip'
        },
        # Alternative: Download a comprehensive Devanagari font
        {
            'name': 'Siddhanta Regular',
            'filename': 'Siddhanta-Regular.ttf',
            'url': 'https://github.com/silnrsi/font-siddhanta/releases/download/v2.000/Siddhanta-v2.000.zip'
        }
    ]
    
    # For now, let's use system fonts that are more reliable
    print("Checking for better system Devanagari fonts...")
    
    # macOS system fonts that have good Devanagari support
    system_fonts = [
        '/System/Library/Fonts/Supplemental/DevanagariSangamMN.ttc',
        '/System/Library/Fonts/Helvetica.ttc',
        '/Library/Fonts/Arial Unicode MS.ttf',
        '/System/Library/Fonts/Apple Color Emoji.ttc'
    ]
    
    registered_fonts = []
    for font_path in system_fonts:
        if os.path.exists(font_path):
            try:
                font_name = f"System{os.path.basename(font_path).split('.')[0]}"
                if font_name not in pdfmetrics.getRegisteredFontNames():
                    if font_path.endswith('.ttc'):
                        # Try multiple subfonts for TTC files
                        for subfont_index in range(4):
                            try:
                                subfont_name = f"{font_name}-{subfont_index}"
                                pdfmetrics.registerFont(TTFont(subfont_name, font_path, subfontIndex=subfont_index))
                                registered_fonts.append(subfont_name)
                                print(f"Registered system font: {subfont_name}")
                                break
                            except:
                                continue
                    else:
                        pdfmetrics.registerFont(TTFont(font_name, font_path))
                        registered_fonts.append(font_name)
                        print(f"Registered system font: {font_name}")
            except Exception as e:
                print(f"Failed to register {font_path}: {e}")
    
    return registered_fonts

def preprocess_devanagari_text(text):
    """Preprocess Devanagari text to handle complex characters better"""
    if not text:
        return text
    
    # Normalize the text to decomposed form for better rendering
    normalized = unicodedata.normalize('NFD', text)
    
    # Remove problematic zero-width characters that might cause issues
    cleaned = ''.join(char for char in normalized if unicodedata.category(char) not in ['Mn', 'Cf'])
    
    # Convert back to composed form
    result = unicodedata.normalize('NFC', cleaned)
    
    return result

def get_best_devanagari_font():
    """Get the best available Devanagari font"""
    registered_fonts = pdfmetrics.getRegisteredFontNames()
    
    # Priority order for Devanagari fonts
    font_preferences = [
        'SystemDevanagariSangamMN-0',
        'SystemDevanagariSangamMN-1', 
        'SystemDevanagariSangamMN-2',
        'NotoSansDevanagari-Regular',
        'NotoSansDevanagari-Bold',
        'NotoSans-Regular',
        'Helvetica'
    ]
    
    for font in font_preferences:
        if font in registered_fonts:
            return font
    
    return 'Helvetica'

def test_devanagari_rendering():
    """Test Devanagari rendering with different approaches"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    # Download and register better fonts
    better_fonts = download_better_devanagari_fonts()
    
    # Test text samples
    test_samples = [
        'नमस्ते',
        'वापसी पता',
        'प्रिय श्री शर्मा',
        'नैदरलैंड्स नागरिकता',
        'इस पत्र के माध्यम से मैं आपको सूचित कर रहा हूँ।'
    ]
    
    output_path = 'improved_devanagari_test.pdf'
    c = canvas.Canvas(output_path, pagesize=letter)
    
    y_position = 750
    
    # Test with different fonts and preprocessing
    for i, text in enumerate(test_samples):
        if y_position < 100:
            c.showPage()
            y_position = 750
        
        # Original text
        c.setFont('Helvetica', 10)
        c.drawString(50, y_position, f"Original: {text}")
        
        # Preprocessed text
        preprocessed = preprocess_devanagari_text(text)
        c.drawString(50, y_position - 15, f"Preprocessed: {preprocessed}")
        
        # Try with best font
        best_font = get_best_devanagari_font()
        try:
            c.setFont(best_font, 14)
            c.drawString(50, y_position - 35, preprocessed)
            c.setFont('Helvetica', 8)
            c.drawString(300, y_position - 35, f"Font: {best_font}")
        except Exception as e:
            c.setFont('Helvetica', 10)
            c.drawString(50, y_position - 35, f"[ERROR: {e}]")
        
        y_position -= 60
    
    c.save()
    print(f"Improved test PDF created: {output_path}")
    return output_path

def main():
    """Main function to test and fix Devanagari rendering"""
    print("Fixing Devanagari Rendering Issues")
    print("=" * 40)
    
    # Test the current setup
    test_pdf = test_devanagari_rendering()
    
    print("\nRecommendations:")
    print("1. Check the generated PDF for character rendering")
    print("2. If issues persist, ReportLab may have fundamental limitations with complex scripts")
    print("3. Consider alternative PDF libraries like WeasyPrint or xhtml2pdf for better Unicode support")
    
    return test_pdf

if __name__ == "__main__":
    main() 