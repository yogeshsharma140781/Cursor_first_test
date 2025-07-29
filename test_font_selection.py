#!/usr/bin/env python3

import sys
import os

# Add the backend directory to the path
sys.path.append('backend')

from translator_api import get_best_font_for_language, LANGUAGE_SCRIPT_MAP, UNICODE_FONTS, register_unicode_fonts
from reportlab.pdfbase import pdfmetrics

def test_font_selection():
    """Test the font selection logic"""
    
    print("Testing font selection for Hindi...")
    print("=" * 50)
    
    # Register fonts first
    print("Registering Unicode fonts...")
    register_unicode_fonts()
    
    # Test the language script mapping
    print(f"LANGUAGE_SCRIPT_MAP['hi'] = {LANGUAGE_SCRIPT_MAP.get('hi', 'NOT_FOUND')}")
    
    # Test the UNICODE_FONTS configuration
    print(f"UNICODE_FONTS['devanagari'] = {UNICODE_FONTS.get('devanagari', 'NOT_FOUND')}")
    
    # Get registered fonts
    registered_fonts = pdfmetrics.getRegisteredFontNames()
    print(f"Registered fonts: {registered_fonts}")
    
    # Test Hindi text
    hindi_text = "नमस्ते दुनिया"
    print(f"\nTesting with Hindi text: '{hindi_text}'")
    
    # Test font selection for Hindi
    font_name = get_best_font_for_language('hi', hindi_text, False, False)
    print(f"Selected font for Hindi: {font_name}")
    
    # Test font selection for Hindi with bold
    font_name_bold = get_best_font_for_language('hi', hindi_text, True, False)
    print(f"Selected font for Hindi (bold): {font_name_bold}")
    
    # Test with English text to see the difference
    english_text = "Hello world"
    print(f"\nTesting with English text: '{english_text}'")
    
    font_name_en = get_best_font_for_language('en', english_text, False, False)
    print(f"Selected font for English: {font_name_en}")
    
    # Test with English text but Hindi target language
    font_name_en_hi = get_best_font_for_language('hi', english_text, False, False)
    print(f"Selected font for English text with Hindi target: {font_name_en_hi}")

if __name__ == "__main__":
    test_font_selection() 