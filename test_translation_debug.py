#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.append('backend')

from translator_api import AdvancedPDFLayoutParser

async def test_translation():
    """Test the translation function directly"""
    
    # Initialize the parser
    parser = AdvancedPDFLayoutParser(require_api_key=True)
    
    # Test texts that should be translated to Hindi
    test_texts = [
        "Return address P.O. Box 3 9560 AA TER APEL",
        "Directorate Regular Residence and",
        "Dutch nationality",
        "Case number",
        "born on 14 July 1981",
        "nationality: Indian",
        "Dear Mr. Sharma",
        "With this letter, I inform you about the progress of your nationality request.",
        "His Majesty the King has made a positive decision on your request for naturalization.",
        "If your residence permit expires earlier, you must apply for an extension of your residence permit.",
        "You will receive an invitation from the municipality where you live within six weeks.",
        "I request that you wait for the invitation to the naturalization ceremony from the municipality.",
        "Kind regards",
        "The Secretary of State for Justice and Security.",
        "This letter has been sent automatically, which is why there is no signature below.",
        "Page 1 of 1"
    ]
    
    print("Testing Hindi translation...")
    print("=" * 50)
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}:")
        print(f"Original: {text}")
        
        try:
            translated = await parser.translate_text_openai(text, 'hi')
            print(f"Translated: {translated}")
            
            # Check if it contains Devanagari characters
            devanagari_chars = sum(1 for c in translated if 0x0900 <= ord(c) <= 0x097F)
            if devanagari_chars > 0:
                print(f"✅ Contains {devanagari_chars} Devanagari characters")
            else:
                print(f"❌ No Devanagari characters found")
                
        except Exception as e:
            print(f"❌ Translation error: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(test_translation()) 