#!/usr/bin/env python3
"""
Translate sample.pdf into all 14 supported languages
"""

import requests
import time
import os
from pathlib import Path

# Supported languages (14 total)
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'nl': 'Dutch', 
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'es': 'Spanish',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'vi': 'Vietnamese'
}

def wait_for_api():
    """Wait for the API to be ready"""
    print("🔄 Waiting for translator API to start...")
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get('https://cursor-first-test.onrender.com/', timeout=5)
            if response.status_code == 200:
                print("✅ Translator API is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < max_attempts - 1:
            print(f"   Attempt {attempt + 1}/{max_attempts}...")
            time.sleep(2)
    
    print("❌ API failed to start within expected time")
    return False

def translate_pdf(source_file, target_lang, output_dir):
    """Translate a PDF to a specific language"""
    try:
        with open(source_file, 'rb') as f:
            files = {'file': (os.path.basename(source_file), f, 'application/pdf')}
            data = {
                'source_lang': 'auto',
                'target_lang': target_lang
            }
            
            print(f"   📤 Translating to {SUPPORTED_LANGUAGES[target_lang]}...")
            response = requests.post(
                'https://cursor-first-test.onrender.com/translate-pdf',
                files=files,
                data=data,
                timeout=120  # 2 minutes timeout
            )
            
            if response.status_code == 200:
                output_file = output_dir / f"sample_{target_lang}.pdf"
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print(f"   ✅ Saved: {output_file}")
                return True
            else:
                print(f"   ❌ Error: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def main():
    """Main function to translate sample.pdf to all supported languages"""
    print("🚀 Starting translation of sample.pdf to all 14 supported languages")
    print("=" * 60)
    
    # Check if sample.pdf exists
    sample_pdf = Path("sample.pdf")
    if not sample_pdf.exists():
        print(f"❌ sample.pdf not found at {sample_pdf.absolute()}")
        return
    
    print(f"📄 Source file: {sample_pdf.absolute()}")
    
    # Wait for API to be ready
    if not wait_for_api():
        return
    
    # Create output directory
    output_dir = Path("translated_samples")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Translate to each language
    print("\n🌍 Starting translations...")
    print("-" * 60)
    
    success_count = 0
    total_languages = len(SUPPORTED_LANGUAGES)
    
    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        print(f"\n[{success_count + 1}/{total_languages}] {lang_name} ({lang_code})")
        
        if translate_pdf(sample_pdf, lang_code, output_dir):
            success_count += 1
        else:
            print(f"   ⚠️  Failed to translate to {lang_name}")
        
        # Small delay between requests
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TRANSLATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful translations: {success_count}/{total_languages}")
    print(f"❌ Failed translations: {total_languages - success_count}")
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    if success_count > 0:
        print("\n📋 Generated files:")
        for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
            output_file = output_dir / f"sample_{lang_code}.pdf"
            if output_file.exists():
                file_size = output_file.stat().st_size / 1024  # KB
                print(f"   • {lang_name}: sample_{lang_code}.pdf ({file_size:.1f} KB)")
    
    print(f"\n🎉 Translation process completed!")

if __name__ == "__main__":
    main() 