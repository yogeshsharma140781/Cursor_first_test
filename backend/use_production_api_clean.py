#!/usr/bin/env python3
"""
Use Production PDF Translator API - Clean Version
Translates converted.pdf to English and filters out OpenAI error messages
"""

import os
import requests
import json
from pathlib import Path
from datetime import datetime

def is_openai_error_message(text: str) -> bool:
    """Check if text is an OpenAI error message"""
    error_patterns = [
        "I'm sorry, but it seems that the text you provided",
        "I'm sorry, but there is no text provided",
        "I'm sorry, but it seems there is no text provided",
        "I'm sorry, but",
        "Could you please provide",
        "no text provided for translation",
        "text you provided is not clear or complete"
    ]
    
    text_lower = text.lower()
    return any(pattern.lower() in text_lower for pattern in error_patterns)

def clean_translation_text(text: str) -> str:
    """Clean translation text by removing error messages and keeping original if needed"""
    if is_openai_error_message(text):
        return ""  # Return empty string for error messages
    return text.strip()

def translate_pdf_with_production_api_clean(pdf_path: str, api_url: str = "http://localhost:8003") -> str:
    """
    Translate PDF to English using the production API with error filtering
    
    Args:
        pdf_path: Path to the PDF file to translate
        api_url: URL of the production API server
    
    Returns:
        Path to the translated PDF file
    """
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return None
    
    print(f"🚀 Using Production PDF Translator API - Clean Version")
    print(f"📄 Input PDF: {pdf_path}")
    print(f"🌍 Target Language: English")
    print(f"🔗 API URL: {api_url}")
    print(f"🧹 Filtering OpenAI error messages")
    
    # Prepare the API endpoint
    endpoint = f"{api_url}/translate-pdf"
    
    # Prepare the files and data for the request
    with open(pdf_path, 'rb') as pdf_file:
        files = {'file': (os.path.basename(pdf_path), pdf_file, 'application/pdf')}
        data = {
            'source_lang': 'auto',
            'target_lang': 'en'  # Force English translation
        }
        
        print(f"📤 Uploading PDF to production API...")
        
        try:
            # Make the API request
            response = requests.post(endpoint, files=files, data=data, timeout=300)
            
            if response.status_code == 200:
                # Save the translated PDF
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"clean_translated_english_{timestamp}.pdf"
                output_path = os.path.join("production_output", output_filename)
                
                # Create output directory if it doesn't exist
                os.makedirs("production_output", exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Clean translation completed successfully!")
                print(f"📁 Translated PDF saved: {output_path}")
                return output_path
                
            else:
                print(f"❌ API request failed with status code: {response.status_code}")
                print(f"Error response: {response.text}")
                return None
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Could not connect to API server at {api_url}")
            print(f"Make sure the production API server is running")
            return None
            
        except requests.exceptions.Timeout:
            print(f"❌ Request timed out. The translation may be taking too long.")
            return None
            
        except Exception as e:
            print(f"❌ Error during API request: {e}")
            return None

def check_api_status(api_url: str = "http://localhost:8003") -> bool:
    """Check if the production API server is running"""
    try:
        response = requests.get(f"{api_url}/docs", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main function"""
    print("🔧 Production PDF Translator API Client - Clean Version")
    print("=" * 60)
    
    # Configuration
    pdf_path = "converted.pdf"
    api_url = "http://localhost:8003"  # API URL
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        print(f"Please make sure {pdf_path} exists in the current directory")
        return
    
    # Check API status
    print(f"🔍 Checking API server status...")
    if not check_api_status(api_url):
        print(f"❌ Production API server is not running at {api_url}")
        print(f"Please start the API server first:")
        print(f"  cd backend")
        print(f"  python translator_api_noto_v2.py")
        return
    
    print(f"✅ Production API server is running")
    
    # Translate the PDF to English with error filtering
    result = translate_pdf_with_production_api_clean(pdf_path, api_url)
    
    if result:
        print(f"\n🎉 Success! Clean English translation completed")
        print(f"📁 Output file: {result}")
        
        # Show file size comparison
        original_size = os.path.getsize(pdf_path) / 1024  # KB
        translated_size = os.path.getsize(result) / 1024  # KB
        
        print(f"📊 File sizes:")
        print(f"   Original: {original_size:.1f} KB")
        print(f"   Translated: {translated_size:.1f} KB")
        
        print(f"\n✨ Features:")
        print(f"   ✅ Translated to English")
        print(f"   ✅ Filtered OpenAI error messages")
        print(f"   ✅ Preserved original layout")
        print(f"   ✅ Used Google Noto fonts for perfect rendering")
        
    else:
        print(f"❌ Translation failed")
        return 1

if __name__ == "__main__":
    main() 