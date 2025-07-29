#!/usr/bin/env python3
"""
Use Production PDF Translator API
Translates converted.pdf using the production API service
"""

import os
import requests
import json
from pathlib import Path
from datetime import datetime
import sys

def translate_pdf_with_production_api(pdf_path: str, target_language: str = "English", api_url: str = "http://localhost:8000") -> str:
    """
    Translate PDF using the production API
    
    Args:
        pdf_path: Path to the PDF file to translate
        target_language: Target language for translation
        api_url: URL of the production API server
    
    Returns:
        Path to the translated PDF file
    """
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return None
    
    print(f"🚀 Using Production PDF Translator API")
    print(f"📄 Input PDF: {pdf_path}")
    print(f"🌍 Target Language: {target_language}")
    print(f"🔗 API URL: {api_url}")
    
    # Prepare the API endpoint
    endpoint = f"{api_url}/translate-pdf"
    
    # Prepare the files and data for the request
    with open(pdf_path, 'rb') as pdf_file:
        files = {'file': (os.path.basename(pdf_path), pdf_file, 'application/pdf')}
        data = {
            'source_lang': 'auto',
            'target_lang': target_language.lower()[:2]  # Convert to language code
        }
        
        print(f"📤 Uploading PDF to production API...")
        
        try:
            # Make the API request
            response = requests.post(endpoint, files=files, data=data, timeout=300)
            
            if response.status_code == 200:
                # Save the translated PDF
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"production_translated_{target_language}_{timestamp}.pdf"
                output_path = os.path.join("production_output", output_filename)
                
                # Create output directory if it doesn't exist
                os.makedirs("production_output", exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Translation completed successfully!")
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

def check_api_status(api_url: str = "http://localhost:8000") -> bool:
    """Check if the production API server is running"""
    try:
        response = requests.get(f"{api_url}/docs", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main function"""
    print("🔧 Production PDF Translator API Client")
    print("=" * 50)
    # Configuration
    pdf_path = "converted.pdf"
    target_language = "English"  # Default
    api_url = "http://localhost:8000"  # Default API URL
    # Check for command line arguments
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    if len(sys.argv) > 2:
        target_language = sys.argv[2]
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
        print(f"  python translator_api.py")
        return
    print(f"✅ Production API server is running")
    # Available target languages mapping
    language_mapping = {
        "English": "en",
        "Hindi": "hi", 
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Arabic": "ar",
        "Dutch": "nl",
        "Polish": "pl",
        "Turkish": "tr",
        "Ukrainian": "uk",
        "Vietnamese": "vi"
    }
    # Print the actual target language
    print(f"🌍 Target Language: {target_language}")
    # Translate the PDF
    result = translate_pdf_with_production_api(pdf_path, target_language, api_url)
    if result:
        print()
        print(f"🎉 Success! Production translation completed")
        print(f"📁 Output file: {result}")
        print(f"📊 File sizes:")
        print(f"   Original: {os.path.getsize(pdf_path)/1024:.1f} KB")
        print(f"   Translated: {os.path.getsize(result)/1024:.1f} KB")
    else:
        print(f"❌ Translation failed.")

if __name__ == "__main__":
    main() 