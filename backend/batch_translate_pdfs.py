#!/usr/bin/env python3
"""
Batch PDF Translation Script
Translates multiple PDFs to English while preserving layout and images
"""

import os
import sys
import glob
from pathlib import Path
from adobe_openai_complete_workflow import PDFTranslationWorkflow

def find_pdf_files(directory=".", pattern="*.pdf"):
    """Find all PDF files in the directory"""
    pdf_files = []
    for file_path in glob.glob(os.path.join(directory, pattern)):
        # Skip already translated files
        if not file_path.endswith("_translated_to_english.pdf"):
            pdf_files.append(file_path)
    return sorted(pdf_files)

def main():
    """Run batch PDF translation"""
    
    # Configuration
    adobe_credentials_path = "pdfservices-api-credentials.json"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Check if credentials are available
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    if not os.path.exists(adobe_credentials_path):
        print(f"❌ Error: Adobe credentials file not found: {adobe_credentials_path}")
        return
    
    # Get input files
    if len(sys.argv) > 1:
        # Specific files provided
        input_files = sys.argv[1:]
        # Validate files exist
        valid_files = []
        for file_path in input_files:
            if os.path.exists(file_path) and file_path.endswith('.pdf'):
                valid_files.append(file_path)
            else:
                print(f"⚠️  Warning: {file_path} not found or not a PDF file")
        
        if not valid_files:
            print("❌ No valid PDF files found")
            return
    else:
        # Find all PDF files in current directory
        input_files = find_pdf_files()
        if not input_files:
            print("❌ No PDF files found in current directory")
            print("Available files:")
            for file in os.listdir('.'):
                if file.endswith('.pdf'):
                    print(f"  - {file}")
            return
    
    print(f"🚀 Starting Batch PDF Translation Workflow")
    print(f"📄 Found {len(input_files)} PDF file(s) to translate")
    print(f"🌐 Target Language: English")
    print()
    
    # Create workflow instance
    workflow = PDFTranslationWorkflow(adobe_credentials_path, openai_api_key)
    
    # Process each file
    successful_translations = []
    failed_translations = []
    
    for i, input_pdf in enumerate(input_files, 1):
        print(f"📄 Processing {i}/{len(input_files)}: {os.path.basename(input_pdf)}")
        
        # Generate output filename
        base_name = os.path.splitext(input_pdf)[0]
        output_pdf = f"{base_name}_translated_to_english.pdf"
        
        try:
            print(f"⏳ Translating to English...")
            result_path = workflow.run_complete_workflow(input_pdf, output_pdf, "English")
            successful_translations.append((input_pdf, result_path))
            print(f"✅ Success: {os.path.basename(result_path)}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            failed_translations.append((input_pdf, str(e)))
        
        print()  # Add spacing between files
    
    # Summary
    print("=" * 50)
    print("📊 TRANSLATION SUMMARY")
    print("=" * 50)
    print(f"✅ Successful translations: {len(successful_translations)}")
    print(f"❌ Failed translations: {len(failed_translations)}")
    
    if successful_translations:
        print("\n✅ Successfully translated files:")
        for input_file, output_file in successful_translations:
            print(f"  📄 {os.path.basename(input_file)} → {os.path.basename(output_file)}")
    
    if failed_translations:
        print("\n❌ Failed translations:")
        for input_file, error in failed_translations:
            print(f"  📄 {os.path.basename(input_file)}: {error}")
    
    print("\n🎉 Batch translation complete!")

if __name__ == "__main__":
    main() 