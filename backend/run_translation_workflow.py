#!/usr/bin/env python3
"""
Simple script to run the PDF translation workflow
Translates PDF text to English while preserving layout and images
"""

import os
import sys
from adobe_openai_complete_workflow import PDFTranslationWorkflow

def main():
    """Run the PDF translation workflow"""
    
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
    
    # Get input PDF from command line or use default
    if len(sys.argv) > 1:
        input_pdf = sys.argv[1]
    else:
        input_pdf = "sample2.pdf"
    
    if not os.path.exists(input_pdf):
        print(f"❌ Error: Input PDF not found: {input_pdf}")
        print("Available PDF files:")
        for file in os.listdir('.'):
            if file.endswith('.pdf'):
                print(f"  - {file}")
        return
    
    # Generate output filename
    base_name = os.path.splitext(input_pdf)[0]
    output_pdf = f"{base_name}_translated_to_english.pdf"
    
    print(f"🚀 Starting PDF Translation Workflow")
    print(f"📄 Input: {input_pdf}")
    print(f"📄 Output: {output_pdf}")
    print(f"🌐 Target Language: English")
    print()
    
    # Create workflow instance
    workflow = PDFTranslationWorkflow(adobe_credentials_path, openai_api_key)
    
    # Run complete workflow
    try:
        print("⏳ Processing... This may take a few minutes...")
        result_path = workflow.run_complete_workflow(input_pdf, output_pdf, "English")
        print()
        print(f"✅ Translation complete!")
        print(f"📄 Output saved to: {result_path}")
        print()
        print("🎉 Your PDF has been successfully translated to English!")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 