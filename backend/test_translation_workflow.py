#!/usr/bin/env python3
"""
Test script for the PDF translation workflow
Verifies that all components are working correctly
"""

import os
import sys
from adobe_openai_complete_workflow import PDFTranslationWorkflow

def test_workflow_setup():
    """Test if the workflow can be initialized"""
    print("🧪 Testing workflow setup...")
    
    # Check credentials
    adobe_credentials_path = "pdfservices-api-credentials.json"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not set")
        return False
    
    if not os.path.exists(adobe_credentials_path):
        print("❌ Adobe credentials file not found")
        return False
    
    try:
        workflow = PDFTranslationWorkflow(adobe_credentials_path, openai_api_key)
        print("✅ Workflow initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize workflow: {e}")
        return False

def test_pdf_availability():
    """Test if sample PDFs are available"""
    print("\n🧪 Testing PDF availability...")
    
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf') and not f.endswith('_translated_to_english.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found for testing")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF file(s) for testing:")
    for pdf in pdf_files:
        print(f"  📄 {pdf}")
    
    return pdf_files[0] if pdf_files else None

def test_dependencies():
    """Test if all required dependencies are available"""
    print("\n🧪 Testing dependencies...")
    
    dependencies = [
        ('adobe.pdfservices', 'Adobe PDF Services SDK'),
        ('openai', 'OpenAI'),
        ('reportlab', 'ReportLab'),
        ('fitz', 'PyMuPDF'),
        ('PIL', 'Pillow')
    ]
    
    all_available = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} not available")
            all_available = False
    
    return all_available

def test_simple_translation():
    """Test a simple translation workflow"""
    print("\n🧪 Testing simple translation...")
    
    # Get test PDF
    test_pdf = test_pdf_availability()
    if not test_pdf:
        return False
    
    # Initialize workflow
    adobe_credentials_path = "pdfservices-api-credentials.json"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        workflow = PDFTranslationWorkflow(adobe_credentials_path, openai_api_key)
        
        # Test with a small section first
        print(f"📄 Testing with: {test_pdf}")
        
        # Run the complete workflow
        output_pdf = f"test_translation_output.pdf"
        result_path = workflow.run_complete_workflow(test_pdf, output_pdf, "English")
        
        if os.path.exists(result_path):
            print(f"✅ Translation test successful: {result_path}")
            return True
        else:
            print("❌ Translation test failed: output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Translation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 PDF Translation Workflow Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Workflow Setup", test_workflow_setup),
        ("PDF Availability", lambda: test_pdf_availability() is not None),
        ("Simple Translation", test_simple_translation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The workflow is ready to use.")
        print("\n📝 Next steps:")
        print("1. Run: python run_translation_workflow.py")
        print("2. Or run batch: python batch_translate_pdfs.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\n🔧 Common fixes:")
        print("1. Set OPENAI_API_KEY: export OPENAI_API_KEY='your-key'")
        print("2. Install dependencies: pip install -r requirements_adobe_openai.txt")
        print("3. Ensure pdfservices-api-credentials.json exists")

if __name__ == "__main__":
    main() 