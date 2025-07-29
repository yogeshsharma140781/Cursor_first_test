#!/usr/bin/env python3

"""
PDF to Translated PDF Pipeline
Converts PDF to DOCX, translates it, then converts back to PDF with formatting preserved.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import fitz  # PyMuPDF
from docx import Document
import openai
from dotenv import load_dotenv
import glob
import requests
import json

# Import the translation function from the existing script
from translate_docx_openai import translate_docx

load_dotenv()

def find_office_command():
    """Find the correct LibreOffice command ('libreoffice' or 'soffice')"""
    for cmd in ["libreoffice", "soffice"]:
        if shutil.which(cmd):
            return cmd
    return None

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fitz
        print("✓ PyMuPDF (fitz) is available")
    except ImportError:
        print("✗ PyMuPDF not found. Install with: pip install PyMuPDF")
        return False
    
    try:
        from docx import Document
        print("✓ python-docx is available")
    except ImportError:
        print("✗ python-docx not found. Install with: pip install python-docx")
        return False
    
    # Check for LibreOffice (for PDF conversion)
    office_cmd = find_office_command()
    if office_cmd:
        try:
            result = subprocess.run([office_cmd, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✓ {office_cmd} is available")
            else:
                print(f"✗ {office_cmd} not working")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"✗ {office_cmd} not found. Install LibreOffice for PDF conversion.")
            return False
    else:
        print("✗ LibreOffice/soffice not found. Install LibreOffice for PDF conversion.")
        return False
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY not found in environment")
        return False
    else:
        print("✓ OpenAI API key is configured")
    
    return True

def pdf_to_docx_with_adobe(pdf_path, docx_path):
    """Convert PDF to DOCX using Adobe PDF Services SDK. Returns True on success."""
    try:
        print("Converting PDF to DOCX using Adobe PDF Services SDK...")
        # Load credentials
        cred_path = os.getenv("ADOBE_CREDENTIALS", "pdfservices-api-credentials.json")
        if not os.path.exists(cred_path):
            print(f"✗ Adobe credentials file not found: {cred_path}")
            return False
        
        with open(cred_path, "r") as f:
            adobe_creds = json.load(f)
        
        # Import Adobe PDF Services SDK
        from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
        from adobe.pdfservices.operation.pdf_services import PDFServices
        from adobe.pdfservices.operation.pdfjobs.jobs.export_pdf_job import ExportPDFJob
        from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_params import ExportPDFParams
        from adobe.pdfservices.operation.pdfjobs.params.export_pdf.export_pdf_target_format import ExportPDFTargetFormat
        from adobe.pdfservices.operation.pdfjobs.result.export_pdf_result import ExportPDFResult
        
        # Create credentials
        credentials = ServicePrincipalCredentials(
            client_id=adobe_creds['client_credentials']['client_id'],
            client_secret=adobe_creds['client_credentials']['client_secret']
        )
        
        # Initialize PDF Services
        pdf_services = PDFServices(credentials=credentials)
        
        # Read the PDF file
        with open(pdf_path, "rb") as f:
            input_stream = f.read()
        
        print("⏳ Uploading PDF to Adobe PDF Services...")
        input_asset = pdf_services.upload(input_stream=input_stream, mime_type="application/pdf")
        
        # Set up export parameters for DOCX
        export_params = ExportPDFParams(
            target_format=ExportPDFTargetFormat.DOCX
        )
        
        # Create and submit the job
        job = ExportPDFJob(input_asset=input_asset, export_pdf_params=export_params)
        print("⏳ Submitting export job...")
        polling_url = pdf_services.submit(job)
        
        print("⏳ Waiting for job to complete...")
        response = pdf_services.get_job_result(polling_url, ExportPDFResult)
        result_asset = response.get_result().get_asset()
        
        # Download the result
        stream_asset = pdf_services.get_content(result_asset)
        
        # Save the DOCX file
        with open(docx_path, "wb") as f:
            f.write(stream_asset.get_input_stream())
        
        print(f"✓ PDF converted to DOCX using Adobe SDK: {docx_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error using Adobe PDF Services SDK: {e}")
        return False

def pdf_to_docx_with_libreoffice(pdf_path, docx_path):
    """Convert PDF to DOCX using LibreOffice/soffice with formatting preservation"""
    office_cmd = find_office_command()
    if not office_cmd:
        print("✗ LibreOffice/soffice not found.")
        return False
    try:
        print(f"Converting PDF to DOCX using {office_cmd}...")
        
        # Create temporary directory for conversion
        temp_dir = tempfile.mkdtemp()
        
        # Use LibreOffice/soffice to convert PDF to DOCX
        cmd = [
            office_cmd,
            '--headless',
            '--convert-to', 'docx',
            '--outdir', temp_dir,
            pdf_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"LibreOffice conversion failed: {result.stderr}\nSTDOUT: {result.stdout}")
            return False
        
        # Find the converted file
        pdf_name = Path(pdf_path).stem
        converted_file = Path(temp_dir) / f"{pdf_name}.docx"
        
        if not converted_file.exists():
            # Try to find any .docx file in temp_dir
            docx_files = list(Path(temp_dir).glob('*.docx'))
            if docx_files:
                # Use the most recently created .docx file
                docx_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                converted_file = docx_files[0]
                print(f"Warning: Expected file {pdf_name}.docx not found. Using {converted_file.name} instead.")
            else:
                print(f"Converted file not found: {converted_file}")
                return False
        
        # Move to desired location
        shutil.move(str(converted_file), docx_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print(f"✓ PDF converted to DOCX: {docx_path}")
        return True
        
    except subprocess.TimeoutExpired as e:
        print(f"✗ LibreOffice conversion timed out after 5 minutes.\nSTDERR: {e.stderr}\nSTDOUT: {e.stdout}")
        return False
    except Exception as e:
        print(f"✗ Error converting PDF to DOCX: {e}")
        return False

def pdf_to_docx_with_fallback(pdf_path, docx_path):
    """Try Adobe PDF Services first, fallback to LibreOffice/soffice if needed."""
    if pdf_to_docx_with_adobe(pdf_path, docx_path):
        return True
    print("Falling back to LibreOffice/soffice for PDF to DOCX conversion...")
    return pdf_to_docx_with_libreoffice(pdf_path, docx_path)

def analyze_pdf_structure(pdf_path):
    """Analyze PDF to determine if it was likely created in Word"""
    try:
        doc = fitz.open(pdf_path)
        
        # Check for common Word-created PDF indicators
        word_indicators = 0
        total_pages = len(doc)
        
        for page_num in range(min(3, total_pages)):  # Check first 3 pages
            page = doc.load_page(page_num)
            
            # Get text with formatting info
            text_dict = page.get_text("dict")
            
            # Check for Word-specific formatting patterns
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            font_name = span.get("font", "").lower()
                            
                            # Word commonly uses these fonts
                            word_fonts = ['calibri', 'arial', 'times new roman', 'cambria', 'segoe ui']
                            if any(font in font_name for font in word_fonts):
                                word_indicators += 1
                            
                            # Check for Word-specific formatting
                            flags = span.get("flags", 0)
                            if flags & 1:  # Bold
                                word_indicators += 1
                            if flags & 2:  # Italic
                                word_indicators += 1
        
        doc.close()
        
        # If we found Word indicators, it's likely a Word-created PDF
        is_word_pdf = word_indicators > 5
        
        print(f"PDF Analysis:")
        print(f"  - Total pages: {total_pages}")
        print(f"  - Word indicators found: {word_indicators}")
        print(f"  - Likely created in Word: {'Yes' if is_word_pdf else 'No'}")
        
        return is_word_pdf
        
    except Exception as e:
        print(f"Error analyzing PDF: {e}")
        return True  # Assume it's a Word PDF if analysis fails

def docx_to_pdf_with_adobe(docx_path, pdf_path):
    """Convert DOCX to PDF using Adobe PDF Services SDK. Returns True on success."""
    try:
        print("Converting DOCX to PDF using Adobe PDF Services SDK...")
        cred_path = os.getenv("ADOBE_CREDENTIALS", "pdfservices-api-credentials.json")
        if not os.path.exists(cred_path):
            print(f"✗ Adobe credentials file not found: {cred_path}")
            return False
        with open(cred_path, "r") as f:
            adobe_creds = json.load(f)
        from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
        from adobe.pdfservices.operation.pdf_services import PDFServices
        from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
        from adobe.pdfservices.operation.pdfjobs.jobs.create_pdf_job import CreatePDFJob
        from adobe.pdfservices.operation.pdfjobs.result.create_pdf_result import CreatePDFResult
        # Create credentials
        credentials = ServicePrincipalCredentials(
            client_id=adobe_creds['client_credentials']['client_id'],
            client_secret=adobe_creds['client_credentials']['client_secret']
        )
        pdf_services = PDFServices(credentials=credentials)
        with open(docx_path, "rb") as f:
            input_stream = f.read()
        input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.DOCX)
        create_pdf_job = CreatePDFJob(input_asset)
        print("⏳ Submitting create PDF job...")
        polling_url = pdf_services.submit(create_pdf_job)
        print("⏳ Waiting for job to complete...")
        response = pdf_services.get_job_result(polling_url, CreatePDFResult)
        result_asset = response.get_result().get_asset()
        stream_asset = pdf_services.get_content(result_asset)
        with open(pdf_path, "wb") as f:
            f.write(stream_asset.get_input_stream())
        print(f"✓ DOCX converted to PDF using Adobe SDK: {pdf_path}")
        return True
    except Exception as e:
        print(f"✗ Error using Adobe PDF Services SDK for DOCX to PDF: {e}")
        return False

def pdf_to_translated_pdf(input_pdf, output_pdf, target_lang="English"):
    """Complete pipeline: PDF → DOCX → Translation → PDF"""
    
    print("=" * 60)
    print("PDF TO TRANSLATED PDF PIPELINE")
    print("=" * 60)
    
    # Step 1: Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        print("✗ Dependencies check failed. Please install missing components.")
        return False
    
    # Step 2: Analyze PDF
    print(f"\n2. Analyzing PDF: {input_pdf}")
    if not os.path.exists(input_pdf):
        print(f"✗ Input PDF not found: {input_pdf}")
        return False
    
    is_word_pdf = analyze_pdf_structure(input_pdf)
    
    # Step 3: Convert PDF to DOCX
    print(f"\n3. Converting PDF to DOCX...")
    temp_docx = f"{Path(input_pdf).stem}_temp.docx"
    
    if not pdf_to_docx_with_fallback(input_pdf, temp_docx):
        print("✗ PDF to DOCX conversion failed")
        return False
    
    # Step 4: Translate DOCX
    print(f"\n4. Translating DOCX...")
    translated_docx = f"{Path(input_pdf).stem}_translated.docx"
    
    try:
        translate_docx(temp_docx, translated_docx, target_lang)
        print("✓ Translation completed")
    except Exception as e:
        print(f"✗ Translation failed: {e}")
        # Cleanup temp file
        if os.path.exists(temp_docx):
            os.remove(temp_docx)
        return False
    
    # Step 5: Convert translated DOCX to PDF
    print("\n5. Converting translated DOCX to PDF...")
    if not docx_to_pdf_with_adobe(translated_docx, output_pdf):
        print("✗ DOCX to PDF conversion failed")
        # Cleanup temp files
        if os.path.exists(temp_docx):
            os.remove(temp_docx)
        if os.path.exists(translated_docx):
            os.remove(translated_docx)
        return False
    
    # Step 6: Cleanup temporary files
    print(f"\n6. Cleaning up temporary files...")
    if os.path.exists(temp_docx):
        os.remove(temp_docx)
        print("✓ Removed temporary DOCX file")
    
    # Keep the translated DOCX file for reference
    print(f"✓ Kept translated DOCX: {translated_docx}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Input PDF: {input_pdf}")
    print(f"Translated PDF: {output_pdf}")
    print(f"Translated DOCX: {translated_docx}")
    print("=" * 60)
    
    return True

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) != 3:
        print("Usage: python3 pdf_to_translated_pdf.py <input_pdf> <output_pdf>")
        print("\nExample:")
        print("  python3 pdf_to_translated_pdf.py sample3.pdf sample3_translated.pdf")
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    output_pdf = sys.argv[2]
    
    # Validate input file
    if not input_pdf.lower().endswith('.pdf'):
        print("Error: Input file must be a PDF")
        sys.exit(1)
    
    if not os.path.exists(input_pdf):
        print(f"Error: Input PDF file not found: {input_pdf}")
        sys.exit(1)
    
    # Run the pipeline
    success = pdf_to_translated_pdf(input_pdf, output_pdf)
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        print(f"Check the output files:")
        print(f"  - Translated PDF: {output_pdf}")
        print(f"  - Translated DOCX: {Path(input_pdf).stem}_translated.docx")
    else:
        print("\n❌ Pipeline failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 