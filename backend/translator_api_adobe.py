#!/usr/bin/env python3
"""
Adobe PDF Services API - PDF Translation Service
Uses Adobe PDF Services for text extraction and PDF creation instead of ReportLab
"""

import os
import sys
import json
import tempfile
import shutil
import io
import base64
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
    from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
    from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
    from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
    from adobe.pdfservices.operation.io.stream_asset import StreamAsset
    from adobe.pdfservices.operation.pdf_services import PDFServices
    from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
    from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
    ADOBE_AVAILABLE = True
except ImportError:
    ADOBE_AVAILABLE = False
    print("⚠️ Adobe PDF Services SDK not available. Install with: pip install pdfservices-sdk")

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uvicorn

# OpenAI imports
import openai
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI(title="Adobe PDF Translation API", version="1.0")

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADOBE_CLIENT_ID = os.getenv("PDF_SERVICES_CLIENT_ID")
ADOBE_CLIENT_SECRET = os.getenv("PDF_SERVICES_CLIENT_SECRET")
ADOBE_PRIVATE_KEY_PATH = os.getenv("PDF_SERVICES_PRIVATE_KEY_PATH")

# Initialize OpenAI client
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("⚠️ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    client = None

def check_adobe_credentials():
    """Check if Adobe credentials are properly configured"""
    if not ADOBE_AVAILABLE:
        return False, "Adobe PDF Services SDK not installed"
    
    if not all([ADOBE_CLIENT_ID, ADOBE_CLIENT_SECRET, ADOBE_PRIVATE_KEY_PATH]):
        return False, "Adobe credentials not configured. Set ADOBE_CLIENT_ID, ADOBE_CLIENT_SECRET, and ADOBE_PRIVATE_KEY_PATH"
    
    if not os.path.exists(ADOBE_PRIVATE_KEY_PATH):
        return False, f"Adobe private key file not found: {ADOBE_PRIVATE_KEY_PATH}"
    
    return True, "Adobe credentials configured"

def get_adobe_credentials():
    """Get Adobe credentials for API operations"""
    if not ADOBE_AVAILABLE:
        raise Exception("Adobe PDF Services SDK not available")
    
    credentials = ServicePrincipalCredentials(
        client_id=ADOBE_CLIENT_ID,
        client_secret=ADOBE_CLIENT_SECRET
    )
    
    return credentials

def extract_text_with_adobe(pdf_path: str) -> Dict:
    """
    Extract text and structure from PDF using Adobe PDF Services
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing extracted text blocks and structure
    """
    try:
        # Read the PDF file
        with open(pdf_path, 'rb') as file:
            input_stream = file.read()
        
        # Get credentials
        credentials = get_adobe_credentials()
        
        # Create PDF Services instance
        pdf_services = PDFServices(credentials=credentials)
        
        # Upload the PDF as an asset
        input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)
        
        # Create parameters for extraction
        extract_pdf_params = ExtractPDFParams(
            elements_to_extract=[ExtractElementType.TEXT],
        )
        
        # Create and submit the job
        extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)
        location = pdf_services.submit(extract_pdf_job)
        pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)
        
        # Get the result content
        result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
        stream_asset: StreamAsset = pdf_services.get_content(result_asset)
        
        # Save the result to a temporary file
        temp_dir = tempfile.mkdtemp()
        result_path = os.path.join(temp_dir, "extract_result.zip")
        
        with open(result_path, "wb") as f:
            f.write(stream_asset.get_input_stream())
        
        # Extract and parse the JSON result
        import zipfile
        with zipfile.ZipFile(result_path, 'r') as zip_ref:
            # Find the JSON file
            json_files = [f for f in zip_ref.namelist() if f.endswith('.json')]
            if json_files:
                with zip_ref.open(json_files[0]) as json_file:
                    extract_data = json.load(json_file)
            else:
                raise Exception("No JSON file found in Adobe extraction result")
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        
        return extract_data
        
    except Exception as e:
        print(f"❌ Adobe extraction failed: {e}")
        raise

def parse_adobe_extraction(extract_data: Dict) -> List[Dict]:
    """
    Parse Adobe extraction result into structured text blocks
    
    Args:
        extract_data: Raw Adobe extraction data
        
    Returns:
        List of structured text blocks
    """
    text_blocks = []
    
    try:
        # Extract text elements from Adobe result
        if 'elements' in extract_data:
            for element in extract_data['elements']:
                if element.get('Path') and '//Document' in element['Path']:
                    # This is a text element
                    text_content = element.get('Text', '')
                    if text_content.strip():
                        # Get bounding box if available
                        bounds = element.get('Bounds', [])
                        if len(bounds) >= 4:
                            bbox = {
                                'x0': bounds[0],
                                'y0': bounds[1], 
                                'x1': bounds[2],
                                'y1': bounds[3],
                                'width': bounds[2] - bounds[0],
                                'height': bounds[3] - bounds[1]
                            }
                        else:
                            bbox = None
                        
                        # Get font information if available
                        font_info = element.get('Font', {})
                        font_name = font_info.get('name', 'Arial')
                        font_size = font_info.get('size', 12)
                        is_bold = font_info.get('weight', 'normal') == 'bold'
                        is_italic = font_info.get('style', 'normal') == 'italic'
                        
                        text_block = {
                            'text': text_content,
                            'bbox': bbox,
                            'font_name': font_name,
                            'font_size': font_size,
                            'is_bold': is_bold,
                            'is_italic': is_italic,
                            'page': 0  # Adobe doesn't always provide page info
                        }
                        
                        text_blocks.append(text_block)
        
        # If no structured elements found, try alternative parsing
        if not text_blocks and 'text' in extract_data:
            # Fallback to simple text extraction
            text_content = extract_data['text']
            text_block = {
                'text': text_content,
                'bbox': None,
                'font_name': 'Arial',
                'font_size': 12,
                'is_bold': False,
                'is_italic': False,
                'page': 0
            }
            text_blocks.append(text_block)
        
        return text_blocks
        
    except Exception as e:
        print(f"❌ Error parsing Adobe extraction: {e}")
        return []

def translate_text_with_openai(text: str, target_language: str = "English") -> str:
    """
    Translate text using OpenAI API
    
    Args:
        text: Text to translate
        target_language: Target language
        
    Returns:
        Translated text
    """
    if not client:
        return text  # Return original if OpenAI not available
    
    try:
        # Detect source language first
        detect_prompt = f"Detect the language of this text and respond with only the language name: {text[:200]}"
        
        detect_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": detect_prompt}],
            max_tokens=10,
            temperature=0
        )
        
        source_language = detect_response.choices[0].message.content.strip()
        
        # Translate the text
        translate_prompt = f"""
        Translate the following text from {source_language} to {target_language}.
        Preserve the original formatting, line breaks, and structure.
        Only return the translated text, nothing else.
        
        Text to translate:
        {text}
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": translate_prompt}],
            max_tokens=2000,
            temperature=0.3
        )
        
        translated_text = response.choices[0].message.content.strip()
        
        # Clean up any OpenAI error messages
        if "I apologize" in translated_text or "I'm sorry" in translated_text:
            return text  # Return original if translation failed
        
        return translated_text
        
    except Exception as e:
        print(f"❌ OpenAI translation failed: {e}")
        return text  # Return original text on error

def create_simple_pdf_output(text_blocks: List[Dict], output_path: str) -> str:
    """
    Create a simple text output (fallback when Adobe PDF creation is not available)
    
    Args:
        text_blocks: List of translated text blocks
        output_path: Path for output file
        
    Returns:
        Path to created file
    """
    try:
        # Create a simple text file with translated content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Adobe PDF Services Translation Result\n")
            f.write("=" * 50 + "\n\n")
            for i, block in enumerate(text_blocks):
                f.write(f"Block {i+1}:\n")
                f.write(block['text'] + "\n\n")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Simple output creation failed: {e}")
        return output_path

@app.post("/translate-pdf")
async def translate_pdf_endpoint(file: UploadFile = File(...), target_language: str = "English"):
    """Translate PDF using Adobe PDF Services"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Check Adobe availability
    adobe_ok, adobe_message = check_adobe_credentials()
    if not adobe_ok:
        raise HTTPException(status_code=500, detail=f"Adobe services not available: {adobe_message}")
    
    temp_input = None
    temp_output = None
    
    try:
        print(f"Processing PDF: {file.filename}")
        
        # Save uploaded file temporarily
        temp_input = tempfile.mktemp(suffix='.pdf')
        with open(temp_input, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        # Extract text using Adobe
        print("Extracting text with Adobe PDF Services...")
        extract_data = extract_text_with_adobe(temp_input)
        
        # Parse extraction result
        print("Parsing extracted content...")
        text_blocks = parse_adobe_extraction(extract_data)
        print(f"Extracted {len(text_blocks)} text blocks")
        
        # Translate text blocks
        print("Translating text blocks...")
        translated_blocks = []
        for i, block in enumerate(text_blocks):
            print(f"Translating block {i+1}/{len(text_blocks)}")
            translated_text = translate_text_with_openai(block['text'], target_language)
            
            translated_block = block.copy()
            translated_block['text'] = translated_text
            translated_blocks.append(translated_block)
        
        # Create translated output
        print("Creating translated output...")
        temp_output = tempfile.mktemp(suffix='.txt')
        output_path = create_simple_pdf_output(translated_blocks, temp_output)
        
        print(f"✅ Adobe-based translation completed successfully!")
        
        # Return the translated output
        return FileResponse(
            output_path,
            media_type='text/plain',
            filename=f"adobe_translated_{file.filename}.txt"
        )
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
        
    finally:
        # Cleanup
        if temp_input and os.path.exists(temp_input):
            os.remove(temp_input)
        if temp_output and os.path.exists(temp_output):
            os.remove(temp_output)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    adobe_ok, adobe_message = check_adobe_credentials()
    openai_ok = bool(OPENAI_API_KEY)
    
    return {
        "status": "healthy" if adobe_ok and openai_ok else "degraded",
        "version": "1.0-adobe",
        "adobe_services": adobe_ok,
        "openai_services": openai_ok,
        "adobe_message": adobe_message
    }

@app.get("/docs")
async def get_docs():
    """Get API documentation"""
    return {"message": "Adobe PDF Translation API - Use /docs for interactive documentation"}

if __name__ == "__main__":
    print("🚀 Starting Adobe PDF Translation API...")
    print("📋 Configuration:")
    print(f"   Adobe SDK: {'✅ Available' if ADOBE_AVAILABLE else '❌ Not available'}")
    print(f"   Adobe Credentials: {check_adobe_credentials()[1]}")
    print(f"   OpenAI API: {'✅ Configured' if OPENAI_API_KEY else '❌ Not configured'}")
    
    uvicorn.run(app, host="0.0.0.0", port=8005) 