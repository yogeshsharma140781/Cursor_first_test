#!/usr/bin/env python3
"""
Adobe PDF Services OCR Workflow
Convert scanned.pdf to editable PDF using Adobe PDF Services API
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)

try:
    from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
    from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
    from adobe.pdfservices.operation.pdf_services import PDFServices
    from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
    from adobe.pdfservices.operation.pdfjobs.jobs.ocr_pdf_job import OCRPDFJob
    from adobe.pdfservices.operation.pdfjobs.params.ocr_pdf.ocr_params import OCRParams
    from adobe.pdfservices.operation.pdfjobs.params.ocr_pdf.ocr_supported_locale import OCRSupportedLocale
    from adobe.pdfservices.operation.pdfjobs.params.ocr_pdf.ocr_supported_type import OCRSupportedType
    from adobe.pdfservices.operation.pdfjobs.result.ocr_pdf_result import OCRPDFResult
    from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
    from adobe.pdfservices.operation.io.stream_asset import StreamAsset
    ADOBE_SDK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Adobe PDF Services SDK not available: {e}")
    ADOBE_SDK_AVAILABLE = False

class AdobeOCRProcessor:
    """Adobe PDF Services OCR processor"""
    
    def __init__(self, credentials_path: str = "backend/pdfservices-api-credentials.json"):
        """Initialize Adobe OCR processor"""
        
        self.credentials = None
        self.pdf_services = None
        self.credentials_path = credentials_path
        
        self.setup_adobe_credentials()
    
    def setup_adobe_credentials(self):
        """Setup Adobe PDF Services credentials"""
        
        print("🔑 Setting up Adobe PDF Services credentials...")
        
        if not ADOBE_SDK_AVAILABLE:
            print("❌ Adobe PDF Services SDK not available")
            return False
        
        try:
            # Load credentials from file
            if os.path.exists(self.credentials_path):
                with open(self.credentials_path, 'r') as f:
                    creds_data = json.load(f)
                
                client_id = creds_data.get('client_credentials', {}).get('client_id')
                client_secret = creds_data.get('client_credentials', {}).get('client_secret')
                
                if client_id and client_secret:
                    self.credentials = ServicePrincipalCredentials(
                        client_id=client_id,
                        client_secret=client_secret
                    )
                    
                    # Create PDF Services instance
                    self.pdf_services = PDFServices(credentials=self.credentials)
                    print("✅ Adobe PDF Services initialized successfully")
                    return True
            
            print("❌ Adobe credentials not found")
            return False
            
        except Exception as e:
            print(f"❌ Error setting up Adobe credentials: {e}")
            return False
    
    def perform_ocr(self, input_pdf_path: str, output_pdf_path: str, locale: str = "nl-NL") -> bool:
        """Perform OCR on PDF using Adobe PDF Services"""
        
        if not self.pdf_services:
            print("❌ Adobe PDF Services not initialized")
            return False
        
        try:
            print(f"📄 Starting Adobe OCR on {input_pdf_path}...")
            print(f"🌍 Language setting: {locale}")
            
            # Check if input PDF exists
            if not os.path.exists(input_pdf_path):
                print(f"❌ Input PDF not found: {input_pdf_path}")
                return False
            
            # Upload PDF to Adobe
            print("📤 Uploading PDF to Adobe...")
            with open(input_pdf_path, 'rb') as file:
                input_stream = file.read()
            
            input_asset = self.pdf_services.upload(
                input_stream=input_stream,
                mime_type=PDFServicesMediaType.PDF
            )
            
            # Map locale string to Adobe locale enum
            locale_map = {
                "nl-NL": OCRSupportedLocale.NL_NL,
                "en-US": OCRSupportedLocale.EN_US,
                "de-DE": OCRSupportedLocale.DE_DE,
                "fr-FR": OCRSupportedLocale.FR_FR,
                "es-ES": OCRSupportedLocale.ES_ES,
                "it-IT": OCRSupportedLocale.IT_IT,
            }
            
            adobe_locale = locale_map.get(locale, OCRSupportedLocale.NL_NL)
            
            # Create OCR job parameters
            print(f"🔧 Setting up OCR parameters for Dutch (nl-NL)...")
            ocr_params = OCRParams(
                ocr_locale=adobe_locale,
                ocr_type=OCRSupportedType.SEARCHABLE_IMAGE_EXACT
            )
            
            # Create OCR job
            print("🚀 Creating OCR job...")
            ocr_pdf_job = OCRPDFJob(
                input_asset=input_asset,
                ocr_pdf_params=ocr_params
            )
            
            # Submit job
            print("📋 Submitting OCR job to Adobe...")
            location = self.pdf_services.submit(ocr_pdf_job)
            
            # Get result
            print("⏳ Waiting for OCR processing...")
            pdf_services_response = self.pdf_services.get_job_result(location, OCRPDFResult)
            result_asset = pdf_services_response.get_result().get_asset()
            
            # Download result
            print("📥 Downloading OCR result...")
            stream_asset = self.pdf_services.get_content(result_asset)
            
            # Save OCR result
            with open(output_pdf_path, "wb") as file:
                file.write(stream_asset.get_input_stream())
            
            # Check result
            if os.path.exists(output_pdf_path):
                file_size = os.path.getsize(output_pdf_path)
                print(f"✅ Adobe OCR completed successfully!")
                print(f"📄 Output file: {output_pdf_path}")
                print(f"📊 File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                return True
            else:
                print("❌ OCR result file not created")
                return False
                
        except Exception as e:
            print(f"❌ Adobe OCR error: {e}")
            return False
    
    def process_scanned_pdf(self, input_pdf: str = "scanned.pdf", output_pdf: str = "scanned_ADOBE_OCR_DUTCH.pdf", locale: str = "nl-NL"):
        """Complete Adobe OCR workflow"""
        
        print("🚀 ADOBE PDF SERVICES OCR WORKFLOW (DUTCH)")
        print("=" * 50)
        
        if not self.pdf_services:
            print("❌ Adobe PDF Services not initialized")
            return False
        
        # Perform OCR with specified language
        success = self.perform_ocr(input_pdf, output_pdf, locale)
        
        if success:
            print(f"\n🎉 Adobe OCR (Dutch) completed successfully!")
            print(f"📄 Input: {input_pdf}")
            print(f"📄 Output: {output_pdf}")
            print(f"🌍 Language: Dutch (nl-NL)")
            print(f"✅ The PDF is now searchable and editable with Dutch language optimization")
        else:
            print(f"\n❌ Adobe OCR (Dutch) failed")
        
        return success

def main():
    """Main execution function"""
    
    try:
        # Create Adobe OCR processor
        processor = AdobeOCRProcessor()
        
        # Process scanned PDF
        success = processor.process_scanned_pdf()
        
        if success:
            print("\n✅ Adobe OCR workflow (Dutch) completed successfully!")
        else:
            print("\n❌ Adobe OCR workflow (Dutch) failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 