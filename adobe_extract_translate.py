import os
import json
import zipfile
from datetime import datetime

from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult


def extract_with_adobe(pdf_path, output_zip):
    # Load credentials from environment or json
    cred_path = os.getenv("ADOBE_CREDENTIALS", "pdfservices-api-credentials.json")
    with open(cred_path, "r") as f:
        creds = json.load(f)
    credentials = ServicePrincipalCredentials(
        client_id=creds["client_credentials"]["client_id"],
        client_secret=creds["client_credentials"]["client_secret"]
    )
    pdf_services = PDFServices(credentials=credentials)
    with open(pdf_path, 'rb') as file:
        input_stream = file.read()
    input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)
    extract_pdf_params = ExtractPDFParams(elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES])
    extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)
    location = pdf_services.submit(extract_pdf_job)
    pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)
    result_asset = pdf_services_response.get_result().get_resource()
    stream_asset = pdf_services.get_content(result_asset)
    with open(output_zip, "wb") as f:
        f.write(stream_asset.get_input_stream())


def main():
    pdf_path = 'sample4.pdf'
    output_zip = 'sample4_extracted.zip'
    output_json = 'sample4_extracted.json'

    print(f"Extracting structured data from {pdf_path} using Adobe PDF Extract API...")
    extract_with_adobe(pdf_path, output_zip)

    # Unzip and find structuredData.json
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        for name in zip_ref.namelist():
            if name.endswith('structuredData.json'):
                zip_ref.extract(name, '.')
                json_path = os.path.join('.', name)
                break
        else:
            raise FileNotFoundError('structuredData.json not found in zip')

    # Load and display extracted data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("\n--- Extracted Data Preview ---\n")
    for page in data.get('elements', []):
        if page['Path'].startswith('table'):  # Table
            print(f"Table: {page['Path']}")
            for row in page.get('Rows', []):
                print("      | " + " | ".join(cell.get('Text', '') for cell in row.get('Cells', [])))
        elif page['Path'].startswith('text'):  # Text
            print(f"Text: {page.get('Text', '')}")
    print("\n--- End of Extracted Data Preview ---\n")

if __name__ == "__main__":
    main() 