import os
import sys
import json

try:
    from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
    from adobe.pdfservices.operation.pdf_services import PDFServices
    from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob, ExtractPDFParams
    from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
except ImportError as e:
    print('❌ ImportError:', e)
    sys.exit(1)

pdf_path = "sample2.pdf"
output_zip = "sample2_extracted_minimal.zip"

if not os.path.exists(pdf_path):
    print(f"❌ PDF file not found: {pdf_path}")
    sys.exit(1)

# Prepare credentials (read from JSON)
creds_path = "pdfservices-api-credentials.json"
if not os.path.exists(creds_path):
    print(f"❌ Credentials file not found: {creds_path}")
    sys.exit(1)

with open(creds_path, "r") as f:
    creds_json = json.load(f)

client_id = creds_json["client_credentials"]["client_id"]
client_secret = creds_json["client_credentials"]["client_secret"]

creds = ServicePrincipalCredentials(client_id, client_secret)
pdf_services = PDFServices(credentials=creds)

# Upload the PDF as an Asset
with open(pdf_path, "rb") as f:
    input_stream = f.read()

print("⏳ Uploading PDF to Adobe PDF Services...")
input_asset = pdf_services.upload(input_stream=input_stream, mime_type="application/pdf")

# Set extraction parameters (extract available elements)
extract_params = ExtractPDFParams(elements_to_extract=[
    ExtractElementType.TEXT,
    ExtractElementType.TABLES
])

# Create and submit the extract job
job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_params)
print("⏳ Submitting extract job...")
polling_url = pdf_services.submit(job)
print(f"⏳ Job submitted. Polling URL: {polling_url}")

# Poll for result
print("⏳ Waiting for job to complete...")
response = pdf_services.get_job_result(polling_url, ExtractPDFResult)

# Save the result ZIP
result_asset = response.get_result().get_resource()
stream_asset = pdf_services.get_content(result_asset)
with open(output_zip, "wb") as out_f:
    out_f.write(stream_asset.get_input_stream())
print(f"✅ Output saved as {output_zip}") 