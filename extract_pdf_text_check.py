import fitz
import sys
import re

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else 'sample_translated_hindi_final_fixed.pdf'

with fitz.open(PDF_PATH) as doc:
    text = "\n".join(page.get_text() for page in doc)

# Check for Devanagari script (Hindi)
devanagari_count = len(re.findall(r'[\u0900-\u097F]', text))
latin_count = len(re.findall(r'[A-Za-z]', text))

print(f"Devanagari chars: {devanagari_count}")
print(f"Latin chars: {latin_count}")

if devanagari_count > latin_count:
    print("✅ The PDF is mostly in Hindi (Devanagari script).")
else:
    print("❌ The PDF is mostly in English (Latin script) or not properly translated.")

# Print a sample of the extracted text
print("\n--- Extracted Text Sample ---\n")
print(text[:1000]) 