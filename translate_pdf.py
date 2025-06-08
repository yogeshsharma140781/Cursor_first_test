import pdfplumber
import requests
import json

DEEPL_API_KEY = "1160354d-0a75-49d7-b2be-df5eaf692c1a:fx"
TARGET_LANGUAGE = "DE"  # e.g., German

def translate_text(text, target_lang):
    if not text.strip():
        return ""
    url = "https://api-free.deepl.com/v2/translate"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "auth_key": DEEPL_API_KEY,
        "text": text,
        "target_lang": target_lang,
    }
    try:
        response = requests.post(url, data=data, headers=headers)
        response.raise_for_status()
        return response.json()["translations"][0]["text"]
    except requests.exceptions.RequestException as e:
        print(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

def extract_and_translate_blocks(pdf_path):
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Processing PDF with {len(pdf.pages)} pages...")
            for page_num, page in enumerate(pdf.pages, start=1):
                print(f"Processing page {page_num}...")
                for block in page.extract_words(use_text_flow=True, keep_blank_chars=False):
                    translated = translate_text(block['text'], TARGET_LANGUAGE)
                    results.append({
                        "page": page_num,
                        "x": block['x0'],
                        "y": block['top'],
                        "w": block['x1'] - block['x0'],
                        "h": block['bottom'] - block['top'],
                        "original": block['text'],
                        "translated": translated
                    })
                print(f"Completed page {page_num} - processed {len(results)} blocks so far")
        return results
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return results

def main():
    pdf_path = "input.pdf"
    output_path = "translated_blocks.json"

    try:
        print("Starting PDF processing and translation...")
        translated_blocks = extract_and_translate_blocks(pdf_path)
        
        if translated_blocks:
            print(f"Writing {len(translated_blocks)} translated blocks to {output_path}...")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(translated_blocks, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved translations to {output_path}")
        else:
            print("No blocks were processed. Please check if the PDF contains extractable text.")
            
    except FileNotFoundError:
        print(f"Error: Could not find the PDF file '{pdf_path}'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 