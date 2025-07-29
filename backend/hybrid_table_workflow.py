#!/usr/bin/env python3
"""
Adobe-Only PDF Translation Workflow
- Uses Adobe PDF Services SDK for all extraction (text, images, tables)
- Reconstructs PDF using only Adobe data
- Optionally translates text using OpenAI
"""

import json
import os
from typing import Dict, Any

class AdobeOnlyWorkflow:
    def __init__(self, adobe_credentials_path: str, openai_api_key: str = None):
        self.adobe_credentials_path = adobe_credentials_path
        self.openai_api_key = openai_api_key

    def extract_with_adobe(self, pdf_path: str) -> Dict[str, Any]:
        from adobe_openai_complete_workflow import PDFTranslationWorkflow
        workflow = PDFTranslationWorkflow(self.adobe_credentials_path, self.openai_api_key)
        structured_data = workflow.extract_pdf_elements(pdf_path)
        return structured_data

    def translate_text(self, text: str, target_language: str = "English") -> str:
        if not self.openai_api_key:
            return text
        import openai
        client = openai.OpenAI(api_key=self.openai_api_key)
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional translator. Translate the following text to {target_language}. Preserve any formatting, numbers, and special characters. Return only the translated text."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def reconstruct_pdf(self, adobe_data: Dict[str, Any], output_path: str, translate: bool = False, target_language: str = "English") -> str:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        import os
        self.A4 = A4
        styles = getSampleStyleSheet()
        c = canvas.Canvas(output_path, pagesize=A4)
        page_height = self.A4[1]

        # Group elements by page
        elements_by_page = {}
        for element in adobe_data.get('elements', []):
            page_num = element.get('Page', 1)
            elements_by_page.setdefault(page_num, []).append(element)

        num_pages = max(elements_by_page.keys(), default=1)
        for page_num in range(1, num_pages + 1):
            page_elements = elements_by_page.get(page_num, [])
            # Sort by Y descending (top to bottom)
            page_elements.sort(key=lambda el: -el.get('Bounds', [0, 0, 0, 0])[3])
            for element in page_elements:
                if 'Text' in element and element['Text']:
                    self._draw_text_element(c, element, styles, translate, target_language)
                elif 'filePaths' in element:
                    self._draw_image_element(c, element)
            if page_num < num_pages:
                c.showPage()
        c.save()
        return output_path

    def _draw_text_element(self, canvas_obj, element, styles, translate, target_language):
        try:
            text = element.get('Text', '')
            if isinstance(text, dict):
                return
            elif isinstance(text, list):
                text = ' '.join(str(t) for t in text if str(t).strip())
            else:
                text = str(text).strip()
            if not text:
                return
            if translate:
                text = self.translate_text(text, target_language)
            bounds = element.get('Bounds', [])
            if len(bounds) != 4:
                return
            x, y, x1, y1 = bounds
            page_height = self.A4[1]
            y = page_height - y1
            font_size = element.get('FontSize', 12)
            font_name = element.get('Font', 'Helvetica')
            canvas_obj.setFont(font_name, font_size)
            text_width = canvas_obj.stringWidth(text, font_name, font_size)
            text_x = x + (x1 - x - text_width) / 2
            text_y = y + (y1 - y) / 2 + font_size / 3
            canvas_obj.drawString(text_x, text_y, text)
        except Exception as e:
            print(f"Error drawing text element: {e}")

    def _draw_image_element(self, canvas_obj, element):
        try:
            file_paths = element.get('filePaths', [])
            if not file_paths:
                return
            image_path = file_paths[0]
            if image_path.endswith(('.xlsx', '.xls', '.csv')):
                return
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return
            bounds = element.get('Bounds', [])
            if len(bounds) != 4:
                return
            x, y, x1, y1 = bounds
            page_height = self.A4[1]
            y = page_height - y1
            width = x1 - x
            height = y1 - y
            canvas_obj.drawImage(image_path, x, y, width, height)
        except Exception as e:
            print(f"Error drawing image element: {e}")

def main():
    adobe_credentials_path = "pdfservices-api-credentials.json"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    workflow = AdobeOnlyWorkflow(adobe_credentials_path, openai_api_key)
    input_pdf = "sample3.pdf"
    output_pdf = "sample3_adobe_translated.pdf"
    if not os.path.exists(input_pdf):
        print(f"Error: {input_pdf} not found")
        return
    adobe_data = workflow.extract_with_adobe(input_pdf)
    workflow.reconstruct_pdf(adobe_data, output_pdf, translate=False)
    print(f"Adobe-only workflow completed! Output: {output_pdf}")

if __name__ == "__main__":
    main() 