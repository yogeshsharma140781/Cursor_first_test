#!/usr/bin/env python3
"""
PDF Translation Workflow with OpenAI Vision
- Extract text blocks from converted.pdf
- For each block, use OpenAI Vision with scanned.png for context-aware cleaning and translation
- Reconstruct PDF with preserved layout
"""
import os
import json
import base64
from typing import List, Dict, Any
from openai import OpenAI
import fitz  # PyMuPDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime

class PDFTranslationWorkflow:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("❌ No OpenAI API key found in environment variables")
        self.client = OpenAI(api_key=self.api_key)
        print("✅ OpenAI client initialized successfully")
        self.setup_fonts()
    def setup_fonts(self):
        try:
            font_paths = [
                "/System/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Times.ttc",
                "/System/Library/Fonts/Helvetica.ttc"
            ]
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont('Arial', font_path))
                        break
                    except:
                        continue
            print("✅ Fonts setup completed")
        except Exception as e:
            print(f"⚠️  Font setup warning: {e}")
    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    def extract_text_blocks_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        print(f"📖 Extracting text blocks from {pdf_path}...")
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")
                for block in blocks["blocks"]:
                    if "lines" in block:
                        block_text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"] + " "
                        block_text = block_text.strip()
                        if block_text:
                            text_block = {
                                'page': page_num + 1,
                                'text': block_text,
                                'bbox': block["bbox"],
                                'font_size': block["lines"][0]["spans"][0]["size"] if block["lines"] and block["lines"][0]["spans"] else 11,
                                'font_name': block["lines"][0]["spans"][0]["font"] if block["lines"] and block["lines"][0]["spans"] else "Arial",
                                'is_bold': block["lines"][0]["spans"][0]["flags"] & 2**4 != 0 if block["lines"] and block["lines"][0]["spans"] else False,
                                'is_italic': block["lines"][0]["spans"][0]["flags"] & 2**1 != 0 if block["lines"] and block["lines"][0]["spans"] else False,
                                'cleaned_text': None,
                                'translated_text': None,
                                'confidence': None
                            }
                            text_blocks.append(text_block)
            doc.close()
            print(f"✅ Extracted {len(text_blocks)} text blocks")
            return text_blocks
        except Exception as e:
            print(f"❌ Error extracting text blocks: {e}")
            return []
    def clean_and_translate_with_vision(self, block: Dict[str, Any], image_b64: str, target_language: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in document restoration and translation. Given a text block from a PDF and the original scanned image, your job is to clean up any garbled or OCR-corrupted text, and then translate it to the target language. Use the image as the ground truth for accuracy. Return only the cleaned and translated text, no explanations."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Here is a text block from a PDF (may be garbled):\n\n{block['text']}\n\nPlease use the attached scanned image as reference. Clean up the text and translate it to {target_language}. Return only the cleaned and translated text."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.2,
                max_tokens=800
            )
            result_text = response.choices[0].message.content.strip()
            return {
                'cleaned_text': result_text,
                'translated_text': result_text,
                'success': True
            }
        except Exception as e:
            print(f"❌ Error with OpenAI Vision: {e}")
            return {
                'cleaned_text': block['text'],
                'translated_text': block['text'],
                'success': False
            }
    def reconstruct_pdf_with_translations(self, text_blocks: List[Dict[str, Any]], output_path: str) -> bool:
        print(f"📝 Reconstructing PDF: {output_path}")
        try:
            pdf_doc = SimpleDocTemplate(output_path, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            pages = {}
            for block in text_blocks:
                page_num = block['page']
                if page_num not in pages:
                    pages[page_num] = []
                pages[page_num].append(block)
            for page_num in sorted(pages.keys()):
                page_blocks = pages[page_num]
                page_blocks.sort(key=lambda x: x['bbox'][1])
                for block in page_blocks:
                    text_to_use = block.get('translated_text') or block.get('cleaned_text') or block['text']
                    if text_to_use:
                        font_size = block.get('font_size', 11)
                        is_bold = block.get('is_bold', False)
                        is_italic = block.get('is_italic', False)
                        style_name = f"Custom_{font_size}_{is_bold}_{is_italic}"
                        if style_name not in styles:
                            style = ParagraphStyle(
                                style_name,
                                parent=styles['Normal'],
                                fontSize=font_size,
                                spaceAfter=6,
                                alignment=TA_LEFT,
                                fontName='Arial'
                            )
                            styles.add(style)
                        else:
                            style = styles[style_name]
                        p = Paragraph(text_to_use, style)
                        story.append(p)
                        story.append(Spacer(1, 3))
                if page_num < max(pages.keys()):
                    story.append(Spacer(1, 20))
            pdf_doc.build(story)
            print(f"✅ Reconstructed PDF saved: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error reconstructing PDF: {e}")
            import traceback
            traceback.print_exc()
            return False
    def process_workflow(self, pdf_path: str, image_path: str, target_language: str = "English", output_dir: str = "pdf_translation_output") -> Dict[str, Any]:
        print("🚀 Starting PDF Translation Workflow (PDF + Vision)")
        print("=" * 60)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Step 1: Extract text blocks
        text_blocks = self.extract_text_blocks_from_pdf(pdf_path)
        if not text_blocks:
            return {'success': False, 'error': 'No text blocks extracted'}
        # Step 2: Encode scanned image
        image_b64 = self.encode_image(image_path)
        # Step 3: Clean and translate text blocks with vision
        print(f"🧹 Cleaning and translating {len(text_blocks)} text blocks using OpenAI Vision...")
        for i, block in enumerate(text_blocks):
            print(f"Processing block {i + 1}/{len(text_blocks)}...")
            result = self.clean_and_translate_with_vision(block, image_b64, target_language)
            block['cleaned_text'] = result['cleaned_text']
            block['translated_text'] = result['translated_text']
        # Step 4: Reconstruct PDF
        final_pdf_path = os.path.join(output_dir, f"translated_document_{timestamp}.pdf")
        if not self.reconstruct_pdf_with_translations(text_blocks, final_pdf_path):
            return {'success': False, 'error': 'Failed to reconstruct PDF'}
        # Step 5: Create summary report
        summary_path = os.path.join(output_dir, f"translation_summary_{timestamp}.json")
        summary = {
            'processing_date': datetime.now().isoformat(),
            'original_pdf': pdf_path,
            'scanned_image': image_path,
            'final_pdf': final_pdf_path,
            'target_language': target_language,
            'total_blocks': len(text_blocks),
            'successful_blocks': sum(1 for b in text_blocks if b.get('translated_text')),
            'text_blocks': text_blocks
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Workflow completed successfully!")
        print(f"📁 Output files:")
        print(f"   - Final translated PDF: {final_pdf_path}")
        print(f"   - Summary report: {summary_path}")
        return {
            'success': True,
            'summary': summary,
            'files': {
                'final_pdf': final_pdf_path,
                'summary': summary_path
            }
        }
def main():
    print("🔧 PDF Translation Workflow (PDF + Vision)")
    print("=" * 50)
    pdf_path = "converted.pdf"
    image_path = "scanned.png"
    output_dir = "pdf_translation_output"
    target_language = "English"
    if not os.path.exists(pdf_path):
        print(f"❌ Input PDF not found: {pdf_path}")
        return
    if not os.path.exists(image_path):
        print(f"❌ Scanned image not found: {image_path}")
        return
    try:
        workflow = PDFTranslationWorkflow()
        result = workflow.process_workflow(pdf_path, image_path, target_language, output_dir)
        if result['success']:
            print(f"\n📈 Final Summary:")
            print(f"Total text blocks: {result['summary']['total_blocks']}")
            print(f"Successfully processed: {result['summary']['successful_blocks']}")
            print(f"Target language: {result['summary']['target_language']}")
        else:
            print(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error during workflow: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main() 