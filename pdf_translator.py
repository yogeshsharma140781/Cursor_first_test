import pypdfium2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import Color
from reportlab.lib.utils import ImageReader
from PIL import Image
import io
import os
from googletrans import Translator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFTranslator:
    def __init__(self, input_pdf, output_pdf, target_lang='hi'):
        self.input_pdf = input_pdf
        self.output_pdf = output_pdf
        self.target_lang = target_lang
        self.translator = Translator()
        self.width, self.height = A4
        
    def extract_and_translate_text(self, text):
        try:
            if text and text.strip():
                translated = self.translator.translate(text, dest=self.target_lang).text
                return translated
            return text
        except Exception as e:
            logger.error(f"Error translating text: {e}, text: {text}")
            return ""

    def process_page(self, page, canvas):
        try:
            for obj in page.get_objects():
                bbox = obj.get_pos()
                if not (isinstance(bbox, tuple) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
                    continue
                x0, y0, x1, y1 = bbox
                y0_flipped = self.height - y1
                # Image object
                if isinstance(obj, pypdfium2.PdfImage):
                    try:
                        bitmap = obj.get_bitmap(render=True)
                        pil_image = bitmap.to_pil()
                        img_byte_arr = io.BytesIO()
                        pil_image.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        img_reader = ImageReader(img_byte_arr)
                        canvas.drawImage(img_reader, x0, y0_flipped, width=x1-x0, height=y1-y0, preserveAspectRatio=True)
                    except Exception as e:
                        logger.error(f"Error extracting image: {e}")
                # Text object
                elif getattr(obj, 'type', None) == 1:
                    try:
                        text = obj.get_text() if hasattr(obj, 'get_text') else None
                        translated_text = self.extract_and_translate_text(text)
                        if translated_text:
                            font_size = 10  # Default, as font size extraction is not direct here
                            canvas.setFont("Helvetica", font_size)
                            canvas.drawString(x0, y0_flipped, translated_text)
                    except Exception as e:
                        logger.error(f"Error processing text object: {e}")
        except Exception as e:
            logger.error(f"Error processing page: {e}")

    def translate_pdf(self):
        try:
            pdf = pypdfium2.PdfDocument(self.input_pdf)
            c = canvas.Canvas(self.output_pdf, pagesize=A4)
            for page_num in range(len(pdf)):
                logger.info(f"Processing page {page_num + 1}")
                page = pdf[page_num]
                self.process_page(page, c)
                c.showPage()
            c.save()
            logger.info(f"Translation complete. Output saved to {self.output_pdf}")
        except Exception as e:
            logger.error(f"Error in PDF translation: {e}")
            raise

def main():
    input_pdf = "sample.pdf"
    output_pdf = "translated_output.pdf"
    target_lang = "hi"  # Hindi
    translator = PDFTranslator(input_pdf, output_pdf, target_lang)
    translator.translate_pdf()

if __name__ == "__main__":
    main() 