from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import os

# Register the font
FONT_PATH = os.path.abspath("fonts/NotoSansDevanagari-Regular.ttf")
FONT_NAME = "NotoSansDevanagari-Regular"
if FONT_NAME not in pdfmetrics.getRegisteredFontNames():
    pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))

# Create PDF
c = canvas.Canvas("test_devanagari.pdf", pagesize=A4)
c.setFont(FONT_NAME, 32)
c.drawString(100, 700, "प्रिया योगेश शर्मा")
c.save()
print("PDF created: test_devanagari.pdf") 