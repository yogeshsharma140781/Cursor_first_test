from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

FONT_PATH = os.path.join('backend', 'fonts', 'NotoSansDevanagari-Regular.ttf')
FONT_NAME = 'NotoSansDevanagari-Regular'

# Register the font
pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))

# Create a PDF
c = canvas.Canvas('test_hindi.pdf')
c.setFont(FONT_NAME, 18)
c.drawString(100, 700, 'यह एक परीक्षण है')  # This is a test in Hindi
c.save()

print('PDF generated: test_hindi.pdf') 