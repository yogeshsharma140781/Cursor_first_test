import pypdfium2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import red, blue, green
import sys

PDF_PATH = 'sample.pdf'
DEBUG_PDF = 'pypdfium2_debug.pdf'

# Open PDF
pdf = pypdfium2.PdfDocument(PDF_PATH)
page = pdf[0]

# Prepare ReportLab canvas
c = canvas.Canvas(DEBUG_PDF, pagesize=A4)
width, height = A4

# Enumerate page objects and print their attributes
for index, obj in enumerate(page.get_objects()):
    obj_type = getattr(obj, 'type', None)
    try:
        bbox = obj.get_pos()
    except Exception as e:
        print(f"[DEBUG] Object {index}: type={obj_type}, get_pos() error: {e}")
        continue
    print(f"[DEBUG] Object {index}: type={obj_type}, bbox={bbox}")
    if bbox is None:
        continue
    x0, y0, x1, y1 = bbox
    y0_flipped = height - y1
    # Draw by type
    if obj_type == 2:
        c.setStrokeColor(red)
        c.setLineWidth(2)
        c.rect(x0, y0_flipped, x1-x0, y1-y0, stroke=1, fill=0)
    elif obj_type == 1:
        c.setStrokeColor(blue)
        c.setLineWidth(2)
        c.rect(x0, y0_flipped, x1-x0, y1-y0, stroke=1, fill=0)
    else:
        c.setStrokeColor(green)
        c.setLineWidth(1)
        c.rect(x0, y0_flipped, x1-x0, y1-y0, stroke=1, fill=0)

c.save()
print(f"[DEBUG] Created {DEBUG_PDF} with bounding boxes for all detected images and vector objects.") 