import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import os

# --- CONFIG ---
# Path to the JSON file
json_path = "structuredData.json"
# The image filename to render
image_filename = "fileoutpart2.png"
# The directory where the image is (figures/sample/ or figures/)
image_dirs = [os.path.join("figures", "sample"), "figures"]
# Output PDF
output_pdf = "single_image_page.pdf"

# --- LOAD JSON AND FIND BBOX ---
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

bbox = None
for element in data.get("elements", []):
    file_paths = element.get("filePaths")
    if file_paths and any(image_filename in fp for fp in file_paths):
        # Try BBox in attributes
        if "attributes" in element and "BBox" in element["attributes"]:
            bbox = element["attributes"]["BBox"]
            break
        # Fallback: try Bounds
        if "Bounds" in element:
            bbox = element["Bounds"]
            break

if not bbox:
    raise ValueError(f"Could not find BBox for {image_filename} in {json_path}")

x0, y0, x1, y1 = bbox
width = x1 - x0
height = y1 - y0

# --- FIND IMAGE PATH ---
image_path = None
for d in image_dirs:
    candidate = os.path.join(d, image_filename)
    if os.path.exists(candidate):
        image_path = candidate
        break
if not image_path:
    raise FileNotFoundError(f"Could not find {image_filename} in {image_dirs}")

# --- CREATE PDF ---
c = canvas.Canvas(output_pdf, pagesize=A4)
# Optionally, set page size to fit image exactly:
# from reportlab.lib.pagesizes import landscape
# c.setPageSize((max(x1, 595), max(y1, 842)))

# Draw the image at (x0, y0) with (width, height)
c.drawImage(image_path, x0, y0, width, height)
c.save()

print(f"Created {output_pdf} with {image_filename} at {bbox}") 