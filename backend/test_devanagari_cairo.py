import cairo
import gi
import os

gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo

# Output PDF
output_pdf = "test_devanagari_cairo.pdf"
width, height = 600, 200

# Font
font_path = os.path.abspath("fonts/NotoSansDevanagari-Regular.ttf")
font_desc = "Noto Sans Devanagari 48"

# Register font with fontconfig (if needed)
# On most systems, if the font is not installed system-wide, you may need to copy it to ~/.fonts or use fontconfig.
# For this script, we assume the font is available to Pango (try with system font if not).

# Create PDF surface and context
surface = cairo.PDFSurface(output_pdf, width, height)
context = cairo.Context(surface)

# Create Pango layout
pangocairo_ctx = PangoCairo.create_context(context)
layout = Pango.Layout.new(pangocairo_ctx)

# Set text
text = "प्रिया योगेश शर्मा"
layout.set_text(text, -1)

# Set font description
layout.set_font_description(Pango.font_description_from_string(font_desc))

# Move to position and show layout
context.move_to(50, 80)
PangoCairo.show_layout(context, layout)

surface.finish()
print(f"PDF created: {output_pdf}") 