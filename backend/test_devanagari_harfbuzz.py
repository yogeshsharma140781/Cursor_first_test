import cairo
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from fontTools.pens.cairoPen import CairoPen
import os

# Settings
text = "प्रिया योगेश शर्मा"
font_path = os.path.abspath("fonts/NotoSansDevanagari-Regular.ttf")
output_pdf = "test_devanagari_harfbuzz.pdf"
font_size = 64
width, height = 800, 200

# Load font
font = TTFont(font_path)
font_data = open(font_path, 'rb').read()
face = hb.Face(font_data)
font_hb = hb.Font(face)

# Set scale for HarfBuzz
upem = font['head'].unitsPerEm
font_hb.scale = (upem, upem)

# Shape text
buf = hb.Buffer()
buf.add_str(text)
buf.guess_segment_properties()
hb.shape(font_hb, buf)
glyph_infos = buf.glyph_infos
glyph_positions = buf.glyph_positions

# Prepare Cairo surface
surface = cairo.PDFSurface(output_pdf, width, height)
ctx = cairo.Context(surface)
ctx.set_source_rgb(0, 0, 0)
ctx.translate(50, 120)  # margin
ctx.set_line_width(1)

# Draw a red baseline for debugging
debug_baseline_y = 0
ctx.save()
ctx.set_source_rgb(1, 0, 0)
ctx.move_to(-50, debug_baseline_y)
ctx.line_to(width, debug_baseline_y)
ctx.stroke()
ctx.restore()

# Draw glyphs
x, y = 0, 0
for info, pos in zip(glyph_infos, glyph_positions):
    glyph_name = font.getGlyphName(info.codepoint)
    glyph = font.getGlyphSet()[glyph_name]
    print(f"Drawing glyph: {glyph_name} at ({x}, {y})")
    ctx.save()
    ctx.translate(x + pos.x_offset / 64.0, y - pos.y_offset / 64.0)
    ctx.scale(font_size / upem, font_size / upem)
    ctx.new_path()
    pen = CairoPen(font.getGlyphSet(), ctx)
    glyph.draw(pen)
    ctx.set_source_rgb(0, 0, 0)
    ctx.fill()
    ctx.restore()
    x += pos.x_advance / 64.0
    y -= pos.y_advance / 64.0

surface.finish()
print(f"PDF created: {output_pdf}") 