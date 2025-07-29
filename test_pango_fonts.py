#!/usr/bin/env python3

import gi
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo
import cairo

def test_pango_fonts():
    """Test Pango font availability and rendering"""
    
    # Create a test surface
    surface = cairo.PDFSurface("test_pango_fonts.pdf", 595, 842)
    ctx = cairo.Context(surface)
    
    # Set white background
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    
    # Create Pango layout
    layout = PangoCairo.create_layout(ctx)
    
    # Test fonts
    test_fonts = [
        "Noto Sans",
        "Noto Sans Devanagari", 
        "Noto Sans Devanagari Bold",
        "Arial",
        "Helvetica"
    ]
    
    y_position = 50
    
    for font_name in test_fonts:
        try:
            # Create font description
            font_desc = Pango.FontDescription()
            font_desc.set_family(font_name)
            font_desc.set_size(12 * Pango.SCALE)
            
            # Set up layout
            layout.set_font_description(font_desc)
            layout.set_text(f"Test text in {font_name}", -1)
            
            # Render text
            ctx.move_to(50, y_position)
            ctx.set_source_rgb(0, 0, 0)
            PangoCairo.show_layout(ctx, layout)
            
            print(f"✅ Successfully rendered with {font_name}")
            y_position += 30
            
        except Exception as e:
            print(f"❌ Failed to render with {font_name}: {e}")
            y_position += 30
    
    # Test Devanagari text
    try:
        devanagari_text = "प्रिय श्री शर्मा"
        font_desc = Pango.FontDescription()
        font_desc.set_family("Noto Sans Devanagari")
        font_desc.set_size(16 * Pango.SCALE)
        
        layout.set_font_description(font_desc)
        layout.set_text(devanagari_text, -1)
        
        ctx.move_to(50, y_position)
        ctx.set_source_rgb(0, 0, 0)
        PangoCairo.show_layout(ctx, layout)
        
        print(f"✅ Successfully rendered Devanagari: {devanagari_text}")
        
    except Exception as e:
        print(f"❌ Failed to render Devanagari: {e}")
    
    surface.finish()
    print("Test PDF created: test_pango_fonts.pdf")

if __name__ == "__main__":
    test_pango_fonts() 