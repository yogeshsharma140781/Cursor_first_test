#!/usr/bin/env python3

"""
Solution for complex script rendering using HarfBuzz text shaping
This addresses the ReportLab limitation with Devanagari conjuncts
"""

import subprocess
import sys
import os
from pathlib import Path

def check_harfbuzz_available():
    """Check if HarfBuzz Python bindings are available"""
    try:
        import uharfbuzz as hb
        return True
    except ImportError:
        return False

def install_harfbuzz():
    """Install HarfBuzz Python bindings"""
    print("Installing HarfBuzz for complex script shaping...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uharfbuzz"])
        print("✅ HarfBuzz installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install HarfBuzz: {e}")
        return False

def shape_devanagari_text(text: str, font_path: str) -> list:
    """Shape Devanagari text using HarfBuzz for proper glyph formation"""
    try:
        import uharfbuzz as hb
        
        # Load font
        with open(font_path, 'rb') as f:
            font_data = f.read()
        
        # Create HarfBuzz objects
        face = hb.Face(font_data)
        font = hb.Font(face)
        
        # Create buffer and add text
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        
        # Shape the text
        hb.shape(font, buf)
        
        # Get shaped glyphs
        infos = buf.glyph_infos
        positions = buf.glyph_positions
        
        shaped_glyphs = []
        for info, pos in zip(infos, positions):
            shaped_glyphs.append({
                'glyph_id': info.codepoint,
                'cluster': info.cluster,
                'x_advance': pos.x_advance,
                'y_advance': pos.y_advance,
                'x_offset': pos.x_offset,
                'y_offset': pos.y_offset
            })
        
        return shaped_glyphs
        
    except Exception as e:
        print(f"❌ HarfBuzz shaping failed: {e}")
        return None

def create_shaped_pdf_test():
    """Create a test PDF using HarfBuzz-shaped text"""
    
    # Check if HarfBuzz is available
    if not check_harfbuzz_available():
        print("HarfBuzz not available. Installing...")
        if not install_harfbuzz():
            print("❌ Cannot proceed without HarfBuzz")
            return False
    
    import uharfbuzz as hb
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    # Font path
    fonts_dir = Path.home() / 'Library' / 'Fonts'
    font_path = str(fonts_dir / 'NotoSansDevanagari-Regular.ttf')
    
    if not os.path.exists(font_path):
        print(f"❌ Font not found: {font_path}")
        return False
    
    # Test text
    test_text = "प्रिय श्री शर्मा"
    
    print(f"Shaping text: {test_text}")
    shaped_glyphs = shape_devanagari_text(test_text, font_path)
    
    if shaped_glyphs:
        print(f"✅ Successfully shaped {len(shaped_glyphs)} glyphs")
        for i, glyph in enumerate(shaped_glyphs):
            print(f"  Glyph {i}: ID={glyph['glyph_id']}, advance=({glyph['x_advance']}, {glyph['y_advance']})")
    else:
        print("❌ Shaping failed")
        return False
    
    # Register font in ReportLab
    try:
        pdfmetrics.registerFont(TTFont('NotoDevanagari', font_path))
        print("✅ Font registered in ReportLab")
    except Exception as e:
        print(f"❌ Font registration failed: {e}")
        return False
    
    # Create PDF
    c = canvas.Canvas('harfbuzz_shaped_test.pdf', pagesize=letter)
    
    # Title
    c.setFont("Helvetica", 16)
    c.drawString(50, 750, "HarfBuzz-Shaped Devanagari Text Test")
    
    # Original text (ReportLab default)
    c.setFont("Helvetica", 12)
    c.drawString(50, 700, "ReportLab Default:")
    c.setFont("NotoDevanagari", 14)
    c.drawString(70, 680, test_text)
    
    # Shaped text info
    c.setFont("Helvetica", 12)
    c.drawString(50, 650, "HarfBuzz Shaping Analysis:")
    
    y_pos = 630
    for i, glyph in enumerate(shaped_glyphs):
        c.setFont("Helvetica", 8)
        glyph_info = f"Glyph {i}: ID={glyph['glyph_id']}, X-advance={glyph['x_advance']}"
        c.drawString(70, y_pos, glyph_info)
        y_pos -= 15
    
    # Note about limitations
    c.setFont("Helvetica", 10)
    c.drawString(50, 400, "NOTE: ReportLab cannot directly use HarfBuzz-shaped glyphs.")
    c.drawString(50, 385, "This test demonstrates the shaping analysis.")
    c.drawString(50, 370, "A complete solution would require:")
    c.drawString(70, 355, "1. Custom glyph positioning in ReportLab")
    c.drawString(70, 340, "2. Or switching to a different PDF library")
    c.drawString(70, 325, "3. Or using pre-shaped font variants")
    
    c.save()
    
    print("✅ Created: harfbuzz_shaped_test.pdf")
    return True

def alternative_solution_recommendations():
    """Provide recommendations for solving the complex script issue"""
    
    print("\n=== COMPLEX SCRIPT RENDERING SOLUTIONS ===")
    print()
    print("PROBLEM: ReportLab doesn't handle Devanagari conjunct formation properly")
    print("CAUSE: Missing complex script shaping engine (like HarfBuzz)")
    print()
    print("SOLUTION OPTIONS:")
    print()
    print("1. 🔧 FONT-BASED SOLUTION (Recommended)")
    print("   - Use a font that has pre-composed conjuncts")
    print("   - Try different Devanagari fonts (Mangal, Shree714, etc.)")
    print("   - Some fonts render better in ReportLab than others")
    print()
    print("2. 🔄 PDF LIBRARY SWITCH")
    print("   - WeasyPrint: Better Unicode support, loses visual elements")
    print("   - fpdf2: Lightweight but limited features")
    print("   - cairo/pango: Complex setup but excellent text rendering")
    print()
    print("3. 🎯 HYBRID APPROACH")
    print("   - Use ReportLab for layout and visual elements")
    print("   - Use WeasyPrint/other library for text-only overlay")
    print("   - Merge the two PDFs")
    print()
    print("4. 🔤 PRE-PROCESSING")
    print("   - Convert complex conjuncts to simpler forms")
    print("   - Use font fallback for problematic characters")
    print("   - Accept some visual compromise for compatibility")
    print()
    print("IMMEDIATE NEXT STEPS:")
    print("1. Test different system Devanagari fonts")
    print("2. Try font-specific rendering settings")
    print("3. Consider user feedback on acceptable quality level")

if __name__ == "__main__":
    print("=== COMPLEX SCRIPT SOLUTION ANALYSIS ===")
    print()
    
    # Try the HarfBuzz approach
    success = create_shaped_pdf_test()
    
    if not success:
        print("\nHarfBuzz approach not feasible. Showing alternatives...")
    
    alternative_solution_recommendations() 