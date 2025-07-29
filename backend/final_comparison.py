
import os
import fitz
from PIL import Image, ImageDraw, ImageFont

def create_final_comparison():
    original_pdf = "../scanned.pdf"
    perfect_pdf = "scanned_PERFECT_LAYOUT.pdf"
    
    if not os.path.exists(original_pdf) or not os.path.exists(perfect_pdf):
        print("❌ PDFs not found for comparison")
        return
    
    # Convert both to images
    def pdf_to_image(pdf_path, dpi=150):
        doc = fitz.open(pdf_path)
        page = doc[0]
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("ppm")
        image = Image.open(fitz.io.BytesIO(img_data))
        doc.close()
        return image
    
    original_img = pdf_to_image(original_pdf)
    perfect_img = pdf_to_image(perfect_pdf)
    
    # Resize to same height
    target_height = min(original_img.size[1], perfect_img.size[1])
    
    original_ratio = original_img.size[0] / original_img.size[1]
    perfect_ratio = perfect_img.size[0] / perfect_img.size[1]
    
    original_resized = original_img.resize((int(target_height * original_ratio), target_height), Image.Resampling.LANCZOS)
    perfect_resized = perfect_img.resize((int(target_height * perfect_ratio), target_height), Image.Resampling.LANCZOS)
    
    # Create side-by-side comparison
    total_width = original_resized.size[0] + perfect_resized.size[0] + 60
    comparison_img = Image.new('RGB', (total_width, target_height + 60), 'white')
    
    # Paste images
    comparison_img.paste(original_resized, (20, 40))
    comparison_img.paste(perfect_resized, (original_resized.size[0] + 40, 40))
    
    # Add labels
    draw = ImageDraw.Draw(comparison_img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial Bold.ttf", 20)
        label_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Title and labels
    draw.text((total_width//2 - 180, 10), "FINAL LAYOUT COMPARISON", fill='black', font=font)
    draw.text((20, 15), "ORIGINAL (Scanned)", fill='red', font=label_font)
    draw.text((original_resized.size[0] + 40, 15), "RECREATED (Perfect Layout)", fill='blue', font=label_font)
    
    # Save comparison
    comparison_path = "FINAL_layout_comparison.png"
    comparison_img.save(comparison_path, 'PNG', quality=95)
    
    file_size = os.path.getsize(comparison_path) / 1024
    print(f"✅ Final comparison created: {comparison_path} ({file_size:.1f} KB)")
    
    return comparison_path

if __name__ == "__main__":
    create_final_comparison()
