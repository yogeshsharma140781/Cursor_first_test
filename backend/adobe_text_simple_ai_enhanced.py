#!/usr/bin/env python3
"""
Adobe Text-Based PDF Creation with AI Cleaning (Enhanced)
Takes the searchable PDF from Adobe OCR and creates a clean text-based version
with AI-powered artifact removal while preserving original positioning
"""

import os
import sys
import fitz  # PyMuPDF for text extraction with positions
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import re
import time
import json
from typing import List, Dict, Optional

# OpenAI imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not available - will use mechanical cleaning only")

def mechanical_clean(text: str) -> str:
    """
    Apply mechanical cleaning to remove common OCR artifacts
    """
    if not text:
        return text
    
    # Remove common OCR artifacts
    text = re.sub(r'[_~·;]+', '', text)  # Remove underscores, tildes, dots, semicolons
    text = re.sub(r'[|\\/]{2,}', '|', text)  # Fix broken vertical bars
    text = re.sub(r'[0O]{2,}', 'O', text)  # Fix broken O's
    text = re.sub(r'[1l]{2,}', 'l', text)  # Fix broken l's
    
    # Fix common OCR errors (Dutch-specific patterns from your original script)
    text = re.sub(r'\bva\d+\.?_?u _nc,dig\b', 'van u nodig', text)
    text = re.sub(r'c,ve_r', 'over', text)
    text = re.sub(r'pe op\\?\'?a_ng', 'de opvang', text)
    text = re.sub(r'Y\?l!1', 'van', text)
    text = re.sub(r'~in_d', 'kind', text)
    text = re.sub(r'\.informatie_', ' informatie', text)
    text = re.sub(r'_hebben', 'hebben', text)
    
    # General cleanup
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'^\s+|\s+$', '', text)  # Trim whitespace
    
    return text

def ai_clean_text(text: str, client, max_retries: int = 3) -> str:
    """
    Use OpenAI to clean OCR text artifacts
    """
    if not text.strip() or not OPENAI_AVAILABLE:
        return text
    
    prompt = (
        "You are an expert at cleaning up OCR text. "
        "Given the following text block, remove only OCR artifacts (garbled characters, underscores, random symbols, broken words, etc). "
        "Do NOT rephrase, reorder, or group text. Only fix obvious OCR errors in-place. "
        "Preserve the original meaning and structure. "
        "Return the cleaned text only.\n\nText block:\n" + text
    )
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0,
            )
            cleaned = response.choices[0].message.content.strip()
            
            # Validate that the cleaned text is reasonable
            if len(cleaned) > 0 and len(cleaned) <= len(text) * 2:  # Sanity check
                return cleaned
            else:
                print(f"⚠️ AI returned suspicious result, using mechanical cleaning")
                return text
                
        except Exception as e:
            print(f"[AI CLEAN ERROR] Attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
            else:
                print(f"❌ AI cleaning failed after {max_retries} attempts, using mechanical cleaning")
                return text
    
    return text

def extract_text_with_positions(pdf_path: str):
    """
    Extract text with positioning from a PDF using PyMuPDF
    """
    print(f"🔍 EXTRACTING TEXT WITH POSITIONS")
    print(f"📄 Input: {pdf_path}")
    print("-" * 40)
    
    try:
        # Open PDF
        doc = fitz.open(pdf_path)
        all_pages_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get text with detailed position information
            text_dict = page.get_text("dict")
            
            page_data = []
            for block in text_dict["blocks"]:
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:  # Only add non-empty text
                                bbox = span["bbox"]  # [x0, y0, x1, y1]
                                font_info = {
                                    'size': span["size"],
                                    'font': span["font"],
                                    'flags': span["flags"]  # Bold, italic, etc.
                                }
                                
                                page_data.append({
                                    'text': text,
                                    'bbox': bbox,
                                    'font': font_info
                                })
            
            all_pages_data.append(page_data)
            print(f"   📄 Page {page_num + 1}: {len(page_data)} text elements")
        
        doc.close()
        print(f"✅ Extracted text from {len(all_pages_data)} pages")
        return all_pages_data
        
    except Exception as e:
        print(f"❌ Text extraction error: {e}")
        raise

def clean_text_blocks(pages_data: List[List[Dict]], use_ai: bool = True) -> List[List[Dict]]:
    """
    Clean text blocks using mechanical and AI cleaning
    """
    print(f"\n🧹 CLEANING TEXT BLOCKS")
    print("-" * 40)
    
    # Setup OpenAI if needed
    openai_client = None
    if use_ai and OPENAI_AVAILABLE:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                openai_client = openai.Client(api_key=api_key)
                print("✅ OpenAI client initialized")
            except Exception as e:
                print(f"⚠️ OpenAI setup failed: {e}")
                use_ai = False
        else:
            print("⚠️ OPENAI_API_KEY not set, using mechanical cleaning only")
            use_ai = False
    elif not OPENAI_AVAILABLE:
        print("⚠️ OpenAI not available, using mechanical cleaning only")
        use_ai = False
    
    # Progress tracking
    progress_file = "ai_clean_progress.json"
    resume_data = {}
    
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                resume_data = json.load(f)
            print(f"📋 Resuming from saved progress")
        except:
            print("⚠️ Could not load progress file, starting fresh")
    
    cleaned_pages = []
    total_blocks = sum(len(page) for page in pages_data)
    processed_blocks = 0
    
    for page_num, page_elements in enumerate(pages_data):
        print(f"   📄 Page {page_num + 1}: {len(page_elements)} text elements")
        
        cleaned_page = []
        for element_idx, element in enumerate(page_elements):
            # Check if we should resume from this point
            element_key = f"page_{page_num}_element_{element_idx}"
            if element_key in resume_data:
                cleaned_page.append(resume_data[element_key])
                processed_blocks += 1
                continue
            
            text = element['text']
            original_text = text
            
            # Apply mechanical cleaning first
            text = mechanical_clean(text)
            
            # Apply AI cleaning if available
            if use_ai and openai_client:
                text = ai_clean_text(text, openai_client)
            
            # Store cleaned text
            cleaned_element = element.copy()
            cleaned_element['cleaned_text'] = text
            cleaned_element['original_text'] = original_text
            
            cleaned_page.append(cleaned_element)
            processed_blocks += 1
            
            # Save progress every 10 blocks
            if processed_blocks % 10 == 0:
                resume_data[element_key] = cleaned_element
                with open(progress_file, "w") as f:
                    json.dump(resume_data, f, indent=2)
                print(f"   💾 Progress saved: {processed_blocks}/{total_blocks} blocks")
            
            # Rate limiting for AI calls
            if use_ai and openai_client:
                time.sleep(0.5)  # Avoid hitting rate limits
        
        cleaned_pages.append(cleaned_page)
    
    # Clean up progress file
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    print(f"✅ Text cleaning complete: {processed_blocks} blocks processed")
    return cleaned_pages

def create_clean_text_pdf(pages_data, output_path: str):
    """
    Create a clean text-based PDF from cleaned text data
    """
    print(f"\n🔨 CREATING CLEAN TEXT-BASED PDF")
    print(f"📝 Output: {output_path}")
    print("-" * 40)
    
    # Create PDF
    c = canvas.Canvas(output_path, pagesize=A4)
    page_width, page_height = A4
    
    for page_num, page_elements in enumerate(pages_data):
        print(f"   📄 Page {page_num + 1}: {len(page_elements)} text elements")
        
        if not page_elements:
            continue
        
        # Sort elements by vertical position (top to bottom)
        page_elements.sort(key=lambda x: -x['bbox'][1])  # Negative for top-to-bottom
        
        for element in page_elements:
            # Use cleaned text if available, otherwise original
            text = element.get('cleaned_text', element['text'])
            bbox = element['bbox']
            font_info = element['font']
            
            # Position coordinates
            x = bbox[0]
            y = page_height - bbox[1]  # Flip Y coordinate for ReportLab
            
            # Font size
            font_size = font_info['size']
            if font_size < 6:
                font_size = 6
            elif font_size > 24:
                font_size = 24
            
            # Font selection
            font_name = "Helvetica"
            flags = font_info.get('flags', 0)
            if flags & 2**4:  # Bold flag
                font_name = "Helvetica-Bold"
            
            c.setFont(font_name, font_size)
            c.setFillColor(black)
            
            # Draw text
            try:
                c.drawString(x, y, text)
            except:
                # Handle encoding issues
                safe_text = text.encode('utf-8', 'ignore').decode('utf-8')
                c.drawString(x, y, safe_text)
        
        # New page if not the last page
        if page_num < len(pages_data) - 1:
            c.showPage()
    
    c.save()
    
    file_size = os.path.getsize(output_path) / 1024  # KB
    print(f"✅ Created clean text PDF: {file_size:.1f} KB")
    
    return output_path

def create_adobe_text_pdf_with_ai_cleaning(adobe_ocr_pdf: str, output_path: str = None, use_ai: bool = True) -> str:
    """
    Convert Adobe OCR PDF to clean text-based PDF with AI-powered cleaning
    """
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(adobe_ocr_pdf))[0]
        suffix = "_ai_cleaned" if use_ai else "_mechanical_cleaned"
        output_path = f"{base_name}{suffix}.pdf"
    
    print("🚀 ADOBE TO AI-CLEANED TEXT PDF CONVERSION")
    print("=" * 50)
    print(f"📂 Input: {adobe_ocr_pdf}")
    print(f"📝 Output: {output_path}")
    print(f"🤖 AI Cleaning: {'Enabled' if use_ai else 'Disabled'}")
    print()
    
    # Step 1: Extract text with positions
    pages_data = extract_text_with_positions(adobe_ocr_pdf)
    
    # Step 2: Clean text blocks
    cleaned_pages_data = clean_text_blocks(pages_data, use_ai)
    
    # Step 3: Create clean text PDF
    result_path = create_clean_text_pdf(cleaned_pages_data, output_path)
    
    print(f"\n🎉 CONVERSION COMPLETE!")
    print(f"📁 Result: {result_path}")
    print(f"✨ AI-cleaned text-based PDF created!")
    print(f"🔍 Artifacts removed while preserving original positioning")
    
    return result_path

def compare_cleaning_methods():
    """
    Compare different cleaning methods
    """
    print("\n📊 CLEANING METHOD COMPARISON")
    print("=" * 50)
    
    # Input file
    adobe_ocr_result = "scanned_adobe_ocr.pdf"
    
    if not os.path.exists(adobe_ocr_result):
        print("❌ Adobe OCR result not found. Run adobe_ocr_service.py first.")
        return
    
    results = {}
    
    try:
        # 1. Original Adobe OCR result
        results['original'] = {
            'path': adobe_ocr_result,
            'size': os.path.getsize(adobe_ocr_result) / 1024,
            'type': 'Image + Text Layer',
            'cleaning': 'None',
            'quality': 'OCR artifacts present'
        }
        
        # 2. Mechanical cleaning only
        print("\n2️⃣ MECHANICAL CLEANING")
        mechanical_result = create_adobe_text_pdf_with_ai_cleaning(adobe_ocr_result, use_ai=False)
        results['mechanical'] = {
            'path': mechanical_result,
            'size': os.path.getsize(mechanical_result) / 1024,
            'type': 'Text Elements',
            'cleaning': 'Mechanical only',
            'quality': 'Basic artifact removal'
        }
        
        # 3. AI + Mechanical cleaning
        print("\n3️⃣ AI + MECHANICAL CLEANING")
        ai_result = create_adobe_text_pdf_with_ai_cleaning(adobe_ocr_result, use_ai=True)
        results['ai_cleaned'] = {
            'path': ai_result,
            'size': os.path.getsize(ai_result) / 1024,
            'type': 'Text Elements',
            'cleaning': 'AI + Mechanical',
            'quality': 'Advanced artifact removal'
        }
        
        # Display comparison table
        print(f"\n📊 COMPLETE COMPARISON:")
        print("=" * 80)
        print(f"{'Method':<12} {'Size (KB)':<10} {'Type':<20} {'Cleaning':<15} {'Quality':<20}")
        print("=" * 80)
        
        for method, data in results.items():
            print(f"{method:<12} {data['size']:<10.1f} {data['type']:<20} {data['cleaning']:<15} {data['quality']:<20}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("🔹 Original: Good for reference")
        print("🔹 Mechanical: Fast, free, basic cleaning")
        print("🔹 AI + Mechanical: Best quality, requires OpenAI API")
        
        print(f"\n🚀 Opening all results for comparison...")
        for method, data in results.items():
            if os.path.exists(data['path']):
                os.system(f"open '{data['path']}'")
        
        return results
        
    except Exception as e:
        print(f"💥 Comparison failed: {e}")
        return None

def main():
    """Main function"""
    
    # Check if we have the Adobe OCR result
    adobe_ocr_result = "scanned_adobe_ocr.pdf"
    
    if not os.path.exists(adobe_ocr_result):
        print("❌ Adobe OCR result not found!")
        print("   Run: python adobe_ocr_service.py")
        print("   Then run this script again.")
        return
    
    print("🎯 Choose conversion method:")
    print("1. Convert with AI cleaning (requires OpenAI API)")
    print("2. Convert with mechanical cleaning only")
    print("3. Compare all cleaning methods")
    
    choice = input("\nEnter choice (1, 2, or 3) [default: 1]: ").strip()
    
    if choice == "2":
        result = create_adobe_text_pdf_with_ai_cleaning(adobe_ocr_result, use_ai=False)
        print(f"\n🚀 Opening result...")
        os.system(f"open '{result}'")
    elif choice == "3":
        compare_cleaning_methods()
    else:
        result = create_adobe_text_pdf_with_ai_cleaning(adobe_ocr_result, use_ai=True)
        print(f"\n🚀 Opening result...")
        os.system(f"open '{result}'")

if __name__ == "__main__":
    main() 