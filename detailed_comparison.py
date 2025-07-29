import fitz

def detailed_comparison():
    # Open both PDFs
    original = fitz.open('sample.pdf')
    production = fitz.open('sample_translated_final_unicode.pdf')
    
    orig_page = original[0]
    prod_page = production[0]
    
    # Get text blocks from both
    orig_blocks = orig_page.get_text('dict')['blocks']
    prod_blocks = prod_page.get_text('dict')['blocks']
    
    print("=== DETAILED COMPARISON ===")
    print(f"Original blocks: {len(orig_blocks)}")
    print(f"Production blocks: {len(prod_blocks)}")
    print()
    
    # Extract text content from original
    orig_text_blocks = []
    for i, block in enumerate(orig_blocks):
        if 'lines' in block:
            text = ''.join([span['text'] for line in block['lines'] for span in line['spans']])
            bbox = block['bbox']
            orig_text_blocks.append((i, bbox, text.strip()))
    
    # Extract text content from production
    prod_text_blocks = []
    prod_image_blocks = []
    for i, block in enumerate(prod_blocks):
        if 'lines' in block:
            text = ''.join([span['text'] for line in block['lines'] for span in line['spans']])
            bbox = block['bbox']
            prod_text_blocks.append((i, bbox, text.strip()))
        else:
            prod_image_blocks.append((i, block['bbox']))
    
    print("ORIGINAL TEXT BLOCKS:")
    for i, bbox, text in orig_text_blocks:
        print(f"  {i}: {text[:80]}...")
    
    print(f"\nPRODUCTION TEXT BLOCKS:")
    for i, bbox, text in prod_text_blocks:
        print(f"  {i}: {text[:80]}...")
    
    print(f"\nPRODUCTION IMAGE BLOCKS:")
    for i, bbox in prod_image_blocks:
        print(f"  {i}: Image at {bbox}")
    
    # Check for missing content
    print(f"\nMISSING ANALYSIS:")
    
    # Key content that should be present
    key_content = [
        "Pagina 1 van 1",
        "V-nummer",
        "2850241598", 
        "Zaaknummer",
        "Z1-186720992110",
        "Datum",
        "4 juni 2025",
        "Betreft",
        "Yogesh Sharma",
        "IJburglaan 816",
        "1087 EM AMSTERDAM"
    ]
    
    # Check what's missing in production PDF
    prod_all_text = ' '.join([text for _, _, text in prod_text_blocks])
    
    missing_items = []
    for item in key_content:
        if item not in prod_all_text and item.lower() not in prod_all_text.lower():
            # Check if translated version exists
            translations = {
                "Pagina 1 van 1": "Page 1 of 1",
                "V-nummer": "V-number", 
                "Zaaknummer": "Case number",
                "Datum": "Date",
                "4 juni 2025": "June 4, 2025",
                "Betreft": "Subject"
            }
            translated = translations.get(item, item)
            if translated not in prod_all_text:
                missing_items.append(f"{item} (or {translated})")
    
    if missing_items:
        print("MISSING CONTENT:")
        for item in missing_items:
            print(f"  - {item}")
    else:
        print("✅ All key content appears to be present!")
    
    original.close()
    production.close()

if __name__ == "__main__":
    detailed_comparison() 