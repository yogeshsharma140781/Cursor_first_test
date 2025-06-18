import fitz

def analyze_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    blocks = page.get_text('dict')['blocks']
    
    print(f'=== ANALYZING {pdf_path} ===')
    print(f'Total blocks: {len(blocks)}')
    print(f'Page dimensions: {page.rect}')
    print()
    
    text_blocks = []
    image_blocks = []
    
    for i, block in enumerate(blocks):
        if 'lines' in block:
            text = ''.join([span['text'] for line in block['lines'] for span in line['spans']])
            bbox = block['bbox']
            text_blocks.append((i, bbox, text[:100]))
            print(f'Text Block {i}: {bbox}')
            print(f'  Content: "{text[:100]}..."')
            print()
        else:
            bbox = block['bbox']
            image_blocks.append((i, bbox))
            print(f'Image Block {i}: {bbox}')
            print()
    
    doc.close()
    return text_blocks, image_blocks

# Analyze both PDFs
print("ORIGINAL PDF:")
orig_text, orig_images = analyze_pdf('sample.pdf')
print("\n" + "="*50 + "\n")
print("FIXED PDF:")
fixed_text, fixed_images = analyze_pdf('sample_translated_fixed_unicode.pdf')

print(f"\nSUMMARY:")
print(f"Original: {len(orig_text)} text blocks, {len(orig_images)} image blocks")
print(f"Fixed: {len(fixed_text)} text blocks, {len(fixed_images)} image blocks") 