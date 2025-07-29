#!/usr/bin/env python3
"""
Verify and compare the original and cleaned formatted docx files
"""

import os
from docx import Document
from datetime import datetime

def extract_text_from_docx(docx_path: str) -> list:
    """Extract text from docx file"""
    doc = Document(docx_path)
    paragraphs = []
    
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip():
            paragraphs.append({
                'index': i,
                'text': para.text.strip(),
                'style': para.style.name if para.style else 'Normal',
                'alignment': str(para.alignment)
            })
    
    return paragraphs

def compare_documents(original_path: str, cleaned_path: str):
    """Compare original and cleaned documents"""
    print("🔍 Document Comparison")
    print("=" * 50)
    
    # Extract text from both documents
    print(f"📖 Reading original document: {original_path}")
    original_paras = extract_text_from_docx(original_path)
    
    print(f"📖 Reading cleaned document: {cleaned_path}")
    cleaned_paras = extract_text_from_docx(cleaned_path)
    
    print(f"\n📊 Document Statistics:")
    print(f"Original paragraphs: {len(original_paras)}")
    print(f"Cleaned paragraphs: {len(cleaned_paras)}")
    
    # Show sample comparisons
    print(f"\n📝 Sample Comparisons (first 10 paragraphs):")
    print("-" * 80)
    
    for i in range(min(10, len(original_paras), len(cleaned_paras))):
        orig = original_paras[i]
        clean = cleaned_paras[i]
        
        print(f"\nParagraph {i + 1}:")
        print(f"Style: {orig['style']} | Alignment: {orig['alignment']}")
        print(f"Original: {orig['text'][:100]}{'...' if len(orig['text']) > 100 else ''}")
        print(f"Cleaned:  {clean['text'][:100]}{'...' if len(clean['text']) > 100 else ''}")
        
        # Show if text changed
        if orig['text'] != clean['text']:
            print("✅ Text was cleaned!")
        else:
            print("➡️  No changes needed")
    
    # Show formatting preservation
    print(f"\n🎨 Formatting Preservation Check:")
    print("-" * 40)
    
    formatting_preserved = 0
    for i in range(min(len(original_paras), len(cleaned_paras))):
        orig = original_paras[i]
        clean = cleaned_paras[i]
        
        if orig['style'] == clean['style'] and orig['alignment'] == clean['alignment']:
            formatting_preserved += 1
    
    preservation_rate = (formatting_preserved / min(len(original_paras), len(cleaned_paras))) * 100
    print(f"Formatting preserved: {formatting_preserved}/{min(len(original_paras), len(cleaned_paras))} ({preservation_rate:.1f}%)")

def main():
    """Main function"""
    print("🔧 DocX Format Verification")
    print("=" * 50)
    
    # File paths
    original_docx = "converted.docx"
    simple_cleaned = "formatted_output/cleaned_formatted_simple_20250704_134010.docx"
    full_cleaned = "formatted_output/cleaned_formatted_full_20250704_134010.docx"
    
    # Check if files exist
    if not os.path.exists(original_docx):
        print(f"❌ Original file not found: {original_docx}")
        return
    
    if not os.path.exists(simple_cleaned):
        print(f"❌ Simple cleaned file not found: {simple_cleaned}")
        return
    
    if not os.path.exists(full_cleaned):
        print(f"❌ Full cleaned file not found: {full_cleaned}")
        return
    
    # Compare simple version
    print("\n📋 Comparing Simple Formatted Version:")
    compare_documents(original_docx, simple_cleaned)
    
    # Compare full version
    print("\n📋 Comparing Full Formatted Version:")
    compare_documents(original_docx, full_cleaned)
    
    print(f"\n✅ Verification complete!")
    print(f"📁 Files available:")
    print(f"   - Simple formatted: {simple_cleaned}")
    print(f"   - Full formatted: {full_cleaned}")

if __name__ == "__main__":
    main() 