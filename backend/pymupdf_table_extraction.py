#!/usr/bin/env python3
"""
PyMuPDF-based table extraction with proper row/column indices.
"""

import fitz  # PyMuPDF
import json
import os
from collections import defaultdict

def extract_tables_with_pymupdf(pdf_path):
    """Extract tables using PyMuPDF with proper row/column structure"""
    
    doc = fitz.open(pdf_path)
    tables_data = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract tables from the page
        tables = page.get_tables()
        
        for table_idx, table in enumerate(tables):
            table_info = {
                'page': page_num,
                'table_index': table_idx,
                'rows': len(table),
                'cols': len(table[0]) if table else 0,
                'data': table,
                'bbox': page.get_table_bbox(table_idx) if hasattr(page, 'get_table_bbox') else None
            }
            tables_data.append(table_info)
    
    doc.close()
    return tables_data

def extract_tables_with_pymupdf_advanced(pdf_path):
    """Advanced PyMuPDF table extraction with more details"""
    
    doc = fitz.open(pdf_path)
    tables_data = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Get page dimensions
        page_rect = page.rect
        
        # Extract text blocks with their positions
        text_blocks = page.get_text("dict")
        
        # Group text blocks by position to identify table cells
        table_cells = []
        
        for block in text_blocks.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        cell_info = {
                            'text': span['text'].strip(),
                            'bbox': span['bbox'],
                            'font': span['font'],
                            'size': span['size'],
                            'flags': span['flags'],  # bold, italic, etc.
                            'page': page_num
                        }
                        if cell_info['text']:
                            table_cells.append(cell_info)
        
        # Try to identify table structure by analyzing cell positions
        tables = identify_tables_from_cells(table_cells, page_rect)
        
        for table_idx, table in enumerate(tables):
            table_info = {
                'page': page_num,
                'table_index': table_idx,
                'rows': len(table),
                'cols': len(table[0]) if table else 0,
                'data': table,
                'cells': table_cells
            }
            tables_data.append(table_info)
    
    doc.close()
    return tables_data

def identify_tables_from_cells(cells, page_rect):
    """Identify table structure from cell positions"""
    
    if not cells:
        return []
    
    # Sort cells by vertical position (top to bottom)
    cells.sort(key=lambda x: x['bbox'][1])
    
    # Group cells into rows based on vertical position
    rows = []
    current_row = []
    last_y = None
    y_tolerance = 5  # pixels
    
    for cell in cells:
        y = cell['bbox'][1]
        
        if last_y is None or abs(y - last_y) <= y_tolerance:
            current_row.append(cell)
        else:
            if current_row:
                # Sort cells in row by horizontal position
                current_row.sort(key=lambda x: x['bbox'][0])
                rows.append(current_row)
            current_row = [cell]
        
        last_y = y
    
    # Add the last row
    if current_row:
        current_row.sort(key=lambda x: x['bbox'][0])
        rows.append(current_row)
    
    # Convert to 2D table structure
    tables = []
    if rows:
        # Find maximum number of columns
        max_cols = max(len(row) for row in rows)
        
        # Create table structure
        table = []
        for row in rows:
            table_row = []
            for cell in row:
                table_row.append(cell['text'])
            # Pad row if needed
            while len(table_row) < max_cols:
                table_row.append('')
            table.append(table_row)
        
        tables.append(table)
    
    return tables

def compare_extractions(pdf_path):
    """Compare Adobe vs PyMuPDF table extraction"""
    
    print("=" * 80)
    print("COMPARING TABLE EXTRACTION METHODS")
    print("=" * 80)
    
    # PyMuPDF extraction
    print("\n📊 PyMuPDF Table Extraction:")
    print("-" * 40)
    
    try:
        pymupdf_tables = extract_tables_with_pymupdf(pdf_path)
        print(f"Found {len(pymupdf_tables)} tables with PyMuPDF")
        
        for i, table in enumerate(pymupdf_tables):
            print(f"\nTable {i+1} (Page {table['page']+1}):")
            print(f"  Rows: {table['rows']}, Columns: {table['cols']}")
            if table['bbox']:
                print(f"  BBox: {table['bbox']}")
            
            # Show first few rows
            print("  Sample data:")
            for row_idx, row in enumerate(table['data'][:3]):  # First 3 rows
                print(f"    Row {row_idx}: {row}")
            if len(table['data']) > 3:
                print(f"    ... and {len(table['data']) - 3} more rows")
    
    except Exception as e:
        print(f"PyMuPDF extraction error: {e}")
    
    # Advanced PyMuPDF extraction
    print("\n🔍 Advanced PyMuPDF Extraction:")
    print("-" * 40)
    
    try:
        advanced_tables = extract_tables_with_pymupdf_advanced(pdf_path)
        print(f"Found {len(advanced_tables)} tables with advanced method")
        
        for i, table in enumerate(advanced_tables):
            print(f"\nTable {i+1} (Page {table['page']+1}):")
            print(f"  Rows: {table['rows']}, Columns: {table['cols']}")
            print(f"  Cells found: {len(table['cells'])}")
            
            # Show first few rows
            print("  Sample data:")
            for row_idx, row in enumerate(table['data'][:3]):  # First 3 rows
                print(f"    Row {row_idx}: {row}")
            if len(table['data']) > 3:
                print(f"    ... and {len(table['data']) - 3} more rows")
    
    except Exception as e:
        print(f"Advanced PyMuPDF extraction error: {e}")
    
    # Save PyMuPDF results
    output_file = "pymupdf_tables.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(advanced_tables, f, indent=2, ensure_ascii=False)
    
    print(f"\nPyMuPDF results saved to: {output_file}")

def main():
    """Main function"""
    pdf_file = "sample3.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"Error: {pdf_file} not found!")
        return
    
    compare_extractions(pdf_file)

if __name__ == "__main__":
    main() 