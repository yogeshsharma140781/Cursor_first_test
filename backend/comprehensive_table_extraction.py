#!/usr/bin/env python3
"""
Comprehensive table extraction comparison: Adobe vs PyMuPDF vs Camelot
"""

import json
import os
import fitz  # PyMuPDF
import camelot
import pandas as pd
from collections import defaultdict

def extract_with_adobe(json_file):
    """Extract table info from Adobe structuredData.json"""
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tables = []
    cells = []
    
    for element in data.get('elements', []):
        path = element.get('Path', '')
        
        if path.endswith('/Table'):
            # Table container
            attrs = element.get('attributes', {})
            tables.append({
                'method': 'Adobe',
                'path': path,
                'num_rows': attrs.get('NumRow', 0),
                'num_cols': attrs.get('NumCol', 0),
                'bbox': attrs.get('BBox', []),
                'object_id': element.get('ObjectID')
            })
        
        elif '/TD' in path or '/TH' in path:
            # Table cell
            attrs = element.get('attributes', {})
            cells.append({
                'method': 'Adobe',
                'path': path,
                'text': element.get('Text', ''),
                'row_index': attrs.get('RowIndex'),
                'col_index': attrs.get('ColIndex'),
                'bbox': attrs.get('BBox', []),
                'object_id': element.get('ObjectID')
            })
    
    return tables, cells

def extract_with_pymupdf(pdf_path):
    """Extract tables using PyMuPDF"""
    
    doc = fitz.open(pdf_path)
    tables = []
    cells = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Get text blocks with positions
        text_blocks = page.get_text("dict")
        
        # Extract all text spans with their positions
        page_cells = []
        for block in text_blocks.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span['text'].strip():
                            page_cells.append({
                                'method': 'PyMuPDF',
                                'text': span['text'].strip(),
                                'bbox': span['bbox'],
                                'font': span['font'],
                                'size': span['size'],
                                'page': page_num
                            })
        
        # Group cells into table structure
        if page_cells:
            # Sort by position
            page_cells.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
            
            # Simple table detection based on position clustering
            table_data = detect_table_structure(page_cells)
            
            if table_data:
                tables.append({
                    'method': 'PyMuPDF',
                    'page': page_num,
                    'rows': len(table_data),
                    'cols': max(len(row) for row in table_data) if table_data else 0,
                    'data': table_data
                })
                
                # Convert to cell format
                for row_idx, row in enumerate(table_data):
                    for col_idx, cell_text in enumerate(row):
                        if cell_text.strip():
                            cells.append({
                                'method': 'PyMuPDF',
                                'text': cell_text,
                                'row_index': row_idx,
                                'col_index': col_idx,
                                'page': page_num
                            })
    
    doc.close()
    return tables, cells

def extract_with_camelot(pdf_path):
    """Extract tables using Camelot"""
    
    tables = []
    cells = []
    
    try:
        # Extract tables using Camelot
        camelot_tables = camelot.read_pdf(pdf_path, pages='all')
        
        for table_idx, table in enumerate(camelot_tables):
            df = table.df
            
            tables.append({
                'method': 'Camelot',
                'page': table.page,
                'rows': len(df),
                'cols': len(df.columns),
                'accuracy': table.accuracy,
                'whitespace': table.whitespace,
                'data': df.values.tolist()
            })
            
            # Convert to cell format
            for row_idx, row in enumerate(df.values):
                for col_idx, cell_text in enumerate(row):
                    if str(cell_text).strip():
                        cells.append({
                            'method': 'Camelot',
                            'text': str(cell_text).strip(),
                            'row_index': row_idx,
                            'col_index': col_idx,
                            'page': table.page
                        })
    
    except Exception as e:
        print(f"Camelot extraction error: {e}")
    
    return tables, cells

def detect_table_structure(cells, y_tolerance=5, x_tolerance=10):
    """Detect table structure from positioned cells"""
    
    if not cells:
        return []
    
    # Group cells by vertical position (rows)
    rows = defaultdict(list)
    
    for cell in cells:
        y = cell['bbox'][1]
        # Find closest row
        found_row = False
        for row_y in rows.keys():
            if abs(y - row_y) <= y_tolerance:
                rows[row_y].append(cell)
                found_row = True
                break
        
        if not found_row:
            rows[y].append(cell)
    
    # Sort rows by Y position and cells within each row by X position
    sorted_rows = []
    for y in sorted(rows.keys()):
        row_cells = sorted(rows[y], key=lambda x: x['bbox'][0])
        sorted_rows.append(row_cells)
    
    # Convert to 2D table
    if not sorted_rows:
        return []
    
    # Find maximum columns
    max_cols = max(len(row) for row in sorted_rows)
    
    # Create table structure
    table = []
    for row in sorted_rows:
        table_row = []
        for cell in row:
            table_row.append(cell['text'])
        
        # Pad row if needed
        while len(table_row) < max_cols:
            table_row.append('')
        
        table.append(table_row)
    
    return table

def compare_methods(pdf_path, json_path):
    """Compare all extraction methods"""
    
    print("=" * 100)
    print("COMPREHENSIVE TABLE EXTRACTION COMPARISON")
    print("=" * 100)
    
    results = {}
    
    # Adobe extraction
    print("\n📊 ADOBE PDF SERVICES SDK:")
    print("-" * 50)
    try:
        adobe_tables, adobe_cells = extract_with_adobe(json_path)
        results['Adobe'] = {'tables': adobe_tables, 'cells': adobe_cells}
        
        print(f"Tables found: {len(adobe_tables)}")
        for table in adobe_tables:
            print(f"  - {table['path']}: {table['num_rows']} rows, {table['num_cols']} cols")
        
        print(f"Cells found: {len(adobe_cells)}")
        cells_with_indices = sum(1 for cell in adobe_cells if cell['row_index'] is not None and cell['col_index'] is not None)
        print(f"Cells with row/col indices: {cells_with_indices}")
        print(f"Cells missing indices: {len(adobe_cells) - cells_with_indices}")
        
    except Exception as e:
        print(f"Adobe extraction error: {e}")
    
    # PyMuPDF extraction
    print("\n🔍 PYMUPDF:")
    print("-" * 50)
    try:
        pymupdf_tables, pymupdf_cells = extract_with_pymupdf(pdf_path)
        results['PyMuPDF'] = {'tables': pymupdf_tables, 'cells': pymupdf_cells}
        
        print(f"Tables found: {len(pymupdf_tables)}")
        for table in pymupdf_tables:
            print(f"  - Page {table['page']+1}: {table['rows']} rows, {table['cols']} cols")
        
        print(f"Cells found: {len(pymupdf_cells)}")
        cells_with_indices = sum(1 for cell in pymupdf_cells if cell['row_index'] is not None and cell['col_index'] is not None)
        print(f"Cells with row/col indices: {cells_with_indices}")
        print(f"Cells missing indices: {len(pymupdf_cells) - cells_with_indices}")
        
    except Exception as e:
        print(f"PyMuPDF extraction error: {e}")
    
    # Camelot extraction
    print("\n🐪 CAMELOT:")
    print("-" * 50)
    try:
        camelot_tables, camelot_cells = extract_with_camelot(pdf_path)
        results['Camelot'] = {'tables': camelot_tables, 'cells': camelot_cells}
        
        print(f"Tables found: {len(camelot_tables)}")
        for table in camelot_tables:
            print(f"  - Page {table['page']}: {table['rows']} rows, {table['cols']} cols")
            print(f"    Accuracy: {table['accuracy']:.2f}%, Whitespace: {table['whitespace']:.2f}%")
        
        print(f"Cells found: {len(camelot_cells)}")
        cells_with_indices = sum(1 for cell in camelot_cells if cell['row_index'] is not None and cell['col_index'] is not None)
        print(f"Cells with row/col indices: {cells_with_indices}")
        print(f"Cells missing indices: {len(camelot_cells) - cells_with_indices}")
        
    except Exception as e:
        print(f"Camelot extraction error: {e}")
    
    # Summary comparison
    print("\n📈 SUMMARY COMPARISON:")
    print("-" * 50)
    print(f"{'Method':<12} {'Tables':<8} {'Cells':<8} {'With Indices':<12} {'Success Rate':<12}")
    print("-" * 50)
    
    for method, data in results.items():
        tables_count = len(data['tables'])
        cells_count = len(data['cells'])
        cells_with_indices = sum(1 for cell in data['cells'] if cell['row_index'] is not None and cell['col_index'] is not None)
        success_rate = (cells_with_indices / cells_count * 100) if cells_count > 0 else 0
        
        print(f"{method:<12} {tables_count:<8} {cells_count:<8} {cells_with_indices:<12} {success_rate:<12.1f}%")
    
    # Save results
    output_file = "table_extraction_comparison.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return results

def main():
    """Main function"""
    pdf_file = "sample3.pdf"
    json_file = "structuredData.json"
    
    if not os.path.exists(pdf_file):
        print(f"Error: {pdf_file} not found!")
        return
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found!")
        return
    
    compare_methods(pdf_file, json_file)

if __name__ == "__main__":
    main() 