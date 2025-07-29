#!/usr/bin/env python3
"""
Extract all table cells from structuredData.json and display them in a formatted table.
"""

import json
from collections import defaultdict

def extract_table_cells(json_file_path):
    """Extract all table cells and their content from structuredData.json"""
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Dictionary to store cell containers (TD/TH)
    cell_containers = {}
    # Dictionary to store cell content (/P elements)
    cell_content = {}
    
    # Extract all elements
    for element in data.get('elements', []):
        path = element.get('Path', '')
        
        # Check if this is a table cell container (TD or TH)
        if path.endswith('/TD') or path.endswith('/TH'):
            object_id = element.get('ObjectID')
            attributes = element.get('attributes', {})
            row_index = attributes.get('RowIndex')
            col_index = attributes.get('ColIndex')
            
            cell_containers[object_id] = {
                'path': path,
                'row_index': row_index,
                'col_index': col_index,
                'object_id': object_id
            }
        
        # Check if this is a cell content (/P element)
        elif '/P' in path and ('/TD/P' in path or '/TH/P' in path):
            object_id = element.get('ObjectID')
            text = element.get('Text', '')
            
            cell_content[object_id] = {
                'path': path,
                'text': text,
                'object_id': object_id
            }
    
    # Match cells with their content
    table_cells = []
    
    for container_id, container_info in cell_containers.items():
        # Find the corresponding /P element
        cell_path = container_info['path']
        p_path = cell_path + '/P'
        
        # Look for the /P element
        p_element = None
        for content_id, content_info in cell_content.items():
            if content_info['path'] == p_path:
                p_element = content_info
                break
        
        # If no /P element found, use empty text
        text = p_element['text'] if p_element else ''
        
        table_cells.append({
            'cell_path': container_info['path'],
            'row_index': container_info['row_index'],
            'col_index': container_info['col_index'],
            'p_path': p_path,
            'text': text
        })
    
    # Sort by row and column
    table_cells.sort(key=lambda x: (x['row_index'], x['col_index']))
    
    return table_cells

def print_table_cells(cells):
    """Print table cells in a formatted table"""
    
    print("=" * 120)
    print("TABLE CELLS EXTRACTION FROM structuredData.json")
    print("=" * 120)
    print()
    
    # Print header
    print(f"{'Row':<4} {'Col':<4} {'Cell Path':<50} {'/P Path':<50} {'Text':<30}")
    print("-" * 120)
    
    # Print each cell
    for cell in cells:
        row = cell['row_index'] if cell['row_index'] is not None else 'N/A'
        col = cell['col_index'] if cell['col_index'] is not None else 'N/A'
        cell_path = cell['cell_path'][:48] + '..' if len(cell['cell_path']) > 50 else cell['cell_path']
        p_path = cell['p_path'][:48] + '..' if len(cell['p_path']) > 50 else cell['p_path']
        text = cell['text'][:28] + '..' if len(cell['text']) > 30 else cell['text']
        
        print(f"{row:<4} {col:<4} {cell_path:<50} {p_path:<50} {text:<30}")
    
    print("-" * 120)
    print(f"Total cells found: {len(cells)}")
    print()

def main():
    """Main function"""
    json_file = "structuredData.json"
    
    try:
        cells = extract_table_cells(json_file)
        print_table_cells(cells)
        
        # Also save to a file for easier viewing
        output_file = "table_cells_extraction.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("TABLE CELLS EXTRACTION FROM structuredData.json\n")
            f.write("=" * 120 + "\n\n")
            f.write(f"{'Row':<4} {'Col':<4} {'Cell Path':<50} {'/P Path':<50} {'Text':<30}\n")
            f.write("-" * 120 + "\n")
            
            for cell in cells:
                row = cell['row_index'] if cell['row_index'] is not None else 'N/A'
                col = cell['col_index'] if cell['col_index'] is not None else 'N/A'
                cell_path = cell['cell_path'][:48] + '..' if len(cell['cell_path']) > 50 else cell['cell_path']
                p_path = cell['p_path'][:48] + '..' if len(cell['p_path']) > 50 else cell['p_path']
                text = cell['text'][:28] + '..' if len(cell['text']) > 30 else cell['text']
                
                f.write(f"{row:<4} {col:<4} {cell_path:<50} {p_path:<50} {text:<30}\n")
            
            f.write("-" * 120 + "\n")
            f.write(f"Total cells found: {len(cells)}\n")
        
        print(f"Full extraction also saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: {json_file} not found!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 