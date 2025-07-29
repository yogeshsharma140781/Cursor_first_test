#!/usr/bin/env python3
"""
Analyze all text elements in structuredData.json to understand their structure and categorization.
"""

import json
from collections import defaultdict

def analyze_text_elements(json_file_path):
    """Analyze all text elements and categorize them"""
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Categories for text elements
    categories = {
        'table_cells': [],
        'table_headers': [],
        'standalone_text': [],
        'titles': [],
        'footnotes': [],
        'list_items': [],
        'other': []
    }
    
    # Track table structure
    table_elements = []
    cell_elements = []
    
    for element in data.get('elements', []):
        path = element.get('Path', '')
        text = element.get('Text', '').strip()
        object_id = element.get('ObjectID')
        
        if not text:  # Skip empty text
            continue
            
        # Categorize based on path
        if '/Table' in path:
            if path.endswith('/Table'):
                table_elements.append({
                    'object_id': object_id,
                    'path': path,
                    'attributes': element.get('attributes', {})
                })
            elif '/TD' in path:
                cell_elements.append({
                    'object_id': object_id,
                    'path': path,
                    'text': text,
                    'attributes': element.get('attributes', {})
                })
                categories['table_cells'].append({
                    'object_id': object_id,
                    'path': path,
                    'text': text,
                    'row': element.get('attributes', {}).get('RowIndex'),
                    'col': element.get('attributes', {}).get('ColIndex')
                })
            elif '/TH' in path:
                cell_elements.append({
                    'object_id': object_id,
                    'path': path,
                    'text': text,
                    'attributes': element.get('attributes', {})
                })
                categories['table_headers'].append({
                    'object_id': object_id,
                    'path': path,
                    'text': text,
                    'row': element.get('attributes', {}).get('RowIndex'),
                    'col': element.get('attributes', {}).get('ColIndex')
                })
        elif '/Title' in path:
            categories['titles'].append({
                'object_id': object_id,
                'path': path,
                'text': text
            })
        elif '/Footnote' in path:
            categories['footnotes'].append({
                'object_id': object_id,
                'path': path,
                'text': text
            })
        elif '/L/LI' in path:
            categories['list_items'].append({
                'object_id': object_id,
                'path': path,
                'text': text
            })
        elif '/P' in path and '/Table' not in path:
            categories['standalone_text'].append({
                'object_id': object_id,
                'path': path,
                'text': text
            })
        else:
            categories['other'].append({
                'object_id': object_id,
                'path': path,
                'text': text
            })
    
    return categories, table_elements, cell_elements

def print_analysis(categories, table_elements, cell_elements):
    """Print detailed analysis"""
    
    print("=" * 100)
    print("TEXT ELEMENTS ANALYSIS FROM structuredData.json")
    print("=" * 100)
    print()
    
    # Print table structure
    print("📊 TABLE STRUCTURE:")
    print("-" * 50)
    for table in table_elements:
        attrs = table['attributes']
        print(f"Table: {table['path']}")
        print(f"  - ObjectID: {table['object_id']}")
        print(f"  - NumCol: {attrs.get('NumCol', 'N/A')}")
        print(f"  - NumRow: {attrs.get('NumRow', 'N/A')}")
        print(f"  - BBox: {attrs.get('BBox', 'N/A')}")
        print()
    
    # Print cell analysis
    print("📋 CELL ELEMENTS ANALYSIS:")
    print("-" * 50)
    print(f"Total cells found: {len(cell_elements)}")
    
    # Analyze row/column indices
    rows_with_indices = 0
    cols_with_indices = 0
    cells_with_both = 0
    
    for cell in cell_elements:
        attrs = cell['attributes']
        if attrs.get('RowIndex') is not None:
            rows_with_indices += 1
        if attrs.get('ColIndex') is not None:
            cols_with_indices += 1
        if attrs.get('RowIndex') is not None and attrs.get('ColIndex') is not None:
            cells_with_both += 1
    
    print(f"Cells with RowIndex: {rows_with_indices}")
    print(f"Cells with ColIndex: {cols_with_indices}")
    print(f"Cells with both indices: {cells_with_both}")
    print(f"Cells missing indices: {len(cell_elements) - cells_with_both}")
    print()
    
    # Print categories summary
    print("📝 TEXT ELEMENTS BY CATEGORY:")
    print("-" * 50)
    total_elements = 0
    for category, elements in categories.items():
        count = len(elements)
        total_elements += count
        print(f"{category.replace('_', ' ').title()}: {count}")
    print(f"Total text elements: {total_elements}")
    print()
    
    # Print detailed breakdown
    for category, elements in categories.items():
        if elements:
            print(f"\n🔍 {category.replace('_', ' ').upper()}:")
            print("-" * 30)
            for i, elem in enumerate(elements[:5]):  # Show first 5
                text_preview = elem['text'][:50] + "..." if len(elem['text']) > 50 else elem['text']
                print(f"{i+1}. {elem['path']}")
                print(f"   Text: {text_preview}")
                if 'row' in elem and 'col' in elem:
                    print(f"   Position: Row {elem['row']}, Col {elem['col']}")
                print()
            if len(elements) > 5:
                print(f"... and {len(elements) - 5} more elements")

def main():
    """Main function"""
    json_file = "structuredData.json"
    
    try:
        categories, table_elements, cell_elements = analyze_text_elements(json_file)
        print_analysis(categories, table_elements, cell_elements)
        
        # Save detailed analysis to file
        output_file = "text_elements_analysis.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("TEXT ELEMENTS ANALYSIS FROM structuredData.json\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("SUMMARY:\n")
            f.write("-" * 30 + "\n")
            total_elements = sum(len(elements) for elements in categories.values())
            f.write(f"Total text elements: {total_elements}\n")
            f.write(f"Table cells: {len(categories['table_cells'])}\n")
            f.write(f"Table headers: {len(categories['table_headers'])}\n")
            f.write(f"Standalone text: {len(categories['standalone_text'])}\n")
            f.write(f"Titles: {len(categories['titles'])}\n")
            f.write(f"Footnotes: {len(categories['footnotes'])}\n")
            f.write(f"List items: {len(categories['list_items'])}\n")
            f.write(f"Other: {len(categories['other'])}\n\n")
            
            # Write all elements by category
            for category, elements in categories.items():
                if elements:
                    f.write(f"\n{category.upper()}:\n")
                    f.write("-" * 30 + "\n")
                    for elem in elements:
                        f.write(f"Path: {elem['path']}\n")
                        f.write(f"Text: {elem['text']}\n")
                        if 'row' in elem and 'col' in elem:
                            f.write(f"Position: Row {elem['row']}, Col {elem['col']}\n")
                        f.write("\n")
        
        print(f"\nDetailed analysis saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: {json_file} not found!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 