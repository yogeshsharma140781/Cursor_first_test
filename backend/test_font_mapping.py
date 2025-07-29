import json
from adobe_openai_complete_workflow import PDFTranslationWorkflow

def test_font_mapping():
    data = json.load(open('structuredData.json'))
    elements = [e for e in data['elements'] if 'Text' in e and e['Text'].strip()]
    
    # Create workflow instance to test font mapping
    workflow = PDFTranslationWorkflow('pdfservices-api-credentials.json', 'dummy')
    
    print('FONT MAPPING TEST:')
    print('=' * 60)
    
    # Test elements with different weights
    test_elements = []
    for e in elements:
        if 'Font' in e and e['Font']['weight'] != 400:  # Non-normal weights
            test_elements.append(e)
    
    for i, e in enumerate(test_elements[:10]):  # Test first 10
        font_info = e['Font']
        original_font = font_info['family_name']
        weight = font_info['weight']
        style = font_info.get('style', 'normal')
        
        mapped_font = workflow._map_font_family(original_font, weight, style)
        
        print(f'{i+1:2d}. Weight: {weight:3d} | Style: {style:6s} | Original: {original_font:15s} | Mapped: {mapped_font}')
        print(f'    Text: "{e["Text"][:50]}..."')
        print()

if __name__ == "__main__":
    test_font_mapping() 