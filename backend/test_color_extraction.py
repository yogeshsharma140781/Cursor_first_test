import json
from adobe_openai_complete_workflow import PDFTranslationWorkflow
import fitz

def test_color_extraction():
    print('COLOR EXTRACTION TEST WITH COLORED PDF:')
    print('=' * 50)
    
    # Test with the colored PDF first
    print("Testing with test_colored.pdf...")
    
    # Open the colored PDF directly with PyMuPDF
    doc = fitz.open('test_colored.pdf')
    page = doc[0]
    
    # Get text with color information
    text_dict = page.get_text("dict")
    
    print("Color information from test_colored.pdf:")
    for block in text_dict.get("blocks", []):
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span.get("text", "")[:30]
                    color = span.get("color", 0)
                    print(f"  Text: '{text}' Color: {color}")
    
    doc.close()
    
    print("\n" + "="*50)
    print("Testing with sample2.pdf (original):")
    
    # Load the extracted data
    data = json.load(open('structuredData.json'))
    elements = [e for e in data['elements'] if 'Text' in e and e['Text'].strip()]
    
    # Parse text elements
    workflow = PDFTranslationWorkflow('pdfservices-api-credentials.json', 'dummy')
    text_elements = workflow.parse_text_elements(data)
    
    # Extract color information
    text_elements_with_color = workflow.extract_color_information('sample2.pdf', text_elements)
    
    # Show some examples
    print('Color information examples:')
    for i, element in enumerate(text_elements_with_color[:5]):
        color = element.get('font_color', [0, 0, 0])
        print(f'{i+1:2d}. Text: "{element["text"][:30]}..."')
        print(f'    Color: RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})')
        print()

if __name__ == "__main__":
    test_color_extraction() 