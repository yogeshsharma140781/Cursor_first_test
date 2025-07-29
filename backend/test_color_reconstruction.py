import json
from adobe_openai_complete_workflow import PDFTranslationWorkflow

def test_color_reconstruction():
    """Test color extraction and reconstruction without translation"""
    print("TESTING COLOR EXTRACTION AND RECONSTRUCTION")
    print("=" * 60)
    
    # Load the extracted data
    data = json.load(open('structuredData.json'))
    
    # Parse text elements
    workflow = PDFTranslationWorkflow('pdfservices-api-credentials.json', 'dummy')
    text_elements = workflow.parse_text_elements(data)
    
    print(f"Parsed {len(text_elements)} text elements")
    
    # Extract color information
    text_elements_with_color = workflow.extract_color_information('sample2.pdf', text_elements)
    
    # Show color information
    print("\nCOLOR INFORMATION:")
    print("-" * 40)
    for i, element in enumerate(text_elements_with_color[:10]):
        color = element.get('font_color', [0, 0, 0])
        print(f"{i+1:2d}. Text: '{element['text'][:40]}...'")
        print(f"    Color: RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
        print()
    
    # Test reconstruction with original text (no translation)
    print("TESTING RECONSTRUCTION WITH COLORS:")
    print("-" * 40)
    
    # Use original text as "translated" text for testing
    original_texts = [elem['text'] for elem in text_elements_with_color]
    
    # Reconstruct PDF
    output_path = "test_color_reconstruction.pdf"
    result = workflow.reconstruct_pdf(text_elements_with_color, original_texts, output_path)
    
    if result:
        print(f"✅ Color reconstruction test successful: {output_path}")
    else:
        print("❌ Color reconstruction test failed")

if __name__ == "__main__":
    test_color_reconstruction() 