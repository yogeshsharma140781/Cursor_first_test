from adobe_openai_complete_workflow import PDFTranslationWorkflow

def test_colored_pdf_workflow():
    """Test the complete workflow with the colored test PDF"""
    print("TESTING COMPLETE WORKFLOW WITH COLORED PDF")
    print("=" * 60)
    
    # Initialize workflow
    workflow = PDFTranslationWorkflow('pdfservices-api-credentials.json', 'dummy')
    
    # Run extraction on the colored test PDF
    print("Extracting from test_colored.pdf...")
    structured_data = workflow.extract_pdf_elements('test_colored.pdf')
    
    # Parse text elements
    text_elements = workflow.parse_text_elements(structured_data)
    print(f"Parsed {len(text_elements)} text elements")
    
    # Extract color information
    text_elements_with_color = workflow.extract_color_information('test_colored.pdf', text_elements)
    
    # Show color information
    print("\nCOLOR INFORMATION FROM COLORED PDF:")
    print("-" * 50)
    for i, element in enumerate(text_elements_with_color):
        color = element.get('font_color', [0, 0, 0])
        print(f"{i+1:2d}. Text: '{element['text']}'")
        print(f"    Color: RGB({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")
        print()
    
    # Test reconstruction with original text
    print("TESTING RECONSTRUCTION WITH COLORS:")
    print("-" * 40)
    
    # Use original text as "translated" text for testing
    original_texts = [elem['text'] for elem in text_elements_with_color]
    
    # Reconstruct PDF
    output_path = "test_colored_reconstruction.pdf"
    result = workflow.reconstruct_pdf(text_elements_with_color, original_texts, output_path)
    
    if result:
        print(f"✅ Colored PDF reconstruction test successful: {output_path}")
    else:
        print("❌ Colored PDF reconstruction test failed")

if __name__ == "__main__":
    test_colored_pdf_workflow() 