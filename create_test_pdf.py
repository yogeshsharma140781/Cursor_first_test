#!/usr/bin/env python3

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import io

def create_test_pdf():
    """Create a test PDF with readable English text"""
    
    # Create a BytesIO buffer
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    normal_style = styles['Normal']
    
    # Create content
    story = []
    
    # Title
    title = Paragraph("Sample Document for Translation", title_style)
    story.append(title)
    story.append(Spacer(1, 0.5 * inch))
    
    # Sample paragraphs
    paragraphs = [
        "Welcome to our translation service. This is a sample document created to test the PDF translation functionality.",
        
        "This document contains multiple paragraphs of text that should be easily readable and translatable. The text is formatted properly and should extract cleanly from the PDF.",
        
        "Our translation service supports multiple languages and can handle various types of documents. We use advanced AI technology to provide accurate translations while preserving the document structure.",
        
        "This paragraph contains some technical terms like 'artificial intelligence', 'machine learning', and 'natural language processing' to test how well the translation handles specialized vocabulary.",
        
        "The final paragraph serves as a conclusion to our test document. It demonstrates that the PDF contains substantial, meaningful content that can be effectively translated into other languages."
    ]
    
    for para_text in paragraphs:
        para = Paragraph(para_text, normal_style)
        story.append(para)
        story.append(Spacer(1, 0.3 * inch))
    
    # Build the PDF
    doc.build(story)
    
    # Get the PDF content
    pdf_content = buffer.getvalue()
    buffer.close()
    
    return pdf_content

def create_simple_test_pdf():
    """Create a simple test PDF using canvas"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "Test Document for Translation")
    
    # Add content
    c.setFont("Helvetica", 12)
    y_position = 700
    
    text_lines = [
        "This is a simple test document created for PDF translation testing.",
        "",
        "The document contains readable English text that should be easy to extract",
        "and translate into other languages.",
        "",
        "Key features of this test:",
        "- Simple, clear text formatting",
        "- Multiple paragraphs for testing",
        "- Standard fonts and layout",
        "- Proper text encoding",
        "",
        "This text should be successfully extracted by PyPDF2 and translated",
        "by the OpenAI API, then formatted into a new PDF using ReportLab.",
        "",
        "End of test document."
    ]
    
    for line in text_lines:
        c.drawString(50, y_position, line)
        y_position -= 20
        
        if y_position < 50:  # Start new page if needed
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = 750
    
    c.save()
    pdf_content = buffer.getvalue()
    buffer.close()
    
    return pdf_content

if __name__ == "__main__":
    print("Creating test PDF files...")
    
    # Create advanced test PDF
    try:
        pdf_content = create_test_pdf()
        with open("test_pdfs/sample_english_advanced.pdf", "wb") as f:
            f.write(pdf_content)
        print("✓ Created advanced test PDF: test_pdfs/sample_english_advanced.pdf")
    except Exception as e:
        print(f"✗ Failed to create advanced PDF: {e}")
    
    # Create simple test PDF
    try:
        pdf_content = create_simple_test_pdf()
        with open("test_pdfs/sample_english_simple.pdf", "wb") as f:
            f.write(pdf_content)
        print("✓ Created simple test PDF: test_pdfs/sample_english_simple.pdf")
    except Exception as e:
        print(f"✗ Failed to create simple PDF: {e}")
    
    print("\nTest PDFs created successfully!")
    print("You can now test the translation API with these files.") 