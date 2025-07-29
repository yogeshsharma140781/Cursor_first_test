from reportlab.pdfgen import canvas
from reportlab.lib.colors import red, blue, green, black

def create_test_colored_pdf():
    """Create a test PDF with colored text to verify color extraction"""
    c = canvas.Canvas("test_colored.pdf")
    
    # Set page size
    c.setPageSize((612, 792))
    
    # Add colored text
    c.setFont("Helvetica", 12)
    
    # Red text
    c.setFillColor(red)
    c.drawString(50, 750, "This is red text")
    
    # Blue text
    c.setFillColor(blue)
    c.drawString(50, 720, "This is blue text")
    
    # Green text
    c.setFillColor(green)
    c.drawString(50, 690, "This is green text")
    
    # Black text
    c.setFillColor(black)
    c.drawString(50, 660, "This is black text")
    
    # Mixed colors in a paragraph
    c.setFillColor(red)
    c.drawString(50, 600, "Red text ")
    c.setFillColor(blue)
    c.drawString(120, 600, "Blue text ")
    c.setFillColor(green)
    c.drawString(200, 600, "Green text")
    
    c.save()
    print("Created test_colored.pdf with colored text")

if __name__ == "__main__":
    create_test_colored_pdf() 