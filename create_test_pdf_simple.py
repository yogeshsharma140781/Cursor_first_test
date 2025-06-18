from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_test_pdf():
    c = canvas.Canvas("test_simple.pdf", pagesize=letter)
    width, height = letter
    
    # Add some simple text
    c.drawString(100, height - 100, "This is a test document.")
    c.drawString(100, height - 130, "It contains some English text.")
    c.drawString(100, height - 160, "This should be translated properly.")
    c.drawString(100, height - 190, "The formatting should be preserved.")
    
    c.save()
    print("Created test_simple.pdf")

if __name__ == "__main__":
    create_test_pdf() 