from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.units import inch

def create_test_pdf():
    c = canvas.Canvas("sample.pdf", pagesize=letter)
    width, height = letter
    
    # Add a title with background
    c.setFillColor(colors.lightgrey)
    c.rect(50, height-100, width-100, 50, fill=True)
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, height-70, "Test Document")
    
    # Add some regular text
    c.setFont("Helvetica", 12)
    c.drawString(100, height-120, "This is a sample PDF document for testing layout parsing.")
    
    # Add a list with bullets
    y = height-170
    bullets = ["•", "○", "▪"]
    items = [
        "First item with some longer text that might wrap to the next line",
        "Second item with numbers 123456789",
        "Third item with special chars @#$%^&*()"
    ]
    for i, item in enumerate(items):
        bullet = bullets[i % len(bullets)]
        c.drawString(100, y, bullet)
        c.drawString(120, y, item)
        y -= 30
    
    # Add a table with borders
    data = [
        ['Header 1', 'Header 2', 'Header 3'],
        ['Data 1', 'Data 2', 'Data 3'],
        ['More Data', 'Values', 'Numbers']
    ]
    
    table = Table(data, colWidths=[2*inch]*3)
    table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 14),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 12),
        ('TOPPADDING', (0,1), (-1,-1), 12),
    ]))
    
    table.wrapOn(c, width, height)
    table.drawOn(c, 100, y-200)
    
    # Add some Dutch text with different styles
    y -= 350
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, y, "Nederlandse Sectie")
    y -= 30
    
    c.setFont("Helvetica", 12)
    dutch_text = [
        "Dit is een voorbeeldtekst in het Nederlands.",
        "SEPA Machtiging: NL83INGB0000000000",
        "Telefoonnummer: +31 20 123 4567",
        "E-mailadres: test@example.nl",
        "KvK nummer: 12345678"
    ]
    
    for line in dutch_text:
        c.drawString(100, y, line)
        y -= 20
    
    # Add a section with mixed content
    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, y, "Mixed Content Section")
    y -= 30
    
    mixed_content = [
        "Text with numbers: 12.34 € 56,78",
        "Special characters: ©®™ §¶†‡",
        "Date formats: 31-12-2023 | 2023/12/31",
        "Time formats: 13:45 | 1:45 PM",
        "URLs: https://www.example.com"
    ]
    
    c.setFont("Helvetica", 12)
    for line in mixed_content:
        c.drawString(100, y, line)
        y -= 20
    
    # Save the PDF
    c.save()

if __name__ == "__main__":
    create_test_pdf() 