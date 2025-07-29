import layoutparser as lp
import cv2
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import numpy as np

def convert_pdf_to_image(pdf_path):
    print("Converting PDF to image...")
    # Convert PDF to image
    images = convert_from_path(pdf_path)
    # Get the first page
    first_page = images[0]
    # Convert PIL image to numpy array
    return np.array(first_page)

def main():
    # Load the model
    print("Loading the model...")
    model = lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )
    
    # Process the PDF
    print("Loading the PDF...")
    try:
        # Convert PDF to image
        image = convert_pdf_to_image('sample.pdf')
        
        # Detect layout
        print("Detecting layout...")
        layout = model.detect(image)
        
        # Draw layout on the image
        print("Drawing layout...")
        viz_image = lp.draw_box(image, layout, box_width=3)
        
        # Save the visualization
        plt.figure(figsize=(15, 15))
        plt.imshow(viz_image)
        plt.axis('off')
        plt.savefig('output_layout.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print("Layout detection completed! Check 'output_layout.png' for results.")
        
        # Print detected layout information
        print("\nDetected Layout Elements:")
        for block in layout:
            print(f"Type: {block.type}, Confidence: {block.score:.2f}")
            print(f"Coordinates: {block.block.coordinates}\n")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your PDF file exists and is readable")
        print("2. Ensure you have poppler installed for PDF processing")
        print("3. Ensure the model was downloaded successfully")
        print("4. Check if you have sufficient memory for processing")
        print("\nFor macOS users, if poppler is missing, install it with:")
        print("brew install poppler")

if __name__ == "__main__":
    main() 