from PIL import Image
from pdf2image import convert_from_path

def pdf_page_to_svg(pdf_path, page_number, output_path):
    # Convert PDF page to an image
    pages = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    
    if not pages:
        raise ValueError(f"No pages found in the PDF at {pdf_path}")
    
    image = pages[0]
        
    # Convert the image to a format suitable for embedding in an SVG
    image_data = image.convert("RGB")
    image_data.save(output_path, format='PNG')

pdf_path = 'random_augmentation_clean.pdf'
output_path = 'random_augmentation.png'
pdf_page_to_svg(pdf_path, 1, output_path)