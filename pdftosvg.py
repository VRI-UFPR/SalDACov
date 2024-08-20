from pdf2image import convert_from_path
from PIL import Image
import svgwrite

def pdf_page_to_svg(pdf_path, page_number, svg_output_path):
    # Convert PDF page to an image
    pages = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    
    if not pages:
        raise ValueError(f"No pages found in the PDF at {pdf_path}")
    
    image = pages[0]
    
    # Create a new SVG drawing
    dwg = svgwrite.Drawing(svg_output_path, profile='full')
    
    # Convert the image to a format suitable for embedding in an SVG
    image_data = image.convert("RGB")
    image_data.save('temp.png', format='PNG')
    dwg.add(dwg.image('temp.png'))
    
    # Save the SVG file
    dwg.save()

pdf_path = 'a.pdf'
svg_output_path = 'output.svg'
pdf_page_to_svg(pdf_path, 1, svg_output_path)
