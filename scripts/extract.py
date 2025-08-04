import fitz
import pytesseract
from PIL import Image
from lxml import etree

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_png(path):
    image = Image.open(path)
    return pytesseract.image_to_string(image)

def extract_text_from_xml(path):
    tree = etree.parse(path)
    return etree.tostring(tree, pretty_print=True).decode()

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text(path):
    ext = path.lower().split(".")[-1]
    if ext == "pdf":
        return extract_text_from_pdf(path)
    elif ext == "png":
        return extract_text_from_png(path)
    elif ext == "xml":
        return extract_text_from_xml(path)
    elif ext == "txt":
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Okänt filformat: {ext}")
