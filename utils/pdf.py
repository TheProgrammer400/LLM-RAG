import fitz
import os

def read_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text + "\n"

    doc.close()
    return text


if __name__ == "__main__":
    # Test reading one of the actual books
    try:
        text = read_pdf("books/brain_tumour.pdf")
        print(f"Success! Read {len(text)} characters.")
    except Exception as e:
        print(f"Error reading PDF: {e}")