import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypdf import PdfReader
from preprocessing.text_preprocessing import preprocess_text


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        original_text = page.extract_text() or ""

        processed_text = preprocess_text(original_text)

        pages.append({
            "page": page_number,
            "original_text": original_text,
            "processed_text": processed_text
        })

    return pages


if __name__ == "__main__":
    pdf_path = "/Users/sahilbaliyan/Documents/prompt injection/sample_security_report.pdf"

    pages = extract_text_from_pdf(pdf_path)

    for page in pages:
        print(f"\n--- Page {page['page']} ---")

        print("\nOriginal text:")
        print(page["original_text"])

        print("\nPreprocessed text:")
        print(page["processed_text"])