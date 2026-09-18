from preprocessing.text_preprocessing import preprocess_text
from pdf_processing.pdf_extractor import extract_text_from_pdf


# ============================================================
# TEXT INPUT
# ============================================================

def handle_text_input(text):

    if not isinstance(text, str):
        raise TypeError(
            "Text input must be a string."
        )

    if not text.strip():
        raise ValueError(
            "Text input cannot be empty."
        )

    return preprocess_text(text)


# ============================================================
# PDF INPUT
# ============================================================

def handle_pdf_input(pdf_path):

    pages = extract_text_from_pdf(
        pdf_path
    )

    if not pages:
        raise ValueError(
            "No text could be extracted from the PDF."
        )

    combined_text = ""

    for page in pages:

        original_text = page.get(
            "original_text",
            ""
        )

        if original_text:
            combined_text += (
                original_text + " "
            )

    combined_text = combined_text.strip()

    if not combined_text:
        raise ValueError(
            "PDF does not contain extractable text."
        )

    return preprocess_text(
        combined_text
    )


# ============================================================
# MAIN INPUT HANDLER
# ============================================================

def handle_input(
    input_type,
    input_data
):

    if input_type == "text":

        return handle_text_input(
            input_data
        )

    elif input_type == "pdf":

        return handle_pdf_input(
            input_data
        )

    else:

        raise ValueError(
            "Invalid input type. "
            "Use 'text' or 'pdf'."
        )