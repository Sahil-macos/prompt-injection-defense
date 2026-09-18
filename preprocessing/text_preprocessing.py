import re


def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    text = text.lower()

    return text


if __name__ == "__main__":
    text = input("Enter text: ")

    processed_text = preprocess_text(text)

    print("\nOriginal text:")
    print(text)

    print("\nPreprocessed text:")
    print(processed_text)