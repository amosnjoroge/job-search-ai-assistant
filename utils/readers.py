from PyPDF2 import PdfReader


def read_pdf(file_path: str) -> str:
    """
    Read text content from a PDF file.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        str: Extracted text content from the PDF

    Raises:
        FileNotFoundError: If the PDF file does not exist
        ValueError: If the file is not a valid PDF
        Exception: For other PDF processing errors
    """
    try:
        if not file_path:
            raise ValueError("File path cannot be empty")

        reader = PdfReader(file_path)
        if not reader.pages:
            return ""

        text = ""
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e: 
                print(f"Warning: Error extracting text from page: {str(e)}")
                continue

        return text.replace("\n", " ").strip()

    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")
    except Exception as e:
        raise ValueError(f"Error processing PDF file: {str(e)}")


