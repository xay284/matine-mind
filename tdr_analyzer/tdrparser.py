# tdr_analyzer/parser.py
import fitz  # pymupdf
from docx import Document
from pathlib import Path


def parse_pdf(file_path: str) -> list[dict]:
    """Retourne une liste de {page: int, text: str}"""
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def parse_docx(file_path: str) -> list[dict]:
    """Retourne une liste de {page: int, text: str} (approximatif pour docx)"""
    doc = Document(file_path)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    # DOCX n'a pas de pages réelles → on découpe par blocs de 3000 chars
    chunks = [full_text[i:i+3000] for i in range(0, len(full_text), 3000)]
    return [{"page": i + 1, "text": chunk} for i, chunk in enumerate(chunks)]


def parse(file_path: str) -> list[dict]:
    """Point d'entrée unique"""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    else:
        raise ValueError(f"Format non supporté : {ext}")
