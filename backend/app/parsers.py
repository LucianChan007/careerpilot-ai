from pathlib import Path
import fitz
from docx import Document

def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as doc:
            return "\n".join(page.get_text("text") for page in doc).strip()
    if path.suffix.lower() == ".docx":
        doc = Document(path)
        return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip()).strip()
    raise ValueError("Unsupported file type")
