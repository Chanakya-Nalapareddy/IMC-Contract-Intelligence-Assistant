from pathlib import Path
from typing import Iterator, Optional

from pypdf import PdfReader
import docx
from tqdm import tqdm

def extract_text_pdf_iter(path: str, start_page: int = 0) -> Iterator[str]:
    reader = PdfReader(path)
    total_pages = len(reader.pages)

    if start_page < 0 or start_page > total_pages:
        raise ValueError(f"start_page out of range: {start_page} (total_pages={total_pages})")

    for i in tqdm(range(start_page, total_pages), desc="Extracting PDF pages", unit="page"):
        page = reader.pages[i]
        yield page.extract_text() or ""

def extract_text_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join([p.text for p in d.paragraphs])

def extract_text_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")
