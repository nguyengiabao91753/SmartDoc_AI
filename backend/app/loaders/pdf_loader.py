import pdfplumber
from typing import List, Dict
from app.core.logger import LOG
import os

def load_pdf_from_path(path: str) -> List[Dict]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page": i, "text": text})
    LOG.info("Loaded %d pages from %s", len(pages), os.path.basename(path))
    return pages