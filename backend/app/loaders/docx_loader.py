# from docx import Document
# from typing import List, Dict
# from app.core.logger import LOG

# def load_docx_from_path(path: str):
#     doc = Document(path)
#     texts = []
#     for i, p in enumerate(doc.paragraphs, start=1):
#         text = p.text.strip()
#         if text:
#             texts.append({"paragraph_index": i, "text": text})
#     LOG.info("Loaded %d paragraphs from %s", len(texts), path)
#     return texts