#========================================
#Read every page & extract text and keep text,page number,paper name
#=========================================
import pymupdf
from pathlib import Path


def load_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")

        if text.strip():
            pages.append({
                "text": text,
                "page": page_number,
                "source": Path(pdf_path).name
            })

    return pages


def load_all_papers(folder="data/papers"):
    all_pages = []

    for pdf_path in Path(folder).glob("*.pdf"):
        pages = load_pdf(pdf_path)
        all_pages.extend(pages)

    return all_pages

#======================Chunking==========================
def create_chunks(pages, chunk_size=500, overlap=50):
    assert overlap < chunk_size, "overlap must be smaller than chunk_size"
    step = chunk_size - overlap
    chunks = []

    for page in pages:
        text = " ".join(page["text"].split())   # remove newlines / extra spaces

        for i, start in enumerate(range(0, len(text), step)):
            piece = text[start:start + chunk_size]
            if len(piece) < 100:                # skip tiny tails
                continue
            chunks.append({
                "text": piece,
                "source": page["source"],
                "page": page["page"],
                "chunk_id": f'{page["source"]}_p{page["page"]}_c{i}',
            })
        
    return chunks