from pathlib import Path
from typing import List, Dict
import pdfplumber
import docx

def extract_text_from_pdf(path: Path) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            texts.append(p.extract_text() or "")
    return "\n".join(texts)

def extract_text_from_docx(path: Path) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

# Simple character-based chunker with overlap
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[Dict]:
    chunks = []
    start = 0
    n = len(text)
    i = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"id": i, "text": chunk, "start": start, "end": end})
            i += 1
        start += chunk_size - overlap
    return chunks

def ingest_folder(folder_path: str) -> List[Dict]:
    p = Path(folder_path)
    all_chunks = []
    for file in p.iterdir():
        if file.suffix.lower() == '.pdf':
            text = extract_text_from_pdf(file)
        elif file.suffix.lower() == '.docx':
            text = extract_text_from_docx(file)
        else:
            continue
        chunks = chunk_text(text)
        for c in chunks:
            c['source'] = file.name
        all_chunks.extend(chunks)
    return all_chunks