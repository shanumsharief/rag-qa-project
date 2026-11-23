import json
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")
MAX_BOOKS = 3   # start small; set to None to use all books

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

pdf_files = sorted(DATA_DIR.glob("*.pdf"))
if MAX_BOOKS:
    pdf_files = pdf_files[:MAX_BOOKS]

chunks = []
for pdf_path in pdf_files:
    reader = PdfReader(pdf_path)
    book_chunks = 0
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        for piece in splitter.split_text(text):
            if len(piece.strip()) < 100:  # skip tiny fragments
                continue
            chunks.append({
                "source": pdf_path.stem,
                "page": i + 1,
                "text": piece,
            })
            book_chunks += 1
    print(f"{pdf_path.name}: {book_chunks} chunks")

print(f"\nTotal: {len(chunks)} chunks from {len(pdf_files)} books")

# Save so Day 3 doesn't have to reload the PDFs
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f)

# Eyeball a few
for c in chunks[100:103]:
    print("\n---", c["source"], "| page", c["page"], "---")
    print(c["text"])
