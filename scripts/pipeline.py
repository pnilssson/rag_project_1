import os
from scripts.extract import extract_text
from scripts.chunk import chunk_text
from scripts.embeddings import init_collection, insert_chunks

DATA_DIR = "data"

def process_file(path):
    text = extract_text(path)
    chunks = chunk_text(text)
    insert_chunks(chunks, os.path.basename(path))

def process_folder(folder):
    for file in os.listdir(folder):
        full_path = os.path.join(folder, file)
        if os.path.isfile(full_path):
            try:
                process_file(full_path)
                print(f"✅ Inläst: {file}")
            except Exception as e:
                print(f"⚠️  Fel i {file}: {e}")

if __name__ == "__main__":
    init_collection()
    process_folder(DATA_DIR)
