from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle


MODEL_NAME = "all-MiniLM-L6-v2"


def build_index(chunks, model_name=MODEL_NAME, index_path="faiss_index.bin", meta_path="meta.pkl"):
    model = SentenceTransformer(model_name)
    texts = [c['text'] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # faiss expects float32
    embeddings = embeddings.astype('float32')
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index to {index_path} and metadata to {meta_path}")




if __name__ == '__main__':
    import argparse
    from ingest import ingest_folder

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, help='Folder with PDFs/DOCX to ingest')
    parser.add_argument('--index', default='faiss_index.bin')
    parser.add_argument('--meta', default='meta.pkl')
    args = parser.parse_args()

    chunks = ingest_folder(args.folder)
    build_index(chunks, index_path=args.index, meta_path=args.meta)