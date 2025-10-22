from sentence_transformers import SentenceTransformer
import faiss
import pickle
# from search import Retriever


MODEL_NAME = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, index_path='faiss_index.bin', meta_path='meta.pkl', model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)


    def retrieve(self, query: str, top_k: int = 5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb.astype('float32')
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.meta[idx]
            results.append({"score": float(dist), "text": meta['text'], "meta": meta})
        return results

def search_index(query, top_k=3, index_path='faiss_index.bin', meta_path='meta.pkl'):
    retriever = Retriever(index_path, meta_path)
    results = retriever.retrieve(query, top_k=top_k)
    # Return only the text (simpler for display)
    return [r['text'] for r in results]


if __name__ == '__main__':
    r = Retriever()
    q = input('Query: ')
    res = r.retrieve(q, top_k=5)
    for i, r in enumerate(res, 1):
        print('---')
        print(i, r['score'], r['meta'].get('source'))
        print(r['text'])