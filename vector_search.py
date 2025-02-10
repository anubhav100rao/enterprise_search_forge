import pickle
import numpy as np
import faiss

# Load pre-computed embeddings and document metadata
with open('embeddings.pkl', 'rb') as f:
    embeddings, documents = pickle.load(f)

# Convert embeddings list to a NumPy array
embeddings = np.array(embeddings).astype('float32')

# Build a FAISS index (using inner product or cosine similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def search(query_embedding, top_k=5):
    # Perform the search
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    results = []
    for idx in indices[0]:
        results.append(documents[idx])
    return results

if __name__ == '__main__':
    # Example usage: load the model to embed a query and search
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query = "How do I reset my password?"
    query_embedding = model.encode(query).astype('float32')
    results = search(query_embedding, top_k=3)
    for res in results:
        print(f"From {res['filename']}: {res['content']}\n")
