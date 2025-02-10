import os
from sentence_transformers import SentenceTransformer
import pickle

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Directory containing your text documents
DOCS_DIR = 'docs'
embeddings = []
documents = []

# Process each document file
for filename in os.listdir(DOCS_DIR):
    if filename.endswith('.txt'):
        with open(os.path.join(DOCS_DIR, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            # For large documents, you might want to split text into chunks
            chunks = text.split('\n\n')  # simple split by paragraphs
            for chunk in chunks:
                if chunk.strip():
                    documents.append({
                        'filename': filename,
                        'content': chunk.strip()
                    })
                    embeddings.append(model.encode(chunk.strip()))

# Save the embeddings and document metadata
with open('embeddings.pkl', 'wb') as f:
    pickle.dump((embeddings, documents), f)

print(f"Processed {len(documents)} text chunks from documents.")
