import os
import openai
from vector_search import search
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_answer(query):
    # Embed the query
    query_embedding = model.encode(query).astype('float32')
    # Retrieve the top 3 relevant document chunks
    results = search(query_embedding, top_k=3)

    # Combine the retrieved texts as context
    context = "\n\n".join([res['content'] for res in results])

    # Build the prompt
    prompt = f"Answer the following question based on the context provided.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    # Generate a response using the OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.5,
    )
    answer = response.choices[0].text.strip()
    return answer


if __name__ == '__main__':
    query = "How do I reset my password?"
    answer = generate_answer(query)
    print("Answer:", answer)
