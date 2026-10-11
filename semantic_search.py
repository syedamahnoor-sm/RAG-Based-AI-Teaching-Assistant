import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        },
        timeout=300
    )

    if r.status_code != 200:
        print("Ollama error:", r.text)

    r.raise_for_status()
    return r.json()["embeddings"]


# Load previously generated embeddings
df = pd.read_pickle("data/video_embeddings.pkl")

# Convert stored embeddings into a NumPy matrix
embedding_matrix = np.vstack(df["embedding"].to_numpy())

print(f"Loaded {len(df)} video chunks successfully!")


# Ask the user a question
incoming_query = input("\nAsk a Question: ").strip()

if incoming_query:

    # Generate embedding for the question
    question_embedding = create_embedding([incoming_query])[0]

    # Calculate cosine similarity
    similarities = cosine_similarity(
        embedding_matrix,
        [question_embedding]
    ).flatten()

    # Retrieve top 3 matching chunks
    top_results = 3
    max_indx = similarities.argsort()[::-1][:top_results]

    # Get matching rows
    new_df = df.iloc[max_indx].copy()

    # Add similarity scores
    new_df["similarity_score"] = similarities[max_indx]

    print("\nTop Relevant Video Chunks:\n")

    print(
        new_df[
            ["title", "number", "start", "end", "text", "similarity_score"]
        ].to_string(index=False)
    )

else:
    print("Please enter a valid question.")
