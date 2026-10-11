import requests
import os
import json
import pandas as pd


def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list,
        },
        timeout=300,
    )

    if r.status_code != 200:
        print("Ollama error:", r.text)
        r.raise_for_status()

    return r.json()["embeddings"]


jsons = sorted(f for f in os.listdir("jsons") if f.endswith(".json"))
dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)

    print(f"Creating embeddings for {json_file}")

    texts = [c["text"] for c in content["chunks"]]

    embeddings = []

    batch_size = 5

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        print(f"Processing chunks {i + 1} to {i + len(batch)}")

        batch_embeddings = create_embedding(batch)
        embeddings.extend(batch_embeddings)

    for i, chunk in enumerate(content["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk_id += 1
        chunk["embedding"] = embeddings[i]
        dicts.append(chunk)

df = pd.DataFrame.from_records(dicts)
print(df)

os.makedirs("data", exist_ok=True)

df.to_pickle("data/video_embeddings.pkl")

print(f"Saved {len(df)} chunks with embeddings!")
