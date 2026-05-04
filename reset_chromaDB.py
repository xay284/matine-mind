import chromadb
import json
from sentence_transformers import SentenceTransformer

# Charger chunks
with open('chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
embeddings = embedder.encode(chunks)

# Supprimer ancienne collection
client = chromadb.PersistentClient(path="./chroma_data")
try:
    client.delete_collection("projets")
    print("✅ Ancienne collection supprimée")
except:
    pass

# Créer nouvelle avec COSINE
collection = client.get_or_create_collection(
    "projets",
    metadata={"hnsw:space": "cosine"}
)

# Ré-ajouter les chunks
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    collection.add(
        ids=[str(i)],
        embeddings=[embedding.tolist()],
        documents=[chunk]
    )

print(f"✅ {len(chunks)} chunks ajoutés avec Cosine Similarity")