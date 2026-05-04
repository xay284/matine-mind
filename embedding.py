import json
from sentence_transformers import SentenceTransformer
import chromadb

# ──────────────────────────────────────────────
# CHARGEMENT
# ──────────────────────────────────────────────
with open('chunks.json', 'r', encoding='utf-8') as f:
    all_chunks = json.load(f)

print(f"✅ {len(all_chunks)} chunks chargés")

# Statistiques rapides
projets_uniques = set(c["nom_projet"] for c in all_chunks)
print(f"✅ {len(projets_uniques)} projets uniques")
print(f"✅ Moyenne : {len(all_chunks)/len(projets_uniques):.1f} chunks par projet\n")

# ──────────────────────────────────────────────
# EMBEDDINGS
# ──────────────────────────────────────────────
embedder   = SentenceTransformer('intfloat/multilingual-e5-large')
texts      = [f"passage: {c['text']}" for c in all_chunks]

print("⏳ Création des embeddings...")
embeddings = embedder.encode(texts, show_progress_bar=True)
print(f"✅ {len(embeddings)} embeddings créés\n")

# ──────────────────────────────────────────────
# INGESTION CHROMADB — une seule collection "projets"
# ──────────────────────────────────────────────
client = chromadb.PersistentClient(path="./chroma_data")

try:
    client.delete_collection("projets")
    print("🗑️  Ancienne collection supprimée")
except Exception:
    pass

collection = client.get_or_create_collection(
    "projets",
    metadata={"hnsw:space": "cosine"}
)

print("\n📥 Ingestion en cours...\n")
for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
    # Métadonnées stockées dans ChromaDB
    meta = {
        "nom_projet":   chunk["nom_projet"],
        "pays":         chunk["pays"],
        "nom_client":   chunk["nom_client"],
        "secteur":      chunk["secteur"],
        "valeur":       chunk["valeur"],
        "chunk_index":  chunk["chunk_index"],
        "total_chunks": chunk["total_chunks"],
    }
    collection.add(
        ids=[str(i)],
        embeddings=[embedding.tolist()],
        documents=[chunk["text"]],
        metadatas=[meta]
    )
    if (i + 1) % 50 == 0 or i == 0:
        print(
            f"  [{i+1:04d}/{len(all_chunks)}] "
            f"{chunk['nom_projet'][:40]:<40} "
            f"| chunk {chunk['chunk_index']+1}/{chunk['total_chunks']}"
        )

print(f"\n✅ {len(all_chunks)} chunks ingérés dans ChromaDB")
print(f"✅ Collection 'projets' — {collection.count()} entrées")

# ──────────────────────────────────────────────
# TEST RAPIDE
# ──────────────────────────────────────────────
print("\n🧪 TEST RAPIDE\n")

questions_test = [
    "étude de marché compétences numériques",
    "projets en Tunisie secteur enseignement",
    "formation certifiante emploi des jeunes",
]

for question in questions_test:
    q_vec   = embedder.encode([f"query: {question}"])[0].tolist()
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=3,
        include=["metadatas", "distances"]
    )
    print(f"❓ {question}")
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        print(
            f"   score={1-dist:.3f} | "
            f"{meta['nom_projet'][:45]:<45} | "
            f"chunk {meta['chunk_index']+1}/{meta['total_chunks']}"
        )
    print()