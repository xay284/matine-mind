# Script à lancer séparément
import chromadb

chroma = chromadb.PersistentClient(path=r"C:\Users\Chayma MAJJEDI\Desktop\chatbot_web\new pipeline\chroma_data")
collection = chroma.get_collection("projets")

# 1. Combien de projets total ?
all_metas = collection.get(include=["metadatas"])["metadatas"]
projets_uniques = set(m.get("nom_projet", "") for m in all_metas)
print(f"Total projets : {len(projets_uniques)}")
print(f"Total chunks  : {len(all_metas)}")

# 2. Quels secteurs sont présents ?
from collections import Counter
secteurs = Counter(m.get("secteur", "N/A") for m in all_metas)
print("\nSecteurs en base :")
for s, count in secteurs.most_common():
    print(f"  '{s}' → {count} chunks")

# 3. Combien de projets Services Financiers ?
fin = [m for m in all_metas if "financ" in (m.get("secteur","") or "").lower()]
projets_fin = set(m.get("nom_projet","") for m in fin)
print(f"\nProjets Services Financiers : {len(projets_fin)}")
for p in projets_fin:
    print(f"  - {p[:70]}")