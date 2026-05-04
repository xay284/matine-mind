import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ───────────── CONFIG ─────────────
INPUT_PATH    = r"C:\Users\Chayma MAJJEDI\Desktop\chatbot_web\new pipeline\output.json"
CHROMA_PATH   = r"C:\Users\Chayma MAJJEDI\Desktop\chatbot_web\new pipeline\chroma_data"
EMBED_MODEL   = "intfloat/multilingual-e5-large"
COLLECTION    = "projets"
BATCH_SIZE    = 32

# Sliding window pour le full_text brut
CHUNK_SIZE    = 120
CHUNK_OVERLAP = 40

# ───────────── Chargement ─────────────
print("⏳ Chargement du modèle d'embedding...")
embedder = SentenceTransformer(EMBED_MODEL)

print("📂 Lecture du JSON...")
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    projets = json.load(f)
print(f"✅ {len(projets)} projets chargés\n")

# ───────────── ChromaDB ─────────────
chroma = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    chroma.delete_collection(COLLECTION)
    print("🗑️  Ancienne collection supprimée")
except Exception:
    pass

collection = chroma.create_collection(
    name     = COLLECTION,
    metadata = {"hnsw:space": "cosine"}
)
print(f"✅ Collection '{COLLECTION}' créée\n")

# ───────────── Sliding window ─────────────
def sliding_window(text: str, chunk_size: int, overlap: int) -> list[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

# ──────────────────────────────────────────────────────────────
# build_meta_base — métadonnées communes à tous les chunks
# d'un même projet (stockées dans ChromaDB pour filtrage/affichage)
# ──────────────────────────────────────────────────────────────
def build_meta_base(projet: dict, projet_index: int) -> dict:
    services = projet.get("services", []) or []
    impacts  = projet.get("impacts",  []) or []
    chiffres_cles = projet.get("chiffres_cles", []) or []
    mots_cles     = projet.get("mots_cles",     []) or []
    return {
        # ── Identité ──
        "projet_index" : projet_index,
        "nom_projet"   : projet.get("nom_projet",   "") or "",
        "ref"          : projet.get("ref",          "") or "",
        # ── Géo / temporel ──
        "pays"         : projet.get("pays",         "") or "",
        "annee"        : projet.get("annee",        "") or "",
        # ── Client / secteur ──
        "nom_client"   : projet.get("client",       "") or "",
        "secteur"      : projet.get("secteur",      "") or "",
        "sous_secteur" : projet.get("sous_secteur", "") or "",
        # ── Finance ──
        "valeur": projet.get("budget") or projet.get("valeur") or "",
        # ── Résumés textuels (utiles pour compress_chunk dans rag_query) ──
        "description"  : projet.get("description",  "") or "",
        # Services et impacts : stockés en string "pipe-separated" car
        # ChromaDB n'accepte pas les listes comme valeur de métadonnée
        "services"     : " | ".join(services) if services else "",
        "impacts"      : " | ".join(impacts)  if impacts  else "",
        "chiffres_cles": " | ".join(chiffres_cles) if chiffres_cles else "",  
        "mots_cles"    : " | ".join(mots_cles)     if mots_cles     else "", 

    }

# ──────────────────────────────────────────────────────────────
# build_chunks — construit les chunks selon le schéma de l'image :
#
#   Chunk 0  : IDENTITÉ          → toujours présent
#   Chunk 1  : DESCRIPTION       → si description > 10 mots
#   Chunk 2  : SERVICES          → si liste non vide
#   Chunk 3  : IMPACTS           → si liste non vide
#   Chunk N  : FULL_TEXT (x N)   → sliding window sur full_text brut
#
# Chaque chunk porte le même préfixe identité pour ancrer le contexte.
# ──────────────────────────────────────────────────────────────
def build_chunks(projet: dict, projet_index: int) -> list[dict]:

    nom         = projet.get("nom_projet",   "") or ""
    pays        = projet.get("pays",         "") or ""
    client      = projet.get("client",       "") or ""
    secteur     = projet.get("secteur",      "") or ""
    valeur      = projet.get("budget") or projet.get("valeur") or ""
    annee       = projet.get("annee",        "") or ""
    description = projet.get("description",  "") or ""
    services    = projet.get("services",     []) or []
    impacts     = projet.get("impacts",      []) or []
    chiffres_cles = projet.get("chiffres_cles",[]) or []    
    mots_cles     = projet.get("mots_cles",    []) or [] 
    full_text   = projet.get("full_text",    "") or ""

    meta_base = build_meta_base(projet, projet_index)

    # ── Préfixe identité commun à tous les chunks ──
    # Format compact pour ne pas "polluer" le sens du chunk spécialisé
    id_parts = [f"Projet : {nom}"]
    if pays   : id_parts.append(f"Pays : {pays}")
    if annee  : id_parts.append(f"Année : {annee}")
    if client : id_parts.append(f"Client : {client}")
    if secteur: id_parts.append(f"Secteur : {secteur}")
    if valeur : id_parts.append(f"Budget : {valeur}")
    prefix = " | ".join(id_parts)

    chunks    = []
    chunk_idx = 0

    # ── CHUNK 0 : IDENTITÉ (toujours présent) ──────────────────
    # Contient le préfixe + description courte si disponible.
    # C'est le chunk qui répond aux questions "Qu'est-ce que ce projet ?"
    identity_text = prefix
    if description:
        identity_text += f"\n\nDescription : {description}"

    chunks.append({
        "text"       : identity_text,
        "chunk_type" : "identite",       # label utilisé par rag_query pour le debug
        "chunk_index": chunk_idx,
    })
    chunk_idx += 1

    # ── CHUNK 1 : DESCRIPTION (si > 10 mots) ───────────────────
    # Chunk dédié à la description longue pour maximiser le rappel
    # sur les questions thématiques ("contexte", "objectif du projet").
    if description and len(description.split()) > 10:
        chunks.append({
            "text"       : f"{prefix}\n\nDescription :\n{description}",
            "chunk_type" : "description",
            "chunk_index": chunk_idx,
        })
        chunk_idx += 1

    # ── CHUNK 2 : SERVICES (si liste non vide) ─────────────────
    # Chunk dédié aux activités réalisées.
    # Répond bien aux questions "quelles missions", "quelles activités".
    if len(services) > 5:
        # chunk global + chunks individuels pour les services critiques
        for j, service in enumerate(services[:3]):  # top 3 seulement
            chunks.append({
                "text": f"{prefix}\n\nService clé : {service}",
                "chunk_type": "service_detail",
                "chunk_index": chunk_idx,
            })
            chunk_idx += 1

    # ── CHUNK 3 : IMPACTS (si liste non vide) ──────────────────
    # Chunk dédié aux résultats et impacts concrets.
    # Répond bien aux questions "résultats", "effets", "livrables obtenus".
    if impacts:
        impacts_text = "\n".join(f"• {i}" for i in impacts)
        chunks.append({
            "text"       : f"{prefix}\n\nImpacts et résultats :\n{impacts_text}",
            "chunk_type" : "impacts",
            "chunk_index": chunk_idx,
        })
        chunk_idx += 1
    # ── CHUNK 4 : CHIFFRES CLÉS ────────────────────────────────
    # Chunk dédié aux données quantifiées.
    # Répond aux questions : "combien de participants", "quel volume",
    # "quels chiffres", "projets avec plus de X mesures"...
    if chiffres_cles:
        chiffres_text = "\n".join(f"• {c}" for c in chiffres_cles)
        chunks.append({
            "text"       : f"{prefix}\n\nChiffres clés :\n{chiffres_text}",
            "chunk_type" : "chiffres_cles",
            "chunk_index": chunk_idx,
        })
        chunk_idx += 1

    # ── CHUNK 5 : MOTS CLÉS  ───────────────────────────
    # Chunk dédié aux termes sectoriels, acronymes 
    # Répond aux questions : "projets liés à DEPTH", 
    # "projets avec méthodologie PMO SMART"...
    context_parts = []
    if mots_cles:
        context_parts.append("Domaines et méthodologies : " + ", ".join(mots_cles))
    # ← AJOUTER : répéter les mots_clés sous forme de phrases
    if mots_cles:
        context_parts.append(
            "Ce projet est lié à : " + " et ".join(mots_cles[:6])
        )

    # ← AJOUTER : intégrer aussi secteur et sous-secteur
    secteur = projet.get("secteur", "") or ""
    if secteur:
        context_parts.append(f"Secteur : {secteur}")
    if context_parts:              
        chunks.append({
            "text": f"{prefix}\n\n" + "\n".join(context_parts),
            "chunk_type": "contexte",
            "chunk_index": chunk_idx,
        })
        chunk_idx += 1
    # ── CHUNKS N : FULL_TEXT / sliding window ──────────────────
    # Le texte brut de la fiche est découpé en fenêtres glissantes
    # pour couvrir les détails fins non capturés par les chunks structurés.
    if full_text:
        windows = sliding_window(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, window_text in enumerate(windows):
            chunks.append({
                "text"       : f"{prefix}\n\n{window_text}",
                "chunk_type" : "full_text",
                "chunk_index": chunk_idx + i,
            })
        chunk_idx += len(windows)
    # chunk tout-en-un :
    synthesis_parts = [prefix]
    if mots_cles:
        synthesis_parts.append("Domaines : " + ", ".join(mots_cles[:8]))
    if impacts:
        synthesis_parts.append("Résultats : " + " | ".join(impacts[:3]))
    if chiffres_cles:
        synthesis_parts.append("Chiffres : " + " | ".join(chiffres_cles[:3]))

    chunks.append({
        "text": "\n".join(synthesis_parts),
        "chunk_type": "synthese",
        "chunk_index": chunk_idx,
    })
    chunk_idx += 1
    # ── Fallback : aucune donnée textuelle ─────────────────────
    # Projet présent dans le récap mais sans fiche détaillée.
    # On crée quand même un chunk minimal pour qu'il soit trouvable.
    if len(chunks) == 0:
        chunks.append({
            "text"       : prefix,
            "chunk_type" : "identite_seule",
            "chunk_index": 0,
        })

    # Injecte total_chunks dans chaque chunk (utilisé par rag_query)
    total = len(chunks)
    for c in chunks:
        c["total_chunks"] = total
        c["meta"]         = meta_base   # référence partagée (sera copiée ci-dessous)

    return chunks

# ───────────── Construction de tous les chunks ─────────────
all_chunks = []
for i, projet in enumerate(projets):
    for c in build_chunks(projet, i):
        all_chunks.append(c)

print(f"📦 {len(all_chunks)} chunks générés pour {len(projets)} projets")

# ── Stats par type ──
type_counts: dict[str, int] = {}
for c in all_chunks:
    t = c["chunk_type"]
    type_counts[t] = type_counts.get(t, 0) + 1

print("\n📊 Répartition par type :")
for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"   {t:<20} : {count:>5} chunks")
print(f"   Moyenne  : {len(all_chunks)/max(len(projets),1):.1f} chunks/projet\n")

# ── Qualité des données ──
print("📋 Qualité des données source :")
for field in ["full_text", "annee",
              "description", "services", "impacts",
              "chiffres_cles", "mots_cles"]:
    count = sum(1 for p in projets if p.get(field))
    print(f"   {field:<16} : {count}/{len(projets)}")
print()

# ───────────── Préparation pour ChromaDB ─────────────
# IMPORTANT : ChromaDB n'accepte que str / int / float / bool comme
# valeurs de métadonnées — pas de listes, pas de None.
ids        = []
documents  = []
metadatas  = []

for chunk in all_chunks:
    proj_idx  = chunk["meta"]["projet_index"]
    c_type    = chunk["chunk_type"]
    c_idx     = chunk["chunk_index"]

    # ID unique et lisible : proj0042_services_002
    chunk_id  = f"proj{proj_idx:04d}_{c_type}_{c_idx:03d}"

    meta = {
        **chunk["meta"],            # tous les champs de meta_base
        "chunk_index"  : c_idx,
        "total_chunks" : chunk["total_chunks"],
        "chunk_type"   : c_type,
        # NOTE : pas de champ "section" — il n'existe pas dans build_chunks
    }
    # Sécurité : remplace les None par "" pour ChromaDB
    meta = {k: (v if v is not None else "") for k, v in meta.items()}

    ids.append(chunk_id)
    documents.append(chunk["text"])
    metadatas.append(meta)

print(f"✅ {len(ids)} chunks préparés\n")

# ───────────── Embedding par batch ─────────────
embeddings = []
print("⚡ Embedding en cours...")
for start in tqdm(range(0, len(documents), BATCH_SIZE)):
    end   = min(start + BATCH_SIZE, len(documents))
    # Préfixe "passage:" requis par multilingual-e5 pour l'indexation
    batch = [f"passage: {doc}" for doc in documents[start:end]]
    vecs  = embedder.encode(batch, normalize_embeddings=True)
    embeddings.extend(vecs.tolist())

print(f"\n✅ {len(embeddings)} embeddings générés\n")

# ───────────── Insertion ChromaDB ─────────────
print("💾 Insertion dans ChromaDB...")
for start in tqdm(range(0, len(ids), BATCH_SIZE)):
    end = min(start + BATCH_SIZE, len(ids))
    collection.add(
        ids        = ids[start:end],
        documents  = documents[start:end],
        embeddings = embeddings[start:end],
        metadatas  = metadatas[start:end],
    )

print(f"\n✅ ChromaDB remplie : {collection.count()} chunks stockés")

# ───────────── Test de recherche ─────────────
print("\n=== TEST DE RECHERCHE ===")
test_queries = [
    "étude de marché compétences numériques enseignement supérieur",
    "nombre de participants mesures approuvées conseil des ministres",   # ← teste chiffres_cles
    "PMO SMART Delivery Unit dialogue public-privé",                     # ← teste mots_cles
]
for test_query in test_queries:
    test_vec = embedder.encode(
        [f"query: {test_query}"],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
            query_embeddings = [test_vec],
            n_results        = 3,
            include          = ["documents", "distances", "metadatas"]
    )

    print(f"\nQuery : '{test_query}'")
    for i, (doc, dist, meta) in enumerate(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ), 1):
        print(
            f"  [{i}] score={1-dist:.3f} | "
            f"type={meta.get('chunk_type','?'):<16} | "
            f"année={meta.get('annee','?'):<6} | "
            f"{meta.get('nom_projet','?')[:45]}"
        )

print(f"\n✅ Indexation terminée — {collection.count()} chunks dans ChromaDB")
