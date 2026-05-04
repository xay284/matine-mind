from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import re

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DOCX_PATH    = r"C:\Users\Chayma MAJJEDI\Desktop\chatbot_web\new pipeline\projets1.docx"
CHUNK_SIZE   = 400   # taille d'un sous-chunk en caractères
CHUNK_OVERLAP = 80   # chevauchement entre sous-chunks

doc      = Document(DOCX_PATH)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

# ──────────────────────────────────────────────
# SECTEUR PAR POSITION
# ──────────────────────────────────────────────
def get_secteur(table_index: int) -> str:
    if table_index < 17:
        return "Développement socio-économique"
    elif table_index < 28:
        return "Enseignement"
    elif table_index < 33:
        return "Emploi"
    elif table_index < 38:
        return "Entrepreunariat"
    elif table_index < 56:
        return "Services financiers"
    elif table_index < 70:
        return "Agriculture & Agribusiness"
    elif table_index < 80:
        return "TIC et innovation"
    elif table_index < 85:
        return "Tourisme et Culture"
    elif table_index < 101:
        return "Services & Industries manufacturières"
    elif table_index < 106:
        return "Santé et pharmaceutique"
    elif table_index < 111:
        return "Energy et Mining"
    else:
        return "Logistique & Mobilité"

# ──────────────────────────────────────────────
# EXTRACTION MÉTADONNÉES DE BASE
# (nom_projet, pays, client, secteur)
# Cherche dans toutes les cellules du tableau
# ──────────────────────────────────────────────
def extract_meta_field(all_text: str, field_name: str) -> str:
    pattern = rf"{re.escape(field_name)}\s*:\s*([^\n]+)"
    match   = re.search(pattern, all_text, re.IGNORECASE)
    if match:
        value = re.sub(r'\s+', ' ', match.group(1).strip())
        return value if value else "inconnu"
    return "inconnu"

def extract_metadata(table, table_index: int) -> dict:
    # Récupérer tout le texte du tableau en une fois
    all_text = "\n".join(
        cell.text.strip()
        for row in table.rows
        for cell in row.cells
        if cell.text.strip()
    )
    # Nom du projet = première cellule non vide
    nom = table.rows[0].cells[0].text.strip() if table.rows else "inconnu"

    return {
        "nom_projet": nom,
        "pays":       extract_meta_field(all_text, "Pays"),
        "nom_client": extract_meta_field(all_text, "Nom du Client"),
        "secteur":    get_secteur(table_index),
        "valeur":     extract_meta_field(all_text, "Valeur approximative des services en USD"),
    }

# ──────────────────────────────────────────────
# EXTRACTION DU TEXTE COMPLET D'UN TABLEAU
# ──────────────────────────────────────────────
def extract_full_text(table) -> str:
    """
    Extrait tout le contenu textuel du tableau dans l'ordre
    de lecture, en évitant les doublons des cellules fusionnées.
    """
    lines = []
    seen  = set()

    for row in table.rows:
        for cell in row.cells:
            text = cell.text.strip()
            if text and text not in seen:
                seen.add(text)
                lines.append(text)

    return "\n".join(lines)

# ──────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ──────────────────────────────────────────────
all_chunks = []   # liste finale de tous les sous-chunks

stats_projets       = 0
stats_sous_chunks   = 0
stats_sans_contenu  = 0

for table_index, table in enumerate(doc.tables):
    if len(table.rows) == 0:
        continue

    # 1️⃣ Nom du projet (skip si vide)
    nom = table.rows[0].cells[0].text.strip() if table.rows[0].cells else ""
    if not nom:
        continue

    # 2️⃣ Métadonnées de base
    meta = extract_metadata(table, table_index)

    # 3️⃣ Texte complet du projet
    full_text = extract_full_text(table)

    if not full_text or len(full_text) < 50:
        stats_sans_contenu += 1
        print(f"⚠️  [{table_index+1}] Contenu trop court, ignoré : {nom[:50]}")
        continue

    # 4️⃣ Découpage en sous-chunks
    sous_chunks = splitter.split_text(full_text)

    # 5️⃣ Créer un chunk final pour chaque sous-chunk
    for j, sous_chunk in enumerate(sous_chunks):
        # Préfixer chaque sous-chunk avec les métadonnées clés
        # pour que le vecteur contienne toujours le contexte du projet
        chunk_text = (
            f"Projet : {meta['nom_projet']}\n"
            f"Pays : {meta['pays']}\n"
            f"Client : {meta['nom_client']}\n"
            f"Secteur : {meta['secteur']}\n"
            f"---\n"
            f"{sous_chunk}"
        )
        all_chunks.append({
            "text":        chunk_text,
            "nom_projet":  meta["nom_projet"],
            "pays":        meta["pays"],
            "nom_client":  meta["nom_client"],
            "secteur":     meta["secteur"],
            "valeur":      meta["valeur"],
            "chunk_index": j,
            "total_chunks": len(sous_chunks),
        })

    stats_projets     += 1
    stats_sous_chunks += len(sous_chunks)
    print(f"✅ [{table_index+1:03d}] {nom[:50]:<50} → {len(sous_chunks)} sous-chunks")

# ──────────────────────────────────────────────
# SAUVEGARDE
# ──────────────────────────────────────────────
with open('chunks.json', 'w', encoding='utf-8') as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

# ──────────────────────────────────────────────
# RÉSUMÉ
# ──────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"📊 RÉSUMÉ DU CHUNKING")
print(f"{'='*60}")
print(f"  📁 Projets traités       : {stats_projets}")
print(f"  📦 Total sous-chunks     : {stats_sous_chunks}")
print(f"  ⚠️  Projets sans contenu  : {stats_sans_contenu}")
print(f"  📈 Moyenne chunks/projet : {stats_sous_chunks/stats_projets:.1f}" if stats_projets else "")
print(f"{'='*60}")
print(f"\n✅ Sauvegardé dans chunks.json")