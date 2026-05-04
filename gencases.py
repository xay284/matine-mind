

import json
import re
import random
import argparse
import datetime
from pathlib import Path
from collections import defaultdict, Counter

import requests
import chromadb
from sentence_transformers import SentenceTransformer

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CHROMA_PATH   = r"C:\Users\Chayma MAJJEDI\Desktop\chatbot_web\new pipeline\chroma_data"
EXISTING_TEST = r"C:\Users\Chayma MAJJEDI\Desktop\chatbot_web\new pipeline\test_cases.json"
OUTPUT_PATH   = r"C:\Users\Chayma MAJJEDI\Desktop\chatbot_web\new pipeline\test_cases_enriched.json"
EMBED_MODEL   = "intfloat/multilingual-e5-large"

OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "qwen2.5:7b"

N_SIMPLE      = 10     # questions simples à générer
N_COMPLEX     = 10     # questions complexes à générer

# Matching sémantique pour expected_projects
SEM_TOP_K     = 15     # candidats max à évaluer
SEM_THRESHOLD = 0.75   # seuil similarité cosinus (ajustable via --threshold)

random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

print("⏳ Chargement du modèle d'embedding...")
embedder = SentenceTransformer(EMBED_MODEL)
print(f"✅ Modèle {EMBED_MODEL} chargé")

print("⏳ Connexion ChromaDB...")
chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_collection("projets")
print(f"✅ {collection.count()} chunks disponibles\n")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — EXTRACTION DES PROJETS UNIQUES AVEC CONTENU COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def get_all_unique_projects() -> list[dict]:
    """
    Récupère pour chaque projet tous ses types de chunks :
    identite + description + services + impacts
    → contenu complet pour que le LLM génère des questions riches
    basées sur les vrais livrables, pas juste les métadonnées.
    """
    results = collection.get(include=["documents", "metadatas"])

    # Groupe par nom_projet, accumule les chunks par type
    project_data = defaultdict(lambda: {
        "identite"   : "",
        "description": "",
        "services"   : "",
        "impacts"    : "",
        "meta"       : {}
    })

    for doc, meta in zip(results["documents"], results["metadatas"]):
        nom        = meta.get("nom_projet", "").strip()
        chunk_type = meta.get("chunk_type", "")
        if not nom:
            continue

        if chunk_type == "identite" and not project_data[nom]["identite"]:
            project_data[nom]["identite"] = doc
            project_data[nom]["meta"]     = meta
        elif chunk_type == "description":
            project_data[nom]["description"] += doc + "\n"
        elif chunk_type == "services":
            project_data[nom]["services"] += doc + "\n"
        elif chunk_type == "impacts":
            project_data[nom]["impacts"] += doc + "\n"

    # Construit la liste finale
    projects = []
    for nom, data in project_data.items():
        if not data["meta"]:
            continue
        meta = data["meta"]

        # Contenu complet pour le LLM (limité à 3000 chars)
        full_content = "\n\n".join(filter(None, [
            data["identite"],
            data["description"].strip(),
            data["services"].strip(),
            data["impacts"].strip(),
        ]))[:3000]

        projects.append({
            "nom_projet"  : nom,
            "pays"        : meta.get("pays",       "") or "",
            "secteur"     : meta.get("secteur",    "") or "",
            "annee"       : meta.get("annee",      "") or "",
            "valeur"      : meta.get("valeur",     "") or "",
            "nom_client"  : meta.get("nom_client", "") or "",
            "description" : meta.get("description","") or "",
            "services"    : meta.get("services",   "") or "",
            "full_content": full_content,
        })

    print(f"✅ {len(projects)} projets uniques extraits avec contenu complet")
    return projects


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — EXPECTED PROJECTS PAR MATCHING SÉMANTIQUE
# ══════════════════════════════════════════════════════════════════════════════

def find_expected_projects_semantic(
    query      : str,
    source_nom : str,
    threshold  : float = SEM_THRESHOLD,
    top_k      : int   = SEM_TOP_K,
) -> list[str]:
    """
    Retrouve les projets attendus par recherche sémantique sur la query générée.

    - Encode la query avec le même modèle que le pipeline
    - Filtre par seuil de similarité cosinus → taille VARIABLE
      * question très spécifique  → 1-2 projets
      * question générale         → 4-6 projets
    - Le projet source est TOUJOURS inclus en premier
    """
    query_vec = embedder.encode(
        [f"query: {query}"],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings = [query_vec],
        n_results        = top_k,
        include          = ["distances", "metadatas"]
    )

    seen     = set()
    expected = []

    # Projet source toujours en premier
    expected.append(source_nom)
    seen.add(source_nom)

    for dist, meta in zip(
        results["distances"][0],
        results["metadatas"][0]
    ):
        nom        = meta.get("nom_projet", "").strip()
        similarity = 1.0 - dist   # distance cosinus → similarité

        if not nom or nom in seen:
            continue

        if similarity >= threshold:
            expected.append(nom)
            seen.add(nom)

    return expected


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — GÉNÉRATION LLM DES QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════

def call_ollama(system: str, user: str) -> str:
    payload = {
        "model" : OLLAMA_MODEL,
        "prompt": f"{system}\n\n{user}",
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"]
    except requests.exceptions.ConnectionError:
        print("   ⚠️  Ollama non disponible — utilisation du fallback")
        return ""
    except Exception as e:
        print(f"   ⚠️  Ollama erreur : {e}")
        return ""


def parse_json_safe(raw: str) -> dict | None:
    clean = re.sub(r"```json|```", "", raw.strip()).strip()
    start = clean.find("{")
    if start == -1:
        return None
    depth, end = 0, -1
    for i, ch in enumerate(clean[start:], start):
        if ch == "{"  : depth += 1
        elif ch == "}": depth -= 1
        if depth == 0 : end = i + 1; break
    if end == -1:
        return None
    try:
        return json.loads(clean[start:end])
    except json.JSONDecodeError:
        return None


# ── Question SIMPLE ───────────────────────────────────────────────────────────

SYSTEM_SIMPLE = """/no_think
Tu es un consultant senior dans un cabinet de conseil tunisien.
On te donne le contenu complet d'un projet réalisé par ton cabinet
(description, services rendus, impacts obtenus).

Génère UNE question de recherche SIMPLE qu'un collègue pourrait poser
pour retrouver ce projet dans une base de références.

RÈGLES :
- La question porte sur 1-2 thèmes principaux tirés des VRAIS LIVRABLES du projet
- Max 20 mots
- Style naturel consulting (comme pour chercher des références pour une proposition)
- NE mentionne PAS le nom exact du projet
- NE mentionne PAS de date ou d'année
- Utilise le vocabulaire métier exact des services/livrables décrits
- Langue : même langue que la description (FR si FR, EN si EN)

JSON uniquement : {"query": "la question générée"}
"""

def generate_simple_question(project: dict) -> str | None:
    context = f"""Nom du projet : {project['nom_projet']}
Pays : {project['pays']}
Secteur : {project['secteur']}
Client : {project['nom_client']}

Contenu complet du projet :
{project['full_content']}"""

    raw    = call_ollama(SYSTEM_SIMPLE, f"PROJET :\n{context}")
    result = parse_json_safe(raw)
    if result and result.get("query"):
        return result["query"].strip()

    return _fallback_simple_question(project)


def _fallback_simple_question(project: dict) -> str:
    pays    = project.get("pays", "").split(",")[0].strip()
    content = project.get("full_content", "") or project.get("description", "")
    noise   = {
        'projet', 'mission', 'étude', 'analyse', 'travaux', 'service',
        'dans', 'pour', 'avec', 'les', 'des', 'une', 'the', 'and', 'for'
    }
    words     = [w for w in re.findall(r'\b\w{5,}\b', content.lower()) if w not in noise]
    top_words = [w for w, _ in Counter(words).most_common(4)]
    keywords  = " ".join(top_words) if top_words else project.get("secteur", "consulting")
    if pays:
        return f"projets {keywords} {pays}".strip()
    return f"projets {keywords}".strip()


# ── Question COMPLEXE ─────────────────────────────────────────────────────────

SYSTEM_COMPLEX = """/no_think
Tu es un consultant senior dans un cabinet de conseil tunisien.
On te donne le contenu complet d'un projet réalisé par ton cabinet
(description, services rendus, impacts obtenus).

Génère UNE question de recherche COMPLEXE et réaliste qu'un collègue
pourrait poser pour retrouver ce projet ET des projets similaires.

RÈGLES :
- La question contient PLUSIEURS critères (2-4) tirés des VRAIS LIVRABLES :
    thème principal + type de mission + géographie + contrainte
- Ajoute UNE contrainte naturelle parmi :
    * Temporelle : "des 5 dernières années", "depuis 2020", "entre 2018 et 2023"
    * Budget     : "budget supérieur à 500k USD", "valeur entre 100k et 1M EUR"
    * Client     : type de client (ministère, bailleur de fonds, secteur privé)
- Max 40 mots
- Utilise le vocabulaire métier EXACT des services/livrables décrits
  (ex: "cadre d'opérationnalisation", "dialogue public-privé", "stratégie nationale")
- NE mentionne PAS le nom exact du projet
- Langue : même langue que la description (FR si FR, EN si EN)

JSON uniquement :
{"query": "la question générée", "contrainte_type": "temporelle|budget|client|aucune"}
"""

def generate_complex_question(project: dict) -> str | None:
    context = f"""Nom du projet : {project['nom_projet']}
Pays : {project['pays']}
Secteur : {project['secteur']}
Client : {project['nom_client']}
Année : {project['annee']}
Budget : {project['valeur']}

Contenu complet du projet :
{project['full_content']}"""

    raw    = call_ollama(SYSTEM_COMPLEX, f"PROJET :\n{context}")
    result = parse_json_safe(raw)
    if result and result.get("query"):
        return result["query"].strip()

    return _fallback_complex_question(project)


def _fallback_complex_question(project: dict) -> str:
    pays    = project.get("pays", "").split(",")[0].strip()
    annee   = project.get("annee", "")
    content = project.get("full_content", "") or project.get("description", "")
    noise   = {
        'projet', 'mission', 'étude', 'analyse', 'travaux', 'service',
        'dans', 'pour', 'avec', 'les', 'des', 'une', 'the', 'and', 'for'
    }
    words        = [w for w in re.findall(r'\b\w{5,}\b', content.lower()) if w not in noise]
    top_words    = [w for w, _ in Counter(words).most_common(5)]
    keywords     = " ".join(top_words[:4]) if top_words else project.get("secteur", "consulting")
    current_year = datetime.datetime.now().year
    if annee and annee.isdigit():
        n_years    = min(current_year - int(annee) + 2, 10)
        contrainte = f"des {n_years} dernières années"
    else:
        contrainte = "des 5 dernières années"
    if pays:
        return f"projets {keywords} {pays} {contrainte}".strip()
    return f"projets {keywords} {contrainte}".strip()


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — DÉDUPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_query(q: str) -> str:
    return re.sub(r'\s+', ' ', q.lower().strip())


def is_duplicate(new_query: str, existing_queries: list[str], threshold: float = 0.6) -> bool:
    def words(s):
        return set(re.findall(r'\b\w{3,}\b', normalize_query(s)))
    new_words = words(new_query)
    if not new_words:
        return True
    for eq in existing_queries:
        eq_words = words(eq)
        if not eq_words:
            continue
        intersection = len(new_words & eq_words)
        union        = len(new_words | eq_words)
        if union > 0 and intersection / union >= threshold:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# PROGRAMME PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main(args):

    # ── Chargement base existante ─────────────────────────────────────────────
    existing_cases = []
    existing_path  = Path(args.existing)

    if existing_path.exists():
        with open(existing_path, "r", encoding="utf-8") as f:
            existing_cases = json.load(f)
        print(f"📂 Base existante chargée : {len(existing_cases)} questions\n")
    else:
        print(f"⚠️  Fichier introuvable : {existing_path} — démarrage vide\n")

    existing_queries = [normalize_query(c.get("query", "")) for c in existing_cases]
    next_id          = max((c.get("id", 0) for c in existing_cases), default=0) + 1

    # ── Extraction projets ChromaDB ───────────────────────────────────────────
    all_projects = get_all_unique_projects()
    if not all_projects:
        print("❌ Aucun projet trouvé dans ChromaDB.")
        return

    # Filtre projets sans contenu utile
    all_projects = [p for p in all_projects if p["full_content"] and len(p["full_content"]) > 100]
    print(f"📦 {len(all_projects)} projets avec contenu suffisant\n")

    shuffled      = all_projects.copy()
    random.shuffle(shuffled)

    # ── QUESTIONS SIMPLES ─────────────────────────────────────────────────────
    print(f"{'─'*60}")
    print(f"🎯 Génération de {args.n_simple} questions SIMPLES...")
    print(f"{'─'*60}")

    simple_cases  = []
    used_projects = set()

    for project in shuffled:
        if len(simple_cases) >= args.n_simple:
            break

        print(f"\n[{len(simple_cases)+1}/{args.n_simple}] {project['nom_projet'][:55]}")

        query = generate_simple_question(project)
        if not query:
            print("   ⚠️  Query vide — ignoré")
            continue

        if is_duplicate(query, existing_queries + [c["query"] for c in simple_cases]):
            print(f"   ⚠️  Doublon — ignoré : {query[:50]}")
            continue

        expected = find_expected_projects_semantic(
            query      = query,
            source_nom = project["nom_projet"],
            threshold  = args.threshold,
        )

        simple_cases.append({
            "id"               : next_id,
            "query"            : query,
            "type"             : "simple",
            "source_project"   : project["nom_projet"],
            "expected_projects": expected,
            "generated_at"     : datetime.datetime.now().isoformat(),
        })
        existing_queries.append(normalize_query(query))
        used_projects.add(project["nom_projet"])
        next_id += 1

        print(f"   ✅ Query    : {query}")
        print(f"   📋 Expected : {len(expected)} projet(s)")
        for e in expected[:3]:
            print(f"      • {e[:60]}")
        if len(expected) > 3:
            print(f"      ... +{len(expected)-3} autres")

    print(f"\n✅ {len(simple_cases)} questions simples générées")

    # ── QUESTIONS COMPLEXES ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"🎯 Génération de {args.n_complex} questions COMPLEXES...")
    print(f"{'─'*60}")

    complex_cases = []
    remaining     = [p for p in shuffled if p["nom_projet"] not in used_projects]
    if len(remaining) < args.n_complex:
        remaining += [p for p in shuffled if p["nom_projet"] in used_projects]

    for project in remaining:
        if len(complex_cases) >= args.n_complex:
            break

        print(f"\n[{len(complex_cases)+1}/{args.n_complex}] {project['nom_projet'][:55]}")

        query = generate_complex_question(project)
        if not query:
            print("   ⚠️  Query vide — ignoré")
            continue

        if is_duplicate(query, existing_queries + [c["query"] for c in complex_cases]):
            print(f"   ⚠️  Doublon — ignoré : {query[:50]}")
            continue

        expected = find_expected_projects_semantic(
            query      = query,
            source_nom = project["nom_projet"],
            threshold  = args.threshold,
        )

        complex_cases.append({
            "id"               : next_id,
            "query"            : query,
            "type"             : "complex",
            "source_project"   : project["nom_projet"],
            "expected_projects": expected,
            "generated_at"     : datetime.datetime.now().isoformat(),
        })
        existing_queries.append(normalize_query(query))
        next_id += 1

        print(f"   ✅ Query    : {query}")
        print(f"   📋 Expected : {len(expected)} projet(s)")
        for e in expected[:3]:
            print(f"      • {e[:60]}")
        if len(expected) > 3:
            print(f"      ... +{len(expected)-3} autres")

    print(f"\n✅ {len(complex_cases)} questions complexes générées")

    # ── Fusion + sauvegarde ───────────────────────────────────────────────────
    new_cases    = simple_cases + complex_cases
    merged_cases = existing_cases + new_cases

    with open(Path(args.output), "w", encoding="utf-8") as f:
        json.dump(merged_cases, f, ensure_ascii=False, indent=2)

    # ── Rapport ───────────────────────────────────────────────────────────────
    if new_cases:
        sizes        = [len(c["expected_projects"]) for c in new_cases]
        avg_expected = sum(sizes) / len(sizes)
        min_expected = min(sizes)
        max_expected = max(sizes)
    else:
        avg_expected = min_expected = max_expected = 0

    print(f"\n{'═'*60}")
    print(f"📊 RAPPORT FINAL")
    print(f"{'═'*60}")
    print(f"  Questions existantes         : {len(existing_cases)}")
    print(f"  Questions simples ajoutées   : {len(simple_cases)}")
    print(f"  Questions complexes ajoutées : {len(complex_cases)}")
    print(f"  ────────────────────────────────")
    print(f"  TOTAL base enrichie          : {len(merged_cases)}")
    print(f"\n  Expected projects (nouvelles questions) :")
    print(f"    Moyenne : {avg_expected:.1f} projets/question")
    print(f"    Min / Max : {min_expected} / {max_expected}")
    print(f"    Seuil similarité : {args.threshold}")
    print(f"\n  Exemples :")
    for c in new_cases[:4]:
        tag = "🟢 Simple " if c["type"] == "simple" else "🔵 Complexe"
        print(f"  {tag} [{c['id']}] {c['query'][:65]}")
        print(f"           → {len(c['expected_projects'])} projet(s) attendu(s)")
    print(f"\n💾 Sauvegardé → {args.output}")
    print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Générateur de base de test — Matine Mind")
    parser.add_argument("--existing",  type=str,   default=EXISTING_TEST)
    parser.add_argument("--output",    type=str,   default=OUTPUT_PATH)
    parser.add_argument("--n_simple",  type=int,   default=N_SIMPLE,
                        help="Nombre de questions simples (défaut: 10)")
    parser.add_argument("--n_complex", type=int,   default=N_COMPLEX,
                        help="Nombre de questions complexes (défaut: 10)")
    parser.add_argument("--threshold", type=float, default=SEM_THRESHOLD,
                        help="Seuil similarité cosinus pour expected_projects (défaut: 0.75)")
    args = parser.parse_args()
    main(args)