import os
import re
import json
import tempfile
from pathlib import Path
from collections import Counter, defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE   = 600    # chars par chunk pour la classification
CHUNK_OVERLAP = 80    # overlap entre chunks

BATCH_SIZE   = 8      # chunks par appel LLM de classification

# Limite de chars envoyés au LLM de synthèse par catégorie
CAT_MAX_CHARS = {
    "contexte"       : 3000,
    "objectifs"      : 3000,
    "profil_cabinet" : 3000,
    "methodologie"   : 2000,
}


CATEGORIES = [
    "contexte",
    "objectifs",
    "profil_cabinet",
    "methodologie",
    "autre",          # tout le reste : admin, livrables, chronogramme, etc.
]


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION TEXTE BRUT (PDF / DOCX)
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extrait le texte d'un PDF avec pdfplumber.
    Fallback sur pypdf si pdfplumber échoue.
    """
    text = ""

    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            parts = []
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    parts.append(f"[PAGE {i}]\n{page_text}")
            text = "\n\n".join(parts)
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️  pdfplumber échoué : {e} — tentative avec pypdf")

    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        parts  = []
        for i, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                parts.append(f"[PAGE {i}]\n{page_text}")
        text = "\n\n".join(parts)
    except ImportError:
        raise ImportError("Installez pdfplumber ou pypdf : pip install pdfplumber pypdf")
    except Exception as e:
        raise RuntimeError(f"Impossible d'extraire le PDF : {e}")

    return text


def extract_text_from_docx(file_path: str) -> str:
    """
    Extrait le texte d'un fichier Word (.docx).
    Préserve titres, paragraphes et tableaux.
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Installez python-docx : pip install python-docx")

    doc   = Document(file_path)
    parts = []

    for element in doc.element.body:
        tag = element.tag.split("}")[-1]

        if tag == "p":
            from docx.oxml.ns import qn
            style_elem = element.find(f".//{qn('w:pStyle')}")
            style_name = style_elem.get(qn("w:val"), "") if style_elem is not None else ""
            text = "".join(
                node.text for node in element.iter()
                if node.tag.endswith("}t") and node.text
            ).strip()
            if not text:
                continue
            if "Heading" in style_name or "heading" in style_name or "Titre" in style_name:
                parts.append(f"\n## {text}")
            else:
                parts.append(text)

        elif tag == "tbl":
            rows = element.findall(
                ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr"
            )
            for row in rows:
                cells = row.findall(
                    ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc"
                )
                cell_texts = []
                for cell in cells:
                    cell_text = "".join(
                        node.text for node in cell.iter()
                        if node.tag.endswith("}t") and node.text
                    ).strip()
                    cell_texts.append(cell_text)
                if any(cell_texts):
                    parts.append(" | ".join(cell_texts))

    return "\n".join(parts)


def extract_text_from_tdr(file_path: str) -> tuple[str, str]:
    """
    Point d'entrée unique : détecte le format et extrait le texte.
    Retourne (texte_extrait, format_détecté).
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
        fmt  = "PDF"
    elif ext in (".docx", ".doc"):
        text = extract_text_from_docx(file_path)
        fmt  = "DOCX"
    else:
        raise ValueError(f"Format non supporté : {ext}. Utilisez PDF ou DOCX.")

    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = text.strip()

    char_count = len(text)
    print(f"✅ TDR extrait ({fmt}) — {char_count:,} caractères, ~{char_count//4:,} tokens")

    if char_count < 200:
        print("⚠️  Texte très court — le TDR est peut-être scanné (image). OCR nécessaire.")

    return text, fmt


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES JSON
# ══════════════════════════════════════════════════════════════════════════════

def parse_json_safe(raw: str) -> dict | None:
    """Parse robuste du JSON retourné par le LLM."""
    clean = re.sub(r"```json|```", "", raw.strip()).strip()
    start = clean.find("{")
    if start == -1:
        return None
    depth, end = 0, -1
    for i, ch in enumerate(clean[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0:
            end = i + 1
            break
    if end == -1:
        return None
    try:
        return json.loads(clean[start:end])
    except json.JSONDecodeError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — DÉCOUPE EN CHUNKS NATURELS
# ══════════════════════════════════════════════════════════════════════════════

def split_into_chunks(text: str) -> list[str]:
    """
    Découpe le texte en chunks en respectant les paragraphes naturels.
    Préfère couper sur un double saut de ligne, sinon sur un point.
    """
    text = re.sub(r'\n{3,}', '\n\n', text)
    paragraphs = text.split('\n\n')

    chunks  = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > CHUNK_SIZE:
            # Découpe le paragraphe trop long sur les phrases
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current) + len(sent) < CHUNK_SIZE:
                    current += " " + sent
                else:
                    if current.strip():
                        chunks.append(current.strip())
                    current = current[-CHUNK_OVERLAP:] + " " + sent
        else:
            if len(current) + len(para) < CHUNK_SIZE:
                current += "\n\n" + para
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = current[-CHUNK_OVERLAP:] + "\n\n" + para

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 50]


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — CLASSIFICATION SÉMANTIQUE PAR BATCH
# ══════════════════════════════════════════════════════════════════════════════

def classify_chunks_batch(
    chunks         : list[str],
    call_ollama_fn,
    model          : str,
    verbose        : bool = True,
) -> list[dict]:
    """
    Classifie tous les chunks par leur CONTENU sémantique, en batches.
    Retourne : list of {"chunk": str, "categorie": str, "score": int}

    La classification est basée sur le contenu du texte, jamais sur le titre
    de la section — ce qui rend l'approche robuste à tous les TDR.
    """
    all_results = []
    batches     = [chunks[i:i+BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    print(f"   {len(chunks)} chunks → {len(batches)} batch(es) de {BATCH_SIZE} max")

# Remplace le prompt de classify_chunks_batch
    system = """/no_think
Tu reçois une liste numérotée de chunks extraits d'un TDR (Termes de Référence).
Pour chaque chunk, assigne UNE catégorie parmi ces 5 :

- contexte       : présentation du projet, du client, du pays, du secteur, background général
- objectifs      : buts, finalités, résultats attendus, outputs de la mission
- profil_cabinet : critères d'éligibilité, expérience requise, profil du cabinet/consultant,
                   conditions de participation, références demandées
- methodologie   : approche méthodologique demandée, outils, démarche attendue
- autre          : livrables administratifs, chronogramme, planning, grille d'évaluation,
                   clauses contractuelles, annexes, formulaires, page de garde,
                   conditions de paiement, tout ce qui est purement administratif

RÈGLE ABSOLUE : classe par le CONTENU du texte, JAMAIS par le titre de la section.

JSON uniquement :
{"classifications": [{"id": 0, "categorie": "...", "score": 0-100}, ...]}
score = ta confiance dans la classification (0 = incertain, 100 = évident).
"""
    for batch_idx, batch in enumerate(batches, 1):
        if verbose:
            print(f"   Batch {batch_idx}/{len(batches)}...", end=" ", flush=True)

        numbered = "\n\n".join(
            f"[{i}]\n{chunk}" for i, chunk in enumerate(batch)
        )
        user   = f"CHUNKS À CLASSIFIER :\n\n{numbered}"
        raw    = call_ollama_fn(system, user, model=model)
        result = parse_json_safe(raw)

        if result and "classifications" in result:
            for item in result["classifications"]:
                chunk_idx = item.get("id", 0)
                categorie = item.get("categorie", "autre")
                score     = item.get("score", 50)

                # Validation de la catégorie
                if categorie not in CATEGORIES:
                    categorie = "autre"

                if 0 <= chunk_idx < len(batch):
                    all_results.append({
                        "chunk"    : batch[chunk_idx],
                        "categorie": categorie,
                        "score"    : int(score),
                    })
            if verbose:
                cats = [r["categorie"] for r in all_results[-len(batch):]]
                print(f"OK ({', '.join(cats)})")
        else:
            # Fallback : marque tous les chunks du batch comme "autre"
            for chunk in batch:
                all_results.append({"chunk": chunk, "categorie": "autre", "score": 0})
            if verbose:
                print("⚠️  parsing échoué → 'autre'")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — REGROUPEMENT PAR CATÉGORIE
# ══════════════════════════════════════════════════════════════════════════════

def group_chunks_by_category(
    classified : list[dict],
    min_score  : int = 40,
) -> dict[str, str]:
    """
    Regroupe les chunks par catégorie, dans l'ordre de confiance décroissant.
    Respecte la limite de chars par catégorie (CAT_MAX_CHARS).
    Les chunks "autre" et ceux sous min_score sont ignorés.
    """
    groups: dict[str, list[dict]] = defaultdict(list)

    for item in classified:
        if item["categorie"] != "autre" and item["score"] >= min_score:
            groups[item["categorie"]].append(item)

    result = {}
    for cat, items in groups.items():
        items_sorted = sorted(items, key=lambda x: x["score"], reverse=True)
        max_chars    = CAT_MAX_CHARS.get(cat, 2000)
        parts        = []
        total        = 0

        for item in items_sorted:
            chunk_text = item["chunk"]
            if total + len(chunk_text) <= max_chars:
                parts.append(chunk_text)
                total += len(chunk_text)
            else:
                remaining = max_chars - total
                if remaining > 100:
                    parts.append(chunk_text[:remaining] + "\n[...]")
                break

        if parts:
            result[cat] = "\n\n---\n\n".join(parts)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — SYNTHÈSE LLM FINALE
# ══════════════════════════════════════════════════════════════════════════════

# Remplace synthesize_tdr
def synthesize_tdr(grouped, call_ollama_fn, model):
    label_map = {
        "contexte"       : "CONTEXTE / BACKGROUND",
        "objectifs"      : "OBJECTIFS / RÉSULTATS ATTENDUS",
        "profil_cabinet" : "PROFIL DU CABINET / CRITÈRES D'ÉLIGIBILITÉ",
        "methodologie"   : "MÉTHODOLOGIE DEMANDÉE",
    }

    sections_prompt = []
    for cat in ["contexte", "objectifs", "profil_cabinet", "methodologie"]:
        if grouped.get(cat):
            sections_prompt.append(f"=== {label_map[cat]} ===\n{grouped[cat]}")

    if not sections_prompt:
        return {}

    tdr_structured = "\n\n".join(sections_prompt)

    system = """/no_think
Tu reçois les 4 sections utiles d'un TDR, pré-classifiées.
Extrais uniquement ce qui sert à rechercher des projets similaires.
Réponds dans la langue du document (FR ou EN).

JSON UNIQUEMENT :
{
  "contexte": "2-3 phrases : client, secteur, pays, objectif global",
  "objectifs": ["objectif 1", "objectif 2", ...],
  "query_synthetisee": "max 25 mots : secteur + type de mission + géographie",
  "criteres_qualification": ["critère 1", ...],
  "methodologie_demandee": "résumé en 1 phrase" ou null,
  "langue_detectee": "fr" | "en" | "mixed",
  "complexite_estimee": "faible" | "moyenne" | "elevee"
}

RÈGLES :
- objectifs      : extraire les thèmes métier des OBJECTIFS/RÉSULTATS ATTENDUS uniquement.
                   Jamais de livrables administratifs, jamais de chronogramme.
- criteres_qualification : uniquement depuis la section PROFIL CABINET.
                           Si absent → []
- methodologie_demandee  : uniquement depuis la section MÉTHODOLOGIE. Si absent → null
- Ne jamais inventer ce qui n'est pas dans le texte.
"""
    raw    = call_ollama_fn(system, tdr_structured, model=model)
    result = parse_json_safe(raw)
    return result or {}


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK — si le pipeline sémantique échoue complètement
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_tdr_extraction(text: str) -> dict:
    return {
        "contexte"              : text[:300],
        "objectifs"             : [text[:200]],
        "query_synthetisee"     : text[:150],
        "criteres_qualification": [],
        "methodologie_demandee" : None,
        "langue_detectee"       : "fr",
        "complexite_estimee"    : "moyenne",
    }


def _validate_tdr_result(result: dict) -> dict:
    defaults = {
        "contexte"              : "",
        "objectifs"             : [],
        "query_synthetisee"     : "",
        "criteres_qualification": [],
        "methodologie_demandee" : None,
        "langue_detectee"       : "fr",
        "complexite_estimee"    : "moyenne",
    }
    for key, default in defaults.items():
        if key not in result or result[key] is None and default is not None:
            result[key] = default

    # objectifs doit être une liste
    if isinstance(result["objectifs"], str):
        result["objectifs"] = [
            line.strip("- •*").strip()
            for line in result["objectifs"].split("\n")
            if line.strip()
        ]

    # Nettoyage
    result["objectifs"] = [o for o in result["objectifs"] if o and len(o) > 5]
    result["criteres_qualification"] = [
        c for c in result.get("criteres_qualification", []) if c and len(c) > 3
    ]
    return result



# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE TDR — POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def analyze_tdr(
    tdr_text      : str,
    call_ollama_fn,
    model         : str,
    verbose       : bool = True,
) -> dict:
    """
    Pipeline sémantique complet :
      Étape 1 — Découpe le texte en chunks naturels (~600 chars)
      Étape 2 — Classifie chaque chunk par son contenu (pas son titre)
      Étape 3 — Regroupe les chunks par catégorie
      Étape 4 — Synthèse LLM sur le texte pré-mâché

    Avantage vs ancienne approche :
    - Pas de limite de 12k chars — tout le document est traité
    - Robuste aux titres de sections variables (FR, EN, atypiques)
    - Classification sémantique : "Ce que nous attendons" → "livrables"
    """

    # ── Étape 1 : Découpe ──────────────────────────────────
    print("\n✂️  Étape 1 — Découpe en chunks...")
    chunks = split_into_chunks(tdr_text)
    print(f"   {len(chunks)} chunks générés ({len(tdr_text):,} chars au total)")

    # ── Étape 2 : Classification sémantique ────────────────
    print(f"\n🏷️  Étape 2 — Classification sémantique des chunks...")
    classified = classify_chunks_batch(chunks, call_ollama_fn, model, verbose=verbose)

    if verbose:
        dist = Counter(c["categorie"] for c in classified)
        print(f"\n   Distribution des catégories : {dict(dist)}")
        # Chunks à score faible
        low_score = [c for c in classified if c["score"] < 40]
        if low_score:
            print(f"   ⚠️  {len(low_score)} chunk(s) avec score < 40 (exclus du regroupement)")

    # ── Étape 3 : Regroupement ──────────────────────────────
    print(f"\n📦 Étape 3 — Regroupement par catégorie...")
    grouped = group_chunks_by_category(classified, min_score=40)

    chars_sent  = sum(len(v) for v in grouped.values())
    coverage    = round(chars_sent / len(tdr_text) * 100) if tdr_text else 0
    cats_found  = list(grouped.keys())
    cats_missing = [c for c in ["objectifs", "profil_cabinet"] if c not in grouped]

    print(f"   Catégories extraites : {cats_found}")
    if cats_missing:
        print(f"   ⚠️  Catégories vides  : {cats_missing}")
    print(f"   Chars envoyés au LLM : {chars_sent:,} / {len(tdr_text):,} ({coverage}% du doc)")

    # Fallback si aucune catégorie utile trouvée
    if not grouped or not grouped.get("objectifs"):
        print("   ⚠️  Aucun objectif détecté — fallback début+fin du document")
        half = 4000
        grouped.setdefault("contexte",  tdr_text[:half])
        grouped.setdefault("objectifs", tdr_text[-half:])

    # ── Étape 4 : Synthèse LLM finale ──────────────────────
    print(f"\n🧠 Étape 4 — Synthèse LLM finale...")
    result = synthesize_tdr(grouped, call_ollama_fn, model)

    if not result:
        print("⚠️  Synthèse LLM échouée — fallback extraction basique")
        result = _fallback_tdr_extraction(tdr_text)

    result = _validate_tdr_result(result)

    # Métadonnées de debug
    result["_nb_chunks"]         = len(chunks)
    result["_chars_total"]       = len(tdr_text)
    result["_chars_envoyes_llm"] = chars_sent
    result["_coverage_pct"]      = coverage
    result["_categories_vides"]  = cats_missing

    if verbose:
        _print_tdr_analysis(result)

    return result


def _print_tdr_analysis(result: dict):
    pct = result.get("_coverage_pct", "?")
    print(f"\n{'─'*60}")
    print(f"📋 RÉSULTAT ANALYSE TDR")
    print(f"{'─'*60}")
    print(f"📊 Couverture     : {result.get('_chars_envoyes_llm',0):,} / "
          f"{result.get('_chars_total',0):,} chars ({pct}%)")
    print(f"🌍 Langue         : {result.get('langue_detectee', '?')}")
    print(f"⚙️  Complexité     : {result.get('complexite_estimee', '?')}")
    print(f"\n📌 Contexte :")
    print(f"   {result.get('contexte', '')[:150]}...")
    print(f"\n🎯 Objectifs ({len(result.get('objectifs', []))}) :")
    for i, o in enumerate(result.get("objectifs", []), 1):
        print(f"   [{i:02d}] {o[:80]}")
    print(f"\n🔍 Query synthétisée   : {result.get('query_synthetisee', '')}")
    print(f"📋 Critères qualif.    : {result.get('criteres_qualification', [])}")
    if result.get("methodologie_demandee"):
        print(f"🔧 Méthodologie        : {result['methodologie_demandee']}")
    if result.get("_categories_vides"):
        print(f"⚠️  Catégories vides   : {result['_categories_vides']}")
    print(f"{'─'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DE LA QUERY FINALE POUR ask()
# ══════════════════════════════════════════════════════════════════════════════

def build_query_from_tdr(tdr_analysis: dict) -> str:
    """
    Construit la query à passer dans ask() depuis l'analyse TDR.
    Combine contexte + query synthétisée + livrables (séparés par " ; "
    pour que extract_criteres_tor() les segmente correctement).
    """
    parts = []

    if tdr_analysis.get("contexte"):
        parts.append(tdr_analysis["contexte"][:200])

    if tdr_analysis.get("query_synthetisee"):
        parts.append(tdr_analysis["query_synthetisee"])

    objectifs = tdr_analysis.get("objectifs", [])
    if objectifs:
        parts.append(" ; ".join(objectifs[:6]))

    if tdr_analysis.get("methodologie_demandee"):
        parts.append(tdr_analysis["methodologie_demandee"])

    return "\n".join(parts)
    # ⚠️ criteres_qualification n'est PAS injecté dans la query
    # → il sera utilisé séparément pour filtrer/afficher le profil demandé



# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLET TDR → RÉSULTATS RAG
# ══════════════════════════════════════════════════════════════════════════════

def process_tdr(
    file_path      : str,
    call_ollama_fn,
    ask_fn,
    model          : str,
    verbose        : bool = True,
) -> dict:
    """
    Pipeline complet : fichier TDR → résultats RAG.

    Arguments :
        file_path      : chemin vers le fichier TDR (PDF ou DOCX)
        call_ollama_fn : fonction call_ollama() de rag_pipeline.py
        ask_fn         : fonction ask() de rag_pipeline.py
        model          : nom du modèle Ollama (ex: "qwen2.5:7b")
        verbose        : affiche les logs détaillés

    Retourne le même format que ask() + champs "tdr_analysis" et "tdr_source".
    """
    print(f"\n{'═'*60}")
    print(f"📄 TDR REÇU : {Path(file_path).name}")
    print(f"{'═'*60}")

    # ── Étape A : Extraction texte ────────────────────────────
    print("\n📖 Extraction du texte...")
    tdr_text, fmt = extract_text_from_tdr(file_path)

    if not tdr_text.strip():
        return {
            "error"       : "Texte vide après extraction — PDF scanné ? OCR nécessaire.",
            "table"       : {"projets": [], "criteres": []},
            "answer"      : "❌ Impossible d'extraire le texte du TDR.",
            "tdr_analysis": None,
        }

    # ── Étape B : Analyse sémantique ─────────────────────────
    tdr_analysis = analyze_tdr(
        tdr_text       = tdr_text,
        call_ollama_fn = call_ollama_fn,
        model          = model,
        verbose        = verbose,
    )

    # ── Étape C : Construction de la query ───────────────────
    query = build_query_from_tdr(tdr_analysis)

    if verbose:
        print(f"🔗 Query construite ({len(query)} chars) :")
        print(f"   {query[:200]}{'...' if len(query) > 200 else ''}\n")

    # ── Étape D : Pipeline RAG ────────────────────────────────
    print(f"{'═'*60}")
    print(f"🚀 Lancement du pipeline RAG...")
    print(f"{'═'*60}")

    rag_result = ask_fn(query, verbose=verbose)

    rag_result["tdr_analysis"] = tdr_analysis
    rag_result["tdr_source"]   = {
        "filename"  : Path(file_path).name,
        "format"    : fmt,
        "char_count": len(tdr_text),
        "nb_chunks" : tdr_analysis.get("_nb_chunks", 0),
        "coverage"  : tdr_analysis.get("_coverage_pct", 0),
    }

    return rag_result


# ══════════════════════════════════════════════════════════════════════════════
# DÉTECTION AUTOMATIQUE : fichier TDR ou question texte ?
# ══════════════════════════════════════════════════════════════════════════════

def is_tdr_file(input_str: str) -> bool:
    cleaned = input_str.strip().strip('"').strip("'").strip()
    path    = Path(cleaned)
    return path.exists() and path.suffix.lower() in (".pdf", ".docx", ".doc")


def smart_dispatch(
    user_input     : str,
    call_ollama_fn,
    ask_fn,
    model          : str,
    verbose        : bool = True,
) -> dict:
    """
    Dispatch automatique : détecte si l'input est un fichier TDR ou une question texte.
    """
    cleaned = user_input.strip().strip('"').strip("'").strip()

    if is_tdr_file(cleaned):
        print("📄 Fichier TDR détecté → mode analyse automatique")
        return process_tdr(
            file_path      = cleaned,
            call_ollama_fn = call_ollama_fn,
            ask_fn         = ask_fn,
            model          = model,
            verbose        = verbose,
        )
    else:
        print("💬 Question texte détectée → mode recherche directe")
        return ask_fn(cleaned, verbose=verbose)


# ══════════════════════════════════════════════════════════════════════════════
# MODE STANDALONE — test extraction seule (sans rag_pipeline)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python tdr_parser.py <chemin.pdf|.docx>")
        print("Exemple : python tdr_parser.py tdr_emploi_maroc.pdf")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        text, fmt = extract_text_from_tdr(file_path)
        print(f"\n{'═'*60}")
        print(f"TEXTE EXTRAIT ({fmt}) — {len(text):,} caractères")
        print(f"{'═'*60}")

        print("\n--- APERÇU DÉCOUPE EN CHUNKS ---")
        chunks = split_into_chunks(text)
        print(f"{len(chunks)} chunks générés")
        for i, c in enumerate(chunks[:5], 1):
            print(f"\n[Chunk {i}] ({len(c)} chars)\n{c[:200]}...")

    except Exception as e:
        print(f"❌ Erreur : {e}")
        sys.exit(1)