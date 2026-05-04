import json
import re
import datetime
import math
from collections import defaultdict
from sentence_transformers import CrossEncoder, SentenceTransformer
import requests
import chromadb
import numpy as np
from tdr_parser import smart_dispatch, is_tdr_file
from functools import lru_cache
import time
import hashlib

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
CHROMA_PATH      = r"C:\Users\Chayma MAJJEDI\Desktop\chatbot_web\new pipeline\chroma_data"
EMBED_MODEL      = "intfloat/multilingual-e5-large"
TOP_K_CHROMA     = 60
TOP_K_BM25       = 60
TOP_K_PASS1      = 20
TOP_K_FINAL      = 25
RRF_K            = 60
OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_EXTRACTOR = "qwen2.5:7b"
COCHE_THRESHOLD  = 0.25
CURRENT_YEAR     = datetime.datetime.now().year

LLM_VERIF_SCORE_THRESHOLD = 10.0
VERIF_BATCH_SIZE           = 10   # ← traitement par batch pour éviter les timeouts

# ── Mapping confiance → multiplicateur de score ─────────────
CONFIANCE_BOOST = {1: 0.6, 2: 0.8, 3: 1.0, 4: 1.15, 5: 1.3}

# ══════════════════════════════════════════════════════════════
# INITIALISATION
# ══════════════════════════════════════════════════════════════
print("⏳ Chargement...")
embedder   = SentenceTransformer(EMBED_MODEL)
reranker   = CrossEncoder("BAAI/bge-reranker-v2-m3")
chroma     = chromadb.PersistentClient(path=CHROMA_PATH)

def _get_collection():
    try:
        return chroma.get_collection("projets")
    except Exception:
        existing = [c.name for c in chroma.list_collections()]
        raise RuntimeError(
            f"Collection 'projets' introuvable dans {CHROMA_PATH}\n"
            f"Collections disponibles : {existing or ['(aucune)']}\n"
            f"→ Relancez votre script d'indexation."
        )

collection = _get_collection()

SECTEURS = [
    "Services financiers",
    "Enseignement, Emploi & Entrepreneuriat",
    "Développement socio-économique",
    "Agriculture & Agribusiness",
    "Services & Industries manufacturières",
    "TIC & innovation",
    "Tourisme & Culture",
    "Energy & Mining",
    "Santé et Pharmaceutique",
    "Logistique & Mobilité",
]

SECTEURS_DESCRIPTIONS = {
    "Services financiers": "inclusion financière, microfinance, banque, crédit, assurance, fintech, marchés financiers, système bancaire",
    "Enseignement, Emploi & Entrepreneuriat": "enseignement, formation professionnelle, emploi, entrepreneuriat, compétences, jeunes",
    "Développement socio-économique": "développement économique, protection sociale, réformes, politiques publiques, gouvernance",
    "Agriculture & Agribusiness": "agriculture, irrigation, élevage, pêche, agribusiness, sécurité alimentaire, rural",
    "Services & Industries manufacturières": "industrie, manufacture, services, PME, production",
    "TIC & innovation": "digitalisation, numérique, technologies, innovation, e-gouvernement, systèmes d'information",
    "Tourisme & Culture": "tourisme, culture, patrimoine, hospitality",
    "Energy & Mining": "énergie, mines, hydrogène vert, solaire, renouvelable, environnement",
    "Santé et Pharmaceutique": "santé publique, médecine, hôpitaux, pharmacie, couverture médicale",
    "Logistique & Mobilité": "logistique, transport, mobilité, routes, ports, chaîne d'approvisionnement",
}

print("📊 Encodage des secteurs...")
_secteur_texts = [f"{s} : {SECTEURS_DESCRIPTIONS[s]}" for s in SECTEURS]
_secteur_vecs  = embedder.encode(
    [f"passage: {t}" for t in _secteur_texts],
    normalize_embeddings=True,
    show_progress_bar=False,
)
print(f"✅ {len(SECTEURS)} secteurs encodés\n")
print(f"✅ ChromaDB prête — {collection.count()} chunks")
print(f"✅ Reranker chargé\n")


# ══════════════════════════════════════════════════════════════
# BM25
# ══════════════════════════════════════════════════════════════
print("📚 Construction index BM25...")

_BM25_K1    = 1.5
_BM25_B     = 0.75
_BM25_DELTA = 0.5

_STOPWORDS = {
    'de','du','des','le','la','les','un','une','et','en','à','au','aux',
    'pour','par','sur','dans','avec','ou','est','que','qui','se','sa',
    'son','ses','nous','vous','ils','elles','d','l','j','n','y','ce',
    'cet','cette','ces','mon','ton','leur','leurs','dont','où','mais',
    'the','of','in','for','a','an','and','or','to','is','are','was','were',
    'projet','projets','mission','missions','services','service',
    'notre','personnel','activités','activité',
    'principales','principaux','principal','résultats','résultat',
    'description','effectivement','rendus','impacts','impact',
    'majeurs','majeur','notamment','également','afin','ainsi',
    'suite','cadre','objectifs','objectif','approche',
}

_nlp = None
try:
    import spacy
    _nlp = spacy.load("fr_core_news_md", disable=["parser", "ner"])
    print("✅ spaCy chargé — lemmatisation activée")
except Exception:
    print("⚠️  spaCy non disponible — tokenisation regex uniquement")

@lru_cache(maxsize=8192)
def _tokenize(text: str) -> tuple:
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    if _nlp is not None:
        doc = _nlp(text)
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and len(token.lemma_) > 2
            and token.lemma_ not in _STOPWORDS
        ]
    else:
        tokens = [
            t for t in text.split()
            if t not in _STOPWORDS and len(t) > 2
        ]
    return tuple(tokens)

_all_results    = collection.get(include=["documents", "metadatas"])
_bm25_docs      = _all_results["documents"]
_bm25_metas     = _all_results["metadatas"]
_bm25_ids       = _all_results["ids"]
_tokenized_docs = [list(_tokenize(doc)) for doc in _bm25_docs]
_N              = len(_tokenized_docs)
_avgdl          = sum(len(d) for d in _tokenized_docs) / max(_N, 1)

_df: dict = defaultdict(int)
for tokens in _tokenized_docs:
    for term in set(tokens):
        _df[term] += 1

@lru_cache(maxsize=65536)
def _idf(term: str) -> float:
    df = _df.get(term, 0)
    return math.log((_N - df + 0.5) / (df + 0.5) + 1)

_tf_docs = []
for tokens in _tokenized_docs:
    tf: dict = defaultdict(int)
    for t in tokens:
        tf[t] += 1
    _tf_docs.append(dict(tf))

_doc_lengths = [len(tokens) for tokens in _tokenized_docs]

print(f"✅ BM25+ indexé — {_N} chunks, avgdl={_avgdl:.0f} tokens")
print(f"   Vocabulaire : {len(_df):,} termes uniques\n")


def bm25_search(query: str, top_k: int = TOP_K_BM25) -> list:
    query_terms = list(_tokenize(query))
    if not query_terms:
        return []
    scores: dict = {}
    for term in query_terms:
        idf_val = _idf(term)
        if idf_val < 0.1:
            continue
        for i, tf in enumerate(_tf_docs):
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            dl      = _doc_lengths[i]
            norm    = 1 - _BM25_B + _BM25_B * dl / _avgdl
            tf_norm = _BM25_DELTA + (freq * (_BM25_K1 + 1)) / (freq + _BM25_K1 * norm)
            scores[i] = scores.get(i, 0.0) + idf_val * tf_norm
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(i, score, _bm25_docs[i], _bm25_metas[i]) for i, score in top]


# ══════════════════════════════════════════════════════════════
# RRF
# ══════════════════════════════════════════════════════════════
RRF_WEIGHTS = {
    "embedding_sector" : 1.5,
    "embedding"        : 1.2,
    "bm25"             : 0.8,
}
RRF_CONSENSUS_BOOST = 1.15
RRF_CHUNK_TYPE_BOOST = {
    "identite":      1.10,
    "description":   1.08,
    "services":      1.08,
    "impacts":       1.05,
    "chiffres_cles": 1.06,
    "contexte":      1.05,
    "full_text":     1.00,
}

def reciprocal_rank_fusion(
    ranked_lists   : list,
    source_labels  : list = None,
    chunk_registry : dict = None,
    k              : int = RRF_K,
) -> dict:
    if source_labels is None:
        source_labels = ["embedding"] * len(ranked_lists)
    rrf_scores: dict = defaultdict(float)
    for ranked, label in zip(ranked_lists, source_labels):
        weight = RRF_WEIGHTS.get(label, 1.0)
        for rank, key in enumerate(ranked, start=1):
            rrf_scores[key] += weight / (k + rank)
    n_lists = len(ranked_lists)
    if n_lists > 1:
        sets = [set(ranked) for ranked in ranked_lists]
        consensus_keys = sets[0].intersection(*sets[1:])
        for key in consensus_keys:
            rrf_scores[key] *= RRF_CONSENSUS_BOOST
    if chunk_registry:
        for key in list(rrf_scores.keys()):
            if key in chunk_registry:
                _, meta = chunk_registry[key]
                chunk_type = meta.get("chunk_type", "full_text")
                boost = RRF_CHUNK_TYPE_BOOST.get(chunk_type, 1.0)
                if boost != 1.0:
                    rrf_scores[key] *= boost
    return dict(sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True))


# ══════════════════════════════════════════════════════════════
# OLLAMA
# ══════════════════════════════════════════════════════════════
class OllamaError(Exception):
    pass

OLLAMA_TIMEOUTS = {
    "extraction":    90,
    "normalisation": 180,
    "matching":      240,
    "verification":  300,
    "default":       120,
}

_OLLAMA_CACHE: dict = {}
_OLLAMA_CACHE_MAX = 512

def _cache_key(system: str, user: str, model: str) -> str:
    content = f"{model}||{system[:200]}||{user[:500]}"
    return hashlib.md5(content.encode()).hexdigest()

def call_ollama(
    system          : str,
    user            : str,
    model           : str,
    timeout_profile : str = "default",
    max_retries     : int = 3,
    use_cache       : bool = True,
    raise_on_error  : bool = False,
) -> str:
    if use_cache:
        key = _cache_key(system, user, model)
        if key in _OLLAMA_CACHE:
            return _OLLAMA_CACHE[key]

    timeout = OLLAMA_TIMEOUTS.get(timeout_profile, OLLAMA_TIMEOUTS["default"])
    payload = {
        "model":  model,
        "prompt": f"{system}\n\n{user}",
        "stream": False,
        "options": {
            "temperature": 0,
            "seed":        42,
            "top_p":       1,
            "top_k":       1,
            "num_ctx":     8192,   # ← FIX : fenêtre de contexte explicite
            "num_predict": 2048,   # ← FIX : limite la longueur de sortie
        }
    }
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()["response"]
            if use_cache:
                if len(_OLLAMA_CACHE) >= _OLLAMA_CACHE_MAX:
                    oldest = next(iter(_OLLAMA_CACHE))
                    del _OLLAMA_CACHE[oldest]
                _OLLAMA_CACHE[key] = result
            return result
        except requests.exceptions.ConnectionError as e:
            last_error = e
            break
        except requests.exceptions.Timeout as e:
            last_error = e
            wait = 2 ** attempt
            print(f"   ⏱  Ollama timeout (tentative {attempt+1}/{max_retries}) — attente {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            last_error = e
            wait = 2 ** attempt
            print(f"   ⚠️  Ollama HTTP {e.response.status_code} (tentative {attempt+1}) — attente {wait}s")
            if attempt < max_retries - 1:
                time.sleep(wait)
        except Exception as e:
            last_error = e
            break

    err_msg = f"Ollama indisponible : {type(last_error).__name__}: {last_error}"
    print(f"   ❌ {err_msg}")
    if raise_on_error:
        raise OllamaError(err_msg) from last_error
    return ""


# ══════════════════════════════════════════════════════════════
# PARSE JSON
# ══════════════════════════════════════════════════════════════
try:
    from json_repair import repair_json
    _JSON_REPAIR_AVAILABLE = True
except ImportError:
    _JSON_REPAIR_AVAILABLE = False
    print("⚠️  json-repair non installé — pip install json-repair")

def parse_json_safe(
    raw           : str,
    required_keys : list = None,
    allow_array   : bool = False,
) -> object:
    if not raw or not raw.strip():
        return None
    clean = re.sub(r"```(?:json)?\s*", "", raw.strip())
    clean = clean.strip().strip("`").strip()
    result = _extract_json_structure(clean, allow_array)
    if result is None and _JSON_REPAIR_AVAILABLE:
        try:
            repaired = repair_json(clean, return_objects=True)
            if isinstance(repaired, (dict, list)):
                result = repaired
        except Exception:
            pass
    if result is None:
        return None
    if required_keys and isinstance(result, dict):
        missing = [k for k in required_keys if k not in result]
        if missing:
            print(f"   ⚠️  JSON valide mais clés manquantes : {missing}")
            return result
    return result

def _extract_json_structure(text: str, allow_array: bool) -> object:
    result = _find_and_parse(text, "{", "}")
    if result is not None:
        return result
    if allow_array:
        return _find_and_parse(text, "[", "]")
    return None

def _find_and_parse(text: str, open_char: str, close_char: str) -> object:
    start = text.find(open_char)
    if start == -1:
        return None
    depth       = 0
    in_string   = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
        if depth == 0:
            candidate = text[start:i + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    return json.loads(candidate.encode("utf-8").decode("utf-8-sig"))
                except Exception:
                    return None
    return None


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _safe_int(val):
    if val is None:
        return None
    try:
        v = int(val)
        return v if 1990 <= v <= CURRENT_YEAR + 1 else None
    except (ValueError, TypeError):
        return None

def _safe_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def _detect_temporalite_regex(query: str) -> dict:
    q = query.lower()
    m = re.search(r'(\d+)\s*derni[eè]res?\s*ann[eé]es?', q)
    if m:
        n = int(m.group(1))
        return {
            "type"        : "dernieres_annees",
            "annee_min"   : CURRENT_YEAR - n,
            "annee_max"   : CURRENT_YEAR,
            "description" : f"Projets des {n} dernières années ({CURRENT_YEAR-n}–{CURRENT_YEAR})"
        }
    m = re.search(r'entre\s+(\d{4})\s+et\s+(\d{4})', q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return {
            "type"        : "intervalle",
            "annee_min"   : min(a, b),
            "annee_max"   : max(a, b),
            "description" : f"Projets entre {min(a,b)} et {max(a,b)}"
        }
    m = re.search(r'(?:depuis|à partir de)\s+(\d{4})', q)
    if m:
        a = int(m.group(1))
        return {
            "type"        : "depuis",
            "annee_min"   : a,
            "annee_max"   : CURRENT_YEAR,
            "description" : f"Projets depuis {a}"
        }
    m = re.search(r"(?:avant|jusqu'en)\s+(\d{4})", q)
    if m:
        a = int(m.group(1))
        return {
            "type"        : "avant",
            "annee_min"   : None,
            "annee_max"   : a,
            "description" : f"Projets avant {a}"
        }
    return {"type": "aucune", "annee_min": None, "annee_max": None, "description": ""}

def _detect_budget_regex(query: str) -> dict:
    q = query.lower()

    def parse_amount(s: str):
        s = s.replace(' ', '').replace(',', '.')
        multiplier = 1
        if 'm' in s or 'million' in s:
            multiplier = 1_000_000
            s = re.sub(r'[mM]|million', '', s)
        elif s.endswith('k'):
            multiplier = 1_000
            s = s[:-1]
        try:
            return float(re.sub(r'[^\d.]', '', s)) * multiplier
        except (ValueError, TypeError):
            return None

    m = re.search(r'entre\s+([\d\s,\.]+(?:k|m)?)\s+et\s+([\d\s,\.]+(?:k|m)?)', q)
    if m:
        a, b = parse_amount(m.group(1)), parse_amount(m.group(2))
        if a and b:
            mn, mx = min(a, b), max(a, b)
            return {
                "type"        : "intervalle",
                "montant_min" : mn,
                "montant_max" : mx,
                "description" : f"Budget entre {int(mn):,} et {int(mx):,} USD"
            }
    m = re.search(r'(?:plus de|au moins|minimum)\s+([\d\s,\.]+(?:k|m)?)', q)
    if m:
        a = parse_amount(m.group(1))
        if a:
            return {
                "type"        : "min",
                "montant_min" : a,
                "montant_max" : None,
                "description" : f"Budget minimum {int(a):,} USD"
            }
    m = re.search(r"(?:moins de|maximum|jusqu'à)\s+([\d\s,\.]+(?:k|m)?)", q)
    if m:
        a = parse_amount(m.group(1))
        if a:
            return {
                "type"        : "max",
                "montant_min" : None,
                "montant_max" : a,
                "description" : f"Budget maximum {int(a):,} USD"
            }
    return {"type": "aucune", "montant_min": None, "montant_max": None, "description": ""}


def _postprocess_criteres(criteres: list, raw_query: str) -> list:
    NOISE = {
        'projets','projet','similaires','cherche','références',
        'portant','ayant','inclut','comprend','notamment',
        'une','un','des','les','pour','dans','avec',
        'sur','par','et','ou','qui','que','est','sont',
    }
    cleaned = []
    for c in criteres:
        c = c.strip().strip('"').strip("'").strip(';').strip()
        if not c or len(c) < 3:
            continue
        if c.lower() in NOISE:
            continue
        cleaned.append(c)

    seen_lower, deduped = set(), []
    for c in cleaned:
        key = c.lower().strip()
        if key not in seen_lower:
            seen_lower.add(key)
            deduped.append(c)

    final = []
    deduped_lower = [c.lower() for c in deduped]
    for i, c in enumerate(deduped):
        c_low      = c.lower()
        word_count = len(c.split())
        is_redundant = False
        if word_count <= 2:
            for j, other_low in enumerate(deduped_lower):
                if i != j and c_low != other_low:
                    if (f" {c_low} " in f" {other_low} " or
                            other_low.startswith(c_low + " ") or
                            other_low.endswith(" " + c_low)):
                        is_redundant = True
                        break
        if not is_redundant:
            final.append(c)
    return final


def _fallback_segment_extraction(raw_query: str) -> list:
    STOPWORDS = {
        'de','du','des','le','la','les','un','une','et','en','à','au','aux',
        'pour','par','sur','dans','avec','ou','est','que','qui','se','sa',
        'son','ses','nous','vous','ils','elles','d','l','j','n','y',
        'projets','projet','similaires','cherche','références','portant',
        'axés','axé','liées','lié','visant','permettant','relatif','relatifs',
        'the','of','in','for','a','an','and','or','to','is','are'
    }
    segments = re.split(r'[;•\n]|(?:\s*-\s+)', raw_query)
    segments = [s.strip() for s in segments if len(s.strip()) > 10]
    if not segments:
        segments = [raw_query]
    criteres = []
    seen     = set()
    for seg in segments:
        words   = re.findall(r'\b\w{3,}\b', seg.lower())
        content = [w for w in words if w not in STOPWORDS]
        for i in range(len(content) - 2):
            phrase = " ".join(content[i:i+3]).capitalize()
            if phrase.lower() not in seen:
                seen.add(phrase.lower())
                criteres.append(phrase)
                break
        for i in range(len(content) - 1):
            phrase = " ".join(content[i:i+2]).capitalize()
            if phrase.lower() not in seen:
                seen.add(phrase.lower())
                criteres.append(phrase)
                break
    return criteres[:8]


def parse_budget(value_str: str):
    if not value_str:
        return None, None
    s = value_str.lower()
    if "usd" in s or "$" in s:
        devise = "USD"
    elif "eur" in s or "€" in s:
        devise = "EUR"
    elif "dt" in s or "tnd" in s:
        devise = "TND"
    else:
        devise = "UNKNOWN"
    s = re.sub(r'[^\d.,kKmM ]', '', s)
    multiplier = 1
    if "m" in s:
        multiplier = 1_000_000
        s = s.replace("m", "")
    elif "k" in s:
        multiplier = 1_000
        s = s.replace("k", "")
    numbers = re.findall(r'[\d]+(?:[.,][\d]+)*', s)
    if not numbers:
        return None, devise
    raw = numbers[0].replace(' ', '').replace(',', '.')
    try:
        return float(raw) * multiplier, devise
    except Exception:
        return None, devise


def convert_to_usd(amount: float, devise: str) -> float:
    rates = {"USD": 1.0, "EUR": 1.08, "TND": 0.32}
    return amount * rates.get(devise, 1.0)


def respecte_contrainte_temporelle(meta: dict, contrainte: dict) -> bool:
    if contrainte["type"] == "aucune":
        return True
    annee_str = meta.get("annee", "") or ""
    if not annee_str or annee_str in ("N/A", ""):
        return True
    annees = re.findall(r'\b(20[0-2]\d|199\d)\b', str(annee_str))
    if not annees:
        return True
    annee_projet = int(annees[0])
    annee_min    = contrainte.get("annee_min")
    annee_max    = contrainte.get("annee_max")
    if annee_min is not None and annee_projet < annee_min:
        return False
    if annee_max is not None and annee_projet > annee_max:
        return False
    return True


def respecte_contrainte_budget(meta: dict, contrainte: dict) -> bool:
    if contrainte["type"] == "aucune":
        return True
    valeur_str = meta.get("valeur", "") or ""
    if not valeur_str or valeur_str.lower() in ("confidentiel", "n/a", ""):
        return True
    amount, devise = parse_budget(valeur_str)
    if amount is None:
        return True
    montant     = convert_to_usd(amount, devise)
    montant_min = contrainte.get("montant_min")
    montant_max = contrainte.get("montant_max")
    if montant_min is not None and montant < montant_min:
        return False
    if montant_max is not None and montant > montant_max:
        return False
    return True


# ══════════════════════════════════════════════════════════════
# EXTRACT ALL IN ONE — 1 seul appel LLM
# ══════════════════════════════════════════════════════════════

def extract_all_in_one(raw_query: str, verbose: bool = True) -> dict:

    system = f"""/no_think
Tu es un expert en analyse de besoins consulting. Tu analyses une question/requête et tu retournes UN SEUL JSON structuré.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARTIE 1 — REWRITE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reformule la question en français clair et précis :
- Traduire si la question est en anglais ou contient des termes anglais
- Expliquer les abréviations (ex: "SNE" → "Stratégie Nationale pour l'Emploi (SNE)", "PPP" → "Partenariat Public-Privé (PPP)")
- Clarifier sans changer le sens
- Conserver : pays, secteur, type de client, contraintes temporelles et budgétaires si présents
- Style naturel, max 40 mots
- Ne rien inventer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARTIE 2 — CRITÈRES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extrais CHAQUE critère thématique distinct mentionné dans la question.
Un critère = une thématique, un domaine, une mission, un secteur, un type d'activité, un livrable, un pays.

RÈGLE FONDAMENTALE :
- Si la question contient une liste de livrables (séparés par ";" ou "et" ou des tirets ou ","),
  chaque livrable = UN groupe de critères distincts à extraire séparément
- Entre 2 et 10 critères selon la richesse de la question
- Chaque critère : 1 à 4 mots, concis et précis

CE QUE TU NE DOIS JAMAIS EXTRAIRE :
① MÉTA-TERMES : "références", "projets", "expériences", "missions", "travaux",
   "donne moi", "liste", "cherche", "trouve", "similaires", "équivalents"
② TEMPORALITÉ : "10 dernières années", "depuis 2015", "récents", "derniers"
   → toujours ignorée ici, gérée dans la partie 3
③ VERBES ET CONNECTEURS : "portant sur", "ayant trait à", "relatifs à",
   "concernant", "axés sur", "ont été menés", "visant à"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARTIE 3 — CONTRAINTES (uniquement si présentes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Année actuelle : {CURRENT_YEAR}

CONTRAINTE TEMPORELLE (si absente → type="aucune", annee_min=null, annee_max=null) :
Types : "aucune" | "dernieres_annees" | "annee_exacte" | "depuis" | "avant" | "intervalle"
Calcul :
  "10 dernières années" → annee_min={CURRENT_YEAR-10}, annee_max={CURRENT_YEAR}
  "depuis 2015"         → annee_min=2015, annee_max={CURRENT_YEAR}
  "avant 2020"          → annee_min=null, annee_max=2020
  "en 2021"             → annee_min=2021, annee_max=2021
  "entre 2015 et 2020"  → annee_min=2015, annee_max=2020

CONTRAINTE BUDGET (si absente → type="aucune", montant_min=null, montant_max=null) :
Types : "aucune" | "min" | "max" | "intervalle" | "exact"
Convertis en USD : EUR×1.08 | TND×0.32 | k=×1000 | M=×1000000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT JSON ATTENDU — STRICT, aucun texte avant/après
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "rewrite": "...",
  "criteres": ["critère 1", "critère 2", "..."],
  "contrainte_temporelle": {{
    "type": "aucune",
    "annee_min": null,
    "annee_max": null,
    "description": ""
  }},
  "contrainte_budget": {{
    "type": "aucune",
    "montant_min": null,
    "montant_max": null,
    "description": ""
  }}
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXEMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: "projets portant sur la politique publique d'emploi"
→ {{
  "rewrite": "Projets portant sur la politique publique de l'emploi.",
  "criteres": ["emploi", "politique publique"],
  "contrainte_temporelle": {{"type":"aucune","annee_min":null,"annee_max":null,"description":""}},
  "contrainte_budget": {{"type":"aucune","montant_min":null,"montant_max":null,"description":""}}
}}

Q: "Une étude de marché sur les compétences numériques dans l'enseignement supérieur ; Un cadre d'opérationnalisation des programmes de formation certifiants axés sur l'emploi des jeunes ; Un manuel de procédures des subventions liées à la performance"
→ {{
  "rewrite": "Étude de marché sur les compétences numériques dans l'enseignement supérieur, cadre d'opérationnalisation des programmes de formation certifiants pour l'emploi des jeunes, et manuel de procédures des subventions à la performance.",
  "criteres": ["étude de marché", "compétences numériques", "enseignement supérieur", "cadre d'opérationnalisation", "programmes de formation certifiants", "emploi des jeunes", "manuel de procédures", "subventions à la performance"],
  "contrainte_temporelle": {{"type":"aucune","annee_min":null,"annee_max":null,"description":""}},
  "contrainte_budget": {{"type":"aucune","montant_min":null,"montant_max":null,"description":""}}
}}

Q: "missions d'appui institutionnel pour la réforme de la protection sociale et de la couverture santé en Tunisie depuis 2018 avec un budget de plus de 50k USD"
→ {{
  "rewrite": "Missions d'appui institutionnel pour la réforme de la protection sociale et de la couverture santé en Tunisie, depuis 2018, avec un budget supérieur à 50 000 USD.",
  "criteres": ["appui institutionnel", "réforme institutionnelle", "protection sociale", "santé", "Tunisie"],
  "contrainte_temporelle": {{"type":"depuis","annee_min":2018,"annee_max":{CURRENT_YEAR},"description":"Projets depuis 2018"}},
  "contrainte_budget": {{"type":"min","montant_min":50000,"montant_max":null,"description":"Budget minimum 50 000 USD"}}
}}

Q: "give me references on green hydrogen investment framework in Tunisia"
→ {{
  "rewrite": "Références sur l'élaboration d'un cadre d'investissement pour l'hydrogène vert en Tunisie.",
  "criteres": ["hydrogène vert", "cadre d'investissement", "énergies renouvelables", "Tunisie"],
  "contrainte_temporelle": {{"type":"aucune","annee_min":null,"annee_max":null,"description":""}},
  "contrainte_budget": {{"type":"aucune","montant_min":null,"montant_max":null,"description":""}}
}}
"""

    raw = call_ollama(
        system,
        f"Question :\n{raw_query}",
        model           = OLLAMA_EXTRACTOR,
        timeout_profile = "extraction",
    )

    result = parse_json_safe(raw, required_keys=["rewrite", "criteres"]) or {}

    DEFAULT_CONTRAINTE_T = {
        "type": "aucune", "annee_min": None, "annee_max": None, "description": ""
    }
    DEFAULT_CONTRAINTE_B = {
        "type": "aucune", "montant_min": None, "montant_max": None, "description": ""
    }

    rewrite  = result.get("rewrite",  "").strip() or raw_query[:200]
    criteres = result.get("criteres", []) or []
    contr_t  = result.get("contrainte_temporelle", DEFAULT_CONTRAINTE_T) or DEFAULT_CONTRAINTE_T
    contr_b  = result.get("contrainte_budget",     DEFAULT_CONTRAINTE_B) or DEFAULT_CONTRAINTE_B

    contr_t["annee_min"]   = _safe_int(contr_t.get("annee_min"))
    contr_t["annee_max"]   = _safe_int(contr_t.get("annee_max"))
    contr_b["montant_min"] = _safe_float(contr_b.get("montant_min"))
    contr_b["montant_max"] = _safe_float(contr_b.get("montant_max"))

    if contr_t["type"] == "aucune":
        contr_t_regex = _detect_temporalite_regex(raw_query)
        if contr_t_regex["type"] != "aucune":
            contr_t = contr_t_regex

    if contr_b["type"] == "aucune":
        contr_b_regex = _detect_budget_regex(raw_query)
        if contr_b_regex["type"] != "aucune":
            contr_b = contr_b_regex

    criteres = _postprocess_criteres(criteres, raw_query)
    if not criteres:
        criteres = _fallback_segment_extraction(raw_query)

    if verbose:
        print(f"✏️  Rewrite   : {rewrite}")
        print(f"📊 Critères  : {criteres}")
        if contr_t["type"] != "aucune":
            print(f"📅 Temporel  : {contr_t['description']}")
        if contr_b["type"] != "aucune":
            print(f"💰 Budget    : {contr_b['description']}")

    return {
        "rewrite"               : rewrite,
        "criteres"              : criteres,
        "contrainte_temporelle" : contr_t,
        "contrainte_budget"     : contr_b,
    }


# ══════════════════════════════════════════════════════════════
# COMPRESSION DU CHUNK
# ══════════════════════════════════════════════════════════════

def compress_chunk(meta: dict) -> str:
    nom    = meta.get("nom_projet", "Projet sans titre")
    client = meta.get("nom_client", "N/A")
    pays   = meta.get("pays",       "N/A")
    annee  = meta.get("annee",      "N/A")
    valeur = meta.get("valeur",     "N/A")

    parts = [
        f"PROJET: {nom}",
        f"CLIENT: {client}",
        f"PAYS: {pays}",
        f"ANNÉE: {annee}",
        f"BUDGET: {valeur}",
    ]

    if meta.get("description"):
        parts.append(f"DESCRIPTION: {meta['description']}")
    if meta.get("services"):
        s_list = meta["services"].replace(" | ", "\n  - ")
        parts.append(f"SERVICES RÉALISÉS:\n  - {s_list}")
    if meta.get("impacts"):
        i_list = meta["impacts"].replace(" | ", "\n  - ")
        parts.append(f"IMPACTS & RÉSULTATS:\n  - {i_list}")
    if meta.get("chiffres_cles"):
        parts.append(f"CHIFFRES CLÉS: {meta['chiffres_cles']}")
    if meta.get("mots_cles"):
        parts.append(f"DOMAINES: {meta['mots_cles']}")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════
# FULL TEXT PAR PROJET
# ══════════════════════════════════════════════════════════════

def fetch_full_context_for_project(nom_projet: str) -> str:
    try:
        results = collection.get(
            where={"nom_projet": {"$eq": nom_projet}},
            include=["documents", "metadatas"]
        )
        PRIORITY = {
            "identite": 0, "description": 1, "services": 2,
            "impacts": 3, "contexte": 4, "chiffres_cles": 5, "synthese": 6, "full_text": 7,
        }
        sorted_chunks = sorted(
            zip(results.get("documents", []), results.get("metadatas", [])),
            key=lambda x: PRIORITY.get(x[1].get("chunk_type", "full_text"), 99)
        )
        seen, parts = set(), []
        total_chars = 0
        for doc, meta in sorted_chunks:
            key = doc[:100]
            if key not in seen and total_chars < 6000:
                seen.add(key)
                parts.append(doc)
                total_chars += len(doc)
        return "\n\n".join(parts)
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════
# SCORING CROSSENCODER
# ══════════════════════════════════════════════════════════════

def _crossencoder_score_to_pct(raw_score: float) -> float:
    pct = 100.0 / (1.0 + math.exp(-raw_score))
    return round(pct, 1)

def compute_matching_crossencoder(
    projet_context : str,
    criteres       : list,
    original_query : str = "",
    verbose        : bool = False
) -> dict:
    if not criteres or not projet_context:
        return {c: 0.0 for c in criteres}
    context_trunc = projet_context[:4000]
    pairs = []
    for critere in criteres:
        query_synth = f"{critere} | {original_query[:150]}" if original_query else critere
        pairs.append((query_synth, context_trunc))
    raw_scores = reranker.predict(pairs)
    matching = {}
    for critere, raw_score in zip(criteres, raw_scores):
        pct = _crossencoder_score_to_pct(float(raw_score))
        matching[critere] = pct
        if verbose:
            print(f"      '{critere}' → raw={raw_score:.3f} → {pct}%")
    return matching


# ══════════════════════════════════════════════════════════════
# FIX 1 — ENRICHISSEMENT BATCH (remplace les N appels individuels)
# ══════════════════════════════════════════════════════════════

def _enrich_metas_batch(noms_projets: list) -> dict:
    """
    Enrichit les métadonnées de TOUS les projets en UN SEUL appel ChromaDB.
    Retourne un dict {nom_projet: merged_meta}.
    """
    if not noms_projets:
        return {}

    try:
        res = collection.get(
            where   = {"nom_projet": {"$in": noms_projets}},
            include = ["metadatas"]
        )
    except Exception as e:
        print(f"   ⚠️  Enrichissement batch échoué : {e}")
        return {}

    # Grouper les metas par projet
    proj_metas: dict = defaultdict(list)
    for m in res.get("metadatas", []):
        nom = m.get("nom_projet", "")
        if nom:
            proj_metas[nom].append(m)

    PRIORITY = {
        "identite": 0, "description": 1, "services": 2,
        "impacts": 3, "chiffres_cles": 4, "contexte": 5, "full_text": 6
    }
    FIELDS = [
        "nom_projet", "annee", "nom_client", "pays", "secteur",
        "valeur", "description", "services", "impacts",
        "mots_cles", "chiffres_cles"
    ]

    all_enriched = {}
    for nom, metas in proj_metas.items():
        sorted_m = sorted(
            metas,
            key=lambda m: PRIORITY.get(m.get("chunk_type", "full_text"), 99)
        )
        merged = {}
        for field in FIELDS:
            for m in sorted_m:
                val = m.get(field, "")
                if val and val not in ("N/A", ""):
                    merged[field] = val
                    break
        all_enriched[nom] = merged

    return all_enriched


# ══════════════════════════════════════════════════════════════
# FIX 2 — PAYLOAD RÉDUIT pour le vérificateur
# ══════════════════════════════════════════════════════════════

def _build_verif_payload(projet: dict, meta: dict) -> str:
    """Payload compact — max ~300 chars par projet pour éviter le timeout."""
    lines = [
        f"PROJET: {meta.get('nom_projet', 'N/A')}",
        f"SECTEUR: {meta.get('secteur', 'N/A')}",
        f"PAYS: {meta.get('pays', 'N/A')}",
        f"ANNÉE: {meta.get('annee', 'N/A')}",
    ]
    if meta.get("description"):
        lines.append(f"DESC: {meta['description'][:200]}")   # ← réduit à 200
    if meta.get("services"):
        lines.append(f"SERVICES: {meta['services'][:200]}")  # ← réduit à 200
    if meta.get("mots_cles"):
        lines.append(f"TAGS: {meta['mots_cles'][:150]}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# LLM VÉRIFICATEUR — avec batch + enrichissement batch
# ══════════════════════════════════════════════════════════════

def llm_verifier(
    projets_candidats : list,
    original_query    : str,
    rewrite           : str,
    criteres          : list,
    verbose           : bool = True,
) -> tuple:
    """
    Vérification LLM des projets candidats.
    Optimisations :
      - Enrichissement ChromaDB en 1 seul appel batch
      - Payload réduit par projet (~200 chars)
      - Traitement par batches de VERIF_BATCH_SIZE projets
    """
    if not projets_candidats:
        return [], []

    # ── Séparation par seuil ────────────────────────────────
    above_threshold = [
        (chunk, sf, pct, meta)
        for chunk, sf, pct, meta in projets_candidats
        if sf >= LLM_VERIF_SCORE_THRESHOLD
    ]
    below_threshold = [
        (chunk, sf, pct, meta)
        for chunk, sf, pct, meta in projets_candidats
        if sf < LLM_VERIF_SCORE_THRESHOLD
    ]

    if verbose:
        print(f"\n🔎 LLM Vérificateur : {len(above_threshold)} projets au-dessus du seuil "
              f"({LLM_VERIF_SCORE_THRESHOLD}%), {len(below_threshold)} exclus avant LLM")

    excluded_pre = [
        {
            "titre"                : meta.get("nom_projet", "N/A"),
            "annee"                : meta.get("annee", "N/A"),
            "client"               : meta.get("nom_client", "N/A"),
            "pays"                 : meta.get("pays", "N/A"),
            "decision"             : "exclude",
            "confiance"            : 0,
            "raison"               : f"Score trop faible ({sf:.1f}% < seuil {LLM_VERIF_SCORE_THRESHOLD}%)",
            "score_final_original" : sf,
            "score_final_ajuste"   : 0.0,
            "matching"             : {},
            "budget"               : meta.get("valeur", "N/A"),
            "mots_cles"            : meta.get("mots_cles", ""),
            "secteur"              : meta.get("secteur", "N/A"),
        }
        for chunk, sf, pct, meta in below_threshold
    ]

    if not above_threshold:
        return [], excluded_pre

    # ── FIX 1 : enrichissement en 1 seul appel batch ────────
    noms_projets = [meta.get("nom_projet", "") for _, _, _, meta in above_threshold]
    if verbose:
        print(f"   🔄 Enrichissement batch ({len(noms_projets)} projets)...")
    all_enriched = _enrich_metas_batch(noms_projets)

    # ── Prompt système (commun à tous les batches) ───────────
    criteres_str = " | ".join(criteres)
    system = f"""/no_think
Tu es un expert senior en analyse de portefeuilles de projets de développement et consulting.
Ta mission : évaluer la pertinence de projets candidats par rapport à une requête utilisateur,
en appliquant une logique de matching multi-niveaux — du plus précis au plus indirect.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUÊTE UTILISATEUR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Requête originale  : {original_query}
Requête reformulée : {rewrite}
Critères extraits  : {criteres_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOGIQUE DE MATCHING — 5 NIVEAUX (du fort au faible)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NIVEAU 1 — Correspondance exacte (confiance 5)
  Le projet mentionne explicitement les mêmes termes, expressions ou concepts.

NIVEAU 2 — Correspondance sémantique forte (confiance 4)
  Le projet traite du même domaine avec des termes différents (synonymes, reformulations).
  Exemples :
    "formation professionnelle" ↔ "renforcement des capacités"
    "inclusion financière" ↔ "accès aux services bancaires"

NIVEAU 3 — Relation thématique partielle (confiance 3)
  Le projet partage un ou plusieurs critères secondaires ou intervient dans le même écosystème.

NIVEAU 4 — Lien indirect ou complémentaire (confiance 2)
  Le projet touche de loin à la requête : secteur voisin, bénéficiaires similaires.

NIVEAU 5 — Pertinence très marginale mais non nulle (confiance 1)
  Un seul point de contact très lointain.

EXCLUSION (confiance 0) — UNIQUEMENT si :
  ✗ Aucun critère couvert, même de loin
  ✗ ET aucun mot-clé, service ou secteur ne touche même indirectement à la requête
  ✗ ET domaine complètement différent

  ⚠️  RÈGLE D'OR : En cas de doute → "keep" confiance 1 ou 2. N'exclure qu'avec certitude absolue.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT JSON — STRICT, aucun texte avant/après
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "projets": [
    {{
      "titre": "Titre exact du projet tel que fourni",
      "niveau": 2,
      "decision": "keep",
      "confiance": 4,
      "criteres_matches": ["critère A", "critère B"],
      "raison": "Explication courte, max 20 mots"
    }}
  ]
}}

RAPPEL : "titre" doit reproduire EXACTEMENT le nom du projet tel que fourni.
"""

    # ── FIX 2 : traitement par batches ──────────────────────
    all_verif_items = []

    for batch_start in range(0, len(above_threshold), VERIF_BATCH_SIZE):
        batch = above_threshold[batch_start : batch_start + VERIF_BATCH_SIZE]
        batch_num = batch_start // VERIF_BATCH_SIZE + 1
        total_batches = math.ceil(len(above_threshold) / VERIF_BATCH_SIZE)

        if verbose:
            print(f"   📤 Batch {batch_num}/{total_batches} — {len(batch)} projets → LLM ({OLLAMA_EXTRACTOR})...")

        projets_texte = ""
        for i, (chunk, sf, pct, meta) in enumerate(batch, 1):
            nom_projet    = meta.get("nom_projet", "")
            enriched_meta = all_enriched.get(nom_projet, {})
            full_meta     = {**meta, **{k: v for k, v in enriched_meta.items() if v}}

            projets_texte += f"\n--- PROJET {i} ---\n"
            projets_texte += _build_verif_payload({"score": sf}, full_meta)
            projets_texte += "\n"

        user_prompt = f"Voici les {len(batch)} projets à évaluer :\n{projets_texte}"

        raw = call_ollama(
            system          = system,
            user            = user_prompt,
            model           = OLLAMA_EXTRACTOR,
            timeout_profile = "verification",
            use_cache       = False,
        )

        # Debug : afficher le raw pour diagnostiquer
        if verbose and raw:
            print(f"   🔍 RAW batch {batch_num} (200 premiers chars) : {raw[:200]}")

        if not raw:
            if verbose:
                print(f"   ⚠️  Batch {batch_num} : Ollama vide — conservation par défaut")
            for chunk, sf, pct, meta in batch:
                all_verif_items.append({
                    "titre"     : meta.get("nom_projet", "N/A"),
                    "decision"  : "keep",
                    "confiance" : 3,
                    "niveau"    : 3,
                    "raison"    : "Ollama vide (fallback)",
                    "criteres_matches": [],
                    "_sf"       : sf,
                    "_meta"     : meta,
                    "_chunk"    : chunk,
                })
            continue

        parsed = parse_json_safe(raw, required_keys=["projets"])

        if not parsed or "projets" not in parsed:
            if verbose:
                print(f"   ⚠️  Batch {batch_num} : JSON invalide — conservation par défaut")
                print(f"      RAW complet : {raw[:500]}")
            for chunk, sf, pct, meta in batch:
                all_verif_items.append({
                    "titre"     : meta.get("nom_projet", "N/A"),
                    "decision"  : "keep",
                    "confiance" : 3,
                    "niveau"    : 3,
                    "raison"    : "JSON invalide (fallback)",
                    "criteres_matches": [],
                    "_sf"       : sf,
                    "_meta"     : meta,
                    "_chunk"    : chunk,
                })
            continue

        # Aligner les résultats LLM avec les projets du batch
        verif_by_titre = {}
        for item in parsed["projets"]:
            titre_llm = (item.get("titre") or "").strip().lower()
            verif_by_titre[titre_llm] = item

        def _find_verif_local(nom_projet: str):
            nom_lower = nom_projet.strip().lower()
            if nom_lower in verif_by_titre:
                return verif_by_titre[nom_lower]
            for titre_llm, item in verif_by_titre.items():
                if titre_llm[:40] in nom_lower or nom_lower[:40] in titre_llm:
                    return item
            return None

        for chunk, sf, pct, meta in batch:
            nom_projet = meta.get("nom_projet", "N/A")
            verif      = _find_verif_local(nom_projet)

            if verif is None:
                all_verif_items.append({
                    "titre"           : nom_projet,
                    "decision"        : "keep",
                    "confiance"       : 3,
                    "niveau"          : 3,
                    "raison"          : "Non évalué par le LLM (conservé par défaut)",
                    "criteres_matches": [],
                    "_sf"             : sf,
                    "_meta"           : meta,
                    "_chunk"          : chunk,
                })
            else:
                decision  = verif.get("decision",  "keep")
                confiance = verif.get("confiance", 3)
                niveau    = verif.get("niveau",    confiance)
                raison    = verif.get("raison",    "")
                criteres_matches = verif.get("criteres_matches", [])

                if decision not in ("keep", "exclude"):
                    decision = "keep"
                try:
                    confiance = max(0, min(5, int(confiance)))
                    niveau    = max(0, min(5, int(niveau)))
                except (ValueError, TypeError):
                    confiance = 3
                    niveau    = 3

                all_verif_items.append({
                    "titre"           : nom_projet,
                    "decision"        : decision,
                    "confiance"       : confiance,
                    "niveau"          : niveau,
                    "raison"          : raison,
                    "criteres_matches": criteres_matches,
                    "_sf"             : sf,
                    "_meta"           : meta,
                    "_chunk"          : chunk,
                })

    # ── Construction des listes validated / excluded ─────────
    validated : list = []
    excluded  : list = list(excluded_pre)

    for item in all_verif_items:
        sf        = item["_sf"]
        meta      = item["_meta"]
        chunk     = item["_chunk"]
        decision  = item["decision"]
        confiance = item["confiance"]
        raison    = item["raison"]
        nom_projet = item["titre"]

        if decision == "exclude":
            score_ajuste = 0.0
        else:
            boost        = CONFIANCE_BOOST.get(confiance, 1.0)
            score_ajuste = round(sf * boost, 1)

        valeur = meta.get("valeur", "N/A") or "N/A"
        if valeur != "N/A":
            amount, devise = parse_budget(valeur)
            if amount:
                budget_usd     = convert_to_usd(amount, devise)
                budget_display = f"{int(amount):,} {devise} (~{int(budget_usd):,} USD)"
            else:
                budget_display = valeur
        else:
            budget_display = "N/A"

        entry = {
            "titre"                : nom_projet,
            "annee"                : meta.get("annee",      "N/A"),
            "client"               : meta.get("nom_client", "N/A"),
            "pays"                 : meta.get("pays",       "N/A"),
            "secteur"              : meta.get("secteur",    "N/A"),
            "decision"             : decision,
            "confiance"            : confiance,
            "raison"               : raison,
            "score_final_original" : round(sf, 1),
            "score_final_ajuste"   : score_ajuste,
            "pertinence"           : score_ajuste,
            "matching"             : {},
            "budget"               : budget_display,
            "mots_cles"            : meta.get("mots_cles", ""),
            "_meta"                : meta,
            "_chunk"               : chunk,
        }

        if decision == "exclude":
            excluded.append(entry)
            if verbose:
                print(f"   ❌ Exclu LLM (confiance={confiance}) : {nom_projet[:55]}")
                print(f"      Raison : {raison}")
        else:
            validated.append(entry)
            if verbose:
                conf_bar = "★" * confiance + "☆" * (5 - confiance)
                print(f"   ✅ Conservé [{conf_bar}] score {sf:.1f}%→{score_ajuste:.1f}% : {nom_projet[:45]}")
                print(f"      Raison : {raison}")

    validated.sort(key=lambda x: x["score_final_ajuste"], reverse=True)

    if verbose:
        print(f"\n   📊 Résultat vérification : {len(validated)} conservés, {len(excluded)} exclus")

    return validated, excluded


# ══════════════════════════════════════════════════════════════
# BUILD TOR TABLE
# ══════════════════════════════════════════════════════════════

def build_tor_table(
    validated         : list,
    excluded          : list,
    criteres_tor      : list,
    full_contexts     : dict,
    contrainte        : dict,
    contrainte_budget : dict,
    original_query    : str = "",
    verbose           : bool = True
) -> dict:
    projets_result = []

    for i, entry in enumerate(validated, 1):
        nom_projet = entry["titre"]
        meta       = entry.get("_meta", {})
        chunk      = entry.get("_chunk", "")

        if verbose:
            print(f"\n   [{i:02d}] {nom_projet[:55]}")
            print(f"         année={entry['annee']} | pays={entry['pays']}")

        context  = full_contexts.get(nom_projet, chunk)
        matching = compute_matching_crossencoder(
            projet_context = context,
            criteres       = criteres_tor,
            original_query = original_query,
            verbose        = verbose,
        )

        projets_result.append({
            "titre"      : nom_projet,
            "annee"      : entry["annee"],
            "client"     : entry["client"],
            "pays"       : entry["pays"],
            "secteur"    : entry["secteur"],
            "pertinence" : entry["score_final_ajuste"],
            "confiance"  : entry["confiance"],
            "raison_llm" : entry["raison"],
            "matching"   : matching,
            "budget"     : entry["budget"],
            "mots_cles"  : entry["mots_cles"],
        })

    exclus_result = [
        {
            "titre"          : e["titre"],
            "annee"          : e.get("annee",    "N/A"),
            "client"         : e.get("client",   "N/A"),
            "pays"           : e.get("pays",     "N/A"),
            "raison"         : e.get("raison",   ""),
            "confiance"      : e.get("confiance", 0),
            "score_original" : e.get("score_final_original", 0.0),
        }
        for e in excluded
    ]

    return {
        "criteres"          : criteres_tor,
        "projets"           : projets_result,
        "exclus"            : exclus_result,
        "contrainte"        : contrainte,
        "contrainte_budget" : contrainte_budget,
    }



def detect_secteur(query: str, rewrite: str, seuil_confiance: float = 0.50) -> dict:
    query_combined = f"{query} {rewrite}"
    query_vec = embedder.encode(
        [f"query: {query_combined}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    similarities = [float(np.dot(query_vec, sv)) for sv in _secteur_vecs]

    best_idx     = int(np.argmax(similarities))
    best_score   = similarities[best_idx]
    best_secteur = SECTEURS[best_idx]

    sorted_scores = sorted(similarities, reverse=True)
    second_score  = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    gap = best_score - second_score

    if best_score >= 0.72 and gap >= 0.05:
        confiance = "haute"
        actif = True
    elif best_score >= seuil_confiance and gap >= 0.03:
        confiance = "moyenne"
        actif = True
    elif best_score >= 0.45:
        confiance = "faible"
        actif = False
    else:
        confiance = "aucune"
        actif = False

    return {
        "secteur"   : best_secteur if actif else None,
        "score"     : round(best_score, 3),
        "gap"       : round(gap, 3),
        "confiance" : confiance,
        "actif"     : actif,
        "top3"      : [
            {"secteur": SECTEURS[i], "score": round(similarities[i], 3)}
            for i in np.argsort(similarities)[::-1][:3]
        ],
    }


def _apply_secteur_boost(
    rrf_scores     : dict,
    chunk_registry : dict,
    secteur_info   : dict,
    boost_factor   : float = 1.25,
) -> dict:
    if not secteur_info["actif"] or not secteur_info["secteur"]:
        return rrf_scores

    secteur_cible = secteur_info["secteur"].lower()
    boosted = {}

    for key, score in rrf_scores.items():
        if key not in chunk_registry:
            boosted[key] = score
            continue
        _, meta      = chunk_registry[key]
        secteur_meta = (meta.get("secteur",      "") or "").lower()
        sous_secteur = (meta.get("sous_secteur", "") or "").lower()

        if (secteur_cible in secteur_meta or
                secteur_meta in secteur_cible or
                secteur_cible in sous_secteur):
            factor = boost_factor if secteur_info["confiance"] == "haute" else 1.15
            boosted[key] = score * factor
        else:
            boosted[key] = score

    return dict(sorted(boosted.items(), key=lambda x: x[1], reverse=True))


def _is_query_sector_independent(query: str, criteres: list) -> bool:
    q = query.lower()

    PATTERNS_INDEPENDANTS = [
        r'\b(pays|country|région|tunisie|maroc|algérie|sénégal|mali|côte d.ivoire)\b',
        r'\b(client|bailleur|banque mondiale|ue|usaid|pnud|bnda|bid)\b',
        r'\b(budget|montant|coût)\b',
        r'\b(depuis|avant|entre|année|période|récent)\b',
        r'\b(tous les projets|liste complète|ensemble des)\b',
    ]

    for pattern in PATTERNS_INDEPENDANTS:
        if re.search(pattern, q):
            print(f"   🔍 DEBUG — pattern déclenché : {pattern}")
            return True

    print(f"   🔍 DEBUG — criteres={criteres}, len={len(criteres)}")
    if len(criteres) == 0:
        return True

    return False


# ══════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE RAG
# ══════════════════════════════════════════════════════════════

def ask(query: str, verbose: bool = True) -> dict:

    print(f"\n🔍 Query : {query[:100]}{'...' if len(query) > 100 else ''}")

    print("\n🧠 Extraction unifiée...")
    extracted = extract_all_in_one(query, verbose=verbose)

    main_query        = extracted["rewrite"]
    criteres_tor      = extracted["criteres"]
    contrainte        = extracted["contrainte_temporelle"]
    contrainte_budget = extracted["contrainte_budget"]

    search_queries = [main_query]
    for i in range(0, min(len(criteres_tor), 9), 3):
        group  = criteres_tor[i:i+3]
        phrase = f"projet portant sur {' et '.join(group)}"
        search_queries.append(phrase)
    search_queries = list(dict.fromkeys(search_queries))

    if verbose:
        print(f"\n🔗 Queries ({len(search_queries)}) : {search_queries}")

    print(f"\n📦 Hybrid Search ({len(search_queries)} queries)...")

    chunk_registry   : dict = {}
    all_ranked_lists : list = []

    all_vecs = embedder.encode(
        [f"query: {sq}" for sq in search_queries],
        normalize_embeddings = True,
        batch_size           = len(search_queries),
        show_progress_bar    = False,
    )

    sector_independent = _is_query_sector_independent(query, criteres_tor)

    if sector_independent:
        secteur_info = {"actif": False, "secteur": None, "confiance": "aucune"}
        if verbose:
            print("   ℹ️  Requête indépendante du secteur — pas de boost sectoriel")
    else:
        secteur_info = detect_secteur(query, main_query)
        if verbose:
            print(f"\n🏷️  Secteur détecté : {secteur_info['secteur'] or 'aucun'} "
                  f"(score={secteur_info['score']}, confiance={secteur_info['confiance']})")
            if secteur_info.get("top3"):
                for t in secteur_info["top3"]:
                    print(f"      {t['secteur']:<40} → {t['score']:.3f}")

    for sq, query_vec in zip(search_queries, all_vecs):

        ranked_lists_for_query  = []
        source_labels_for_query = []

        if secteur_info["actif"] and secteur_info["secteur"]:
            try:
                result_sector = collection.query(
                    query_embeddings = [query_vec.tolist()],
                    n_results        = TOP_K_CHROMA,
                    where            = {"secteur": {"$eq": secteur_info["secteur"]}},
                    include          = ["documents", "distances", "metadatas"]
                )
                emb_sector_ranked = []
                for chunk, dist, meta in zip(
                    result_sector["documents"][0],
                    result_sector["distances"][0],
                    result_sector["metadatas"][0]
                ):
                    key = chunk[:80]
                    if key not in chunk_registry:
                        chunk_registry[key] = (chunk, meta)
                    emb_sector_ranked.append(key)

                if emb_sector_ranked:
                    ranked_lists_for_query.append(emb_sector_ranked)
                    source_labels_for_query.append("embedding_sector")

            except Exception as e:
                print(f"   ⚠️  KNN sectoriel échoué : {e}")

        result_global = collection.query(
            query_embeddings = [query_vec.tolist()],
            n_results        = TOP_K_CHROMA,
            include          = ["documents", "distances", "metadatas"]
        )
        emb_global_ranked = []
        for chunk, dist, meta in zip(
            result_global["documents"][0],
            result_global["distances"][0],
            result_global["metadatas"][0]
        ):
            key = chunk[:80]
            if key not in chunk_registry:
                chunk_registry[key] = (chunk, meta)
            emb_global_ranked.append(key)

        ranked_lists_for_query.append(emb_global_ranked)
        source_labels_for_query.append("embedding")

        bm25_results = bm25_search(sq, top_k=TOP_K_BM25)
        bm25_ranked  = []
        for _, bm25_score, chunk, meta in bm25_results:
            key = chunk[:80]
            if key not in chunk_registry:
                chunk_registry[key] = (chunk, meta)
            bm25_ranked.append(key)

        ranked_lists_for_query.append(bm25_ranked)
        source_labels_for_query.append("bm25")

        fused = reciprocal_rank_fusion(
            ranked_lists   = ranked_lists_for_query,
            source_labels  = source_labels_for_query,
            chunk_registry = chunk_registry,
            k              = RRF_K,
        )
        all_ranked_lists.append(list(fused.keys()))

    final_rrf_scores = reciprocal_rank_fusion(
        ranked_lists   = all_ranked_lists,
        chunk_registry = chunk_registry,
        k              = RRF_K,
    )

    final_rrf_scores = _apply_secteur_boost(
        rrf_scores     = final_rrf_scores,
        chunk_registry = chunk_registry,
        secteur_info   = secteur_info,
        boost_factor   = 1.25,
    )
    print(f"📦 {len(chunk_registry)} chunks uniques")

    query_for_reranker = main_query

    print("\n🔄 Reranker (passe unique)...")

    proj_scores_all: dict = defaultdict(list)
    proj_best_chunk: dict = {}

    for key, rrf_score in final_rrf_scores.items():
        if key not in chunk_registry:
            continue
        chunk, meta = chunk_registry[key]
        nom = meta.get("nom_projet", "").strip()
        if not nom:
            continue
        proj_scores_all[nom].append(rrf_score)
        if nom not in proj_best_chunk or rrf_score > proj_best_chunk[nom][2]:
            proj_best_chunk[nom] = (chunk, meta, rrf_score)

    seen_proj: dict = {}
    for nom, scores in proj_scores_all.items():
        top3_avg = sum(sorted(scores, reverse=True)[:3]) / 3
        chunk, meta, _ = proj_best_chunk[nom]
        seen_proj[nom] = (chunk, meta, top3_avg)

    top_by_rrf = sorted(seen_proj.items(), key=lambda x: x[1][2], reverse=True)

    chunks_rerank = [v[0] for _, v in top_by_rrf]
    metas_rerank  = [v[1] for _, v in top_by_rrf]
    noms_rerank   = [n    for n, _ in top_by_rrf]

    scores_rerank = reranker.predict([(query_for_reranker, c) for c in chunks_rerank])

    if verbose:
        print("\n📦 APRÈS RERANKER :")
        ranked_display = sorted(
            zip(noms_rerank, scores_rerank),
            key=lambda x: x[1], reverse=True
        )
        for i, (nom, s) in enumerate(ranked_display[:20], 1):
            print(f"  [{i:02d}] score={s:.2f} | {nom[:55]}")

    rrf_max = max(final_rrf_scores.values()) if final_rrf_scores else 1.0
    rrf_norm: dict = {k: v / rrf_max for k, v in final_rrf_scores.items()}

    W_RERANKER = 0.5
    W_RRF      = 0.5
    

    best_reranker_by_projet : dict = {}
    best_chunk_by_projet    : dict = {}

    for nom, chunk, meta, raw_score in zip(noms_rerank, chunks_rerank, metas_rerank, scores_rerank):
        pct = _crossencoder_score_to_pct(float(raw_score))
        if nom not in best_reranker_by_projet or pct > best_reranker_by_projet[nom]:
            best_reranker_by_projet[nom] = pct
            best_chunk_by_projet[nom]    = (chunk, meta)

    best_rrf_by_projet: dict = {}
    for key, rrf_score_norm in rrf_norm.items():
        if key not in chunk_registry:
            continue
        _, meta = chunk_registry[key]
        nom = meta.get("nom_projet", "").strip()
        if not nom:
            continue
        if nom not in best_rrf_by_projet or rrf_score_norm > best_rrf_by_projet[nom]:
            best_rrf_by_projet[nom] = rrf_score_norm

    projet_scores: list = []
    for nom, (chunk, meta) in best_chunk_by_projet.items():
        pct_reranker = best_reranker_by_projet.get(nom, 0.0)
        pct_rrf      = best_rrf_by_projet.get(nom, 0.0) * 100
        score_final  = (
            W_RERANKER * pct_reranker +
            W_RRF      * pct_rrf
        )
        projet_scores.append((nom, score_final, pct_reranker, pct_rrf))

    projet_scores.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print("\n📦 CLASSEMENT FINAL (reranker + RRF) :")
        for i, (nom, sf, pr, prrf) in enumerate(projet_scores[:20], 1):
            print(
                f"  [{i:02d}] final={sf:.1f}%"
                f" | reranker={pr:.1f}%"
                f" | rrf={prrf:.1f}%"
                f" | {nom[:45]}"
            )

    seen_final, filtered = set(), []
    exclus_temporel = 0
    exclus_budget   = 0

    for nom, score_final, pct_reranker, pct_rrf in projet_scores:
        if nom in seen_final:
            continue
        chunk, meta = best_chunk_by_projet[nom]
        if not respecte_contrainte_temporelle(meta, contrainte):
            exclus_temporel += 1
            if verbose:
                print(f"   ⏭️  Exclu temporel [{meta.get('annee','?')}] : {nom[:50]}")
            continue
        if not respecte_contrainte_budget(meta, contrainte_budget):
            exclus_budget += 1
            if verbose:
                print(f"   💸 Exclu budget [{meta.get('valeur','?')}] : {nom[:50]}")
            continue
        seen_final.add(nom)
        filtered.append((chunk, score_final, score_final, meta))

    if contrainte["type"] != "aucune" and verbose:
        print(f"\n   📅 Filtre temporel : {exclus_temporel} exclu(s) ({contrainte['description']})")
    if contrainte_budget["type"] != "aucune" and verbose:
        print(f"   💰 Filtre budget   : {exclus_budget} exclu(s) ({contrainte_budget['description']})")

    if filtered:
        best_score    = filtered[0][1]
        seuil_dynamic = max(10.0, best_score * 0.35)
        top_candidats = [
            c for c in filtered[:TOP_K_FINAL]
            if c[1] >= seuil_dynamic
        ]
        if verbose:
            print(f"   📊 Seuil dynamique : {seuil_dynamic:.1f}% "
                  f"({len(top_candidats)}/{len(filtered)} candidats)")
    else:
        top_candidats = []

    print(f"\n📄 Récupération full_text ({len(top_candidats)} projets)...")
    full_contexts = {}
    noms_top = [meta.get("nom_projet", "") for _, _, _, meta in top_candidats]

    try:
        results = collection.get(
            where   = {"nom_projet": {"$in": noms_top}},
            include = ["documents", "metadatas"]
        )
        seen_enriched = set()
        proj_chunks: dict = defaultdict(list)
        for chunk, meta in zip(results["documents"], results["metadatas"]):
            key = chunk[:80]
            if key not in seen_enriched:
                seen_enriched.add(key)
                nom = meta.get("nom_projet", "")
                proj_chunks[nom].append((chunk, meta))

        PRIORITY = {
            "identite": 0, "description": 1, "services": 2,
            "impacts": 3, "contexte": 4, "chiffres_cles": 5, "full_text": 6
        }
        for nom, chunks in proj_chunks.items():
            sorted_c = sorted(chunks, key=lambda x: PRIORITY.get(x[1].get("chunk_type", "full_text"), 99))
            parts, total_chars = [], 0
            for doc, _ in sorted_c:
                if total_chars < 6000:
                    parts.append(doc)
                    total_chars += len(doc)
            full_contexts[nom] = "\n\n".join(parts)

    except Exception as e:
        print(f"⚠️  Fallback full_text individuel : {e}")
        for _, _, _, meta in top_candidats:
            nom = meta.get("nom_projet", "")
            if nom and nom not in full_contexts:
                full_contexts[nom] = fetch_full_context_for_project(nom)

    if verbose:
        for nom, ctx in full_contexts.items():
            print(f"   {nom[:50]} → {len(ctx)} chars")

    print(f"\n🤖 LLM Vérificateur ({len(top_candidats)} candidats, batches de {VERIF_BATCH_SIZE})...")

    validated, excluded = llm_verifier(
        projets_candidats = top_candidats,
        original_query    = query,
        rewrite           = main_query,
        criteres          = criteres_tor,
        verbose           = verbose,
    )

    print(f"\n🎯 Scoring CrossEncoder ({len(validated)} projets validés × {len(criteres_tor)} critères)...")

    clean_budget = contrainte_budget.copy()
    if clean_budget["type"] == "aucune":
        clean_budget["description"] = ""

    table_result = build_tor_table(
        validated         = validated,
        excluded          = excluded,
        criteres_tor      = criteres_tor,
        full_contexts     = full_contexts,
        contrainte        = contrainte,
        contrainte_budget = clean_budget,
        original_query    = query,
        verbose           = verbose,
    )

    n_valides = len(table_result.get("projets", []))
    n_exclus  = len(table_result.get("exclus",  []))
    print(f"\n✅ Tableau final : {n_valides} projets validés, {n_exclus} exclus")
    print(f"   Critères : {len(criteres_tor)}")

    answer = f"J'ai trouvé {n_valides} projet(s) correspondant à votre recherche."
    if n_exclus:
        answer += f" {n_exclus} projet(s) ont été exclus après vérification."
    if contrainte["type"] != "aucune":
        answer += f" Filtre appliqué : {contrainte['description']}."
    if contrainte_budget["type"] != "aucune":
        answer += f" Filtre budget : {contrainte_budget['description']}."

    return {
        "table" : table_result,
        "answer": answer,
    }


# ══════════════════════════════════════════════════════════════
# MODE INTERACTIF
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          Matine Mind — RAG Hybrid Search + LLM Verifier     ║")
    print("║  📄 Envoyez un chemin vers un TDR (PDF/DOCX)                 ║")
    print("║  💬 Ou tapez directement votre question                      ║")
    print("║  Tapez 'exit' pour quitter                                   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    while True:
        user_input = input("❓ TDR ou Question : ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("👋 Au revoir !")
            break
        if not user_input:
            continue

        result = smart_dispatch(
            user_input     = user_input,
            call_ollama_fn = call_ollama,
            ask_fn         = ask,
            model          = OLLAMA_EXTRACTOR,
            verbose        = True,
        )

        print(f"\n{'═'*65}")

        if "tdr_analysis" in result:
            analysis = result["tdr_analysis"]
            src      = result.get("tdr_source", {})
            print(f"📄 SOURCE : {src.get('filename', '?')} ({src.get('format', '?')})")
            print(f"📌 CONTEXTE : {analysis.get('contexte', '')[:150]}...")
            print(f"🎯 OBJECTIFS :")
            for i, o in enumerate(analysis.get("objectifs", [])[:6], 1):
                print(f"   [{i}] {o[:75]}")
            if analysis.get("methodologie_demandee"):
                print(f"🔧 MÉTHODOLOGIE : {analysis['methodologie_demandee']}")
            if analysis.get("criteres_qualification"):
                print(f"📋 QUALIF. : {', '.join(analysis['criteres_qualification'][:3])}")
            print()

        print(result["answer"])

        contrainte = result["table"].get("contrainte", {})
        if contrainte.get("type") != "aucune":
            print(f"📅 {contrainte.get('description', '')}")

        print(f"\n📊 Critères : {result['table']['criteres']}")
        print(f"{'─'*65}")

        projets = result["table"].get("projets", [])
        if projets:
            print(f"\n✅ PROJETS VALIDÉS ({len(projets)}) :")
        for p in projets:
            conf = p.get("confiance")
            if conf is not None:
                conf_bar = "★" * conf + "☆" * (5 - conf)
                conf_str = f" [{conf_bar}]"
            else:
                conf_str = " [fallback]"

            print(f"\n  [{p['annee']}] {p['titre'][:55]}")
            print(f"  Pays : {p['pays']} | Client : {p['client'][:30]}")
            print(f"  Budget : {p.get('budget', 'N/A')} | Score : {p['pertinence']}%{conf_str}")
            if p.get("raison_llm"):
                print(f"  LLM : {p['raison_llm']}")
            if p.get('mots_cles'):
                tags = p['mots_cles'].split(' | ')[:4]
                print(f"  Tags : {' · '.join(tags)}")
            print(f"  Matching par critère :")
            for cr, score in p["matching"].items():
                bar   = "█" * int(score / 10) + "░" * (10 - int(score / 10))
                color = "✅" if score >= 70 else ("⚠️ " if score >= 40 else "❌")
                print(f"    {color} {cr[:28]:28s} [{bar}] {score:.0f}%")

        exclus = result["table"].get("exclus", [])
        if exclus:
            print(f"\n❌ PROJETS EXCLUS ({len(exclus)}) :")
            for e in exclus:
                print(f"  • [{e.get('annee','?')}] {e['titre'][:55]}")
                print(f"    Raison : {e.get('raison', 'N/A')}")

        print(f"\n{'═'*65}\n")