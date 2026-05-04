import sys
import json
import math
from collections import defaultdict

# ─────────────────────────────────────────────
# IMPORT DU RAG (CORRIGÉ)
# ─────────────────────────────────────────────
sys.path.insert(0, r"C:\Users\Chayma MAJJEDI\Desktop\chatbot_web\new pipeline")

try:
    from rag_query import ask   # ⚠️ change si ton fichier a un autre nom
    RAG_AVAILABLE = True
    print("✅ RAG importé")
except ImportError as e:
    print(f"❌ Import RAG échoué: {e}")
    RAG_AVAILABLE = False


# ─────────────────────────────────────────────
# EXTRACTION DES PROJETS (CORRIGÉ)
# ─────────────────────────────────────────────
def extract_project_names(response):
    try:
        projets = response["table"]["projets"]
        return [p["titre"] for p in projets]
    except Exception:
        return []


# ─────────────────────────────────────────────
# MATCHING FLEXIBLE (IMPORTANT)
# ─────────────────────────────────────────────
def match_projects(expected, found):
    expected = [e.lower() for e in expected]
    found = [f.lower() for f in found]

    matches = []
    for e in expected:
        for f in found:
            if e in f or f in e:
                matches.append(f)
                break
    return matches


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def precision_at_k(found, expected, k):
    found_k = found[:k]
    match = match_projects(expected, found_k)
    return len(match) / k if k > 0 else 0


def recall_at_k(found, expected, k):
    found_k = found[:k]
    match = match_projects(expected, found_k)
    return len(match) / len(expected) if expected else 0


def f1(p, r):
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)


def hit_at_k(found, expected, k):
    found_k = found[:k]
    match = match_projects(expected, found_k)
    return 1 if len(match) > 0 else 0


def ndcg_at_k(found, expected, k):
    dcg = 0
    for i, f in enumerate(found[:k]):
        rel = 1 if any(e.lower() in f.lower() for e in expected) else 0
        dcg += rel / math.log2(i + 2)

    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(expected), k)))
    return dcg / idcg if idcg > 0 else 0


# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
with open("basetest.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
K_VALUES = [1, 3, 5, 10, 15, 20]

metrics = defaultdict(list)

print("\n============================================================")
print("📊 RAG EVALUATION")
print("============================================================\n")

for i, item in enumerate(dataset, 1):

    query = item["query"]
    expected = item["expected_projects"]

    print(f"\n🔎 Q{i}: {query}")

    if RAG_AVAILABLE:
        response = ask(query, verbose=False)
        found = extract_project_names(response)
    else:
        found = []

    # DEBUG IMPORTANT
    print("Expected:", expected)
    print("Found:", found)

    for k in K_VALUES:
        p = precision_at_k(found, expected, k)
        r = recall_at_k(found, expected, k)

        metrics[f"P@{k}"].append(p)
        metrics[f"R@{k}"].append(r)
        metrics[f"F1@{k}"].append(f1(p, r))
        metrics[f"Hit@{k}"].append(hit_at_k(found, expected, k))
        metrics[f"NDCG@{k}"].append(ndcg_at_k(found, expected, k))


# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────
print("\n============================================================")
print("📊 RESULTS")
print("============================================================\n")

for k in K_VALUES:
    print(f"\n--- K = {k} ---")
    print(f"Hit@{k}:  {sum(metrics[f'Hit@{k}'])/len(dataset):.3f}")
    print(f"P@{k}:    {sum(metrics[f'P@{k}'])/len(dataset):.3f}")
    print(f"R@{k}:    {sum(metrics[f'R@{k}'])/len(dataset):.3f}")
    print(f"F1@{k}:   {sum(metrics[f'F1@{k}'])/len(dataset):.3f}")
    print(f"NDCG@{k}: {sum(metrics[f'NDCG@{k}'])/len(dataset):.3f}")