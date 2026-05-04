"""
evaluate_rag.py  —  Pipeline d'évaluation complet du système RAG Matine Mind
=============================================================================
Métriques : Recall@K, Precision@K, MRR, Hit Rate, F1@K
Matching   : LLM via Ollama (matching sémantique flou des noms de projets)
"""

import json
import re
import math
import time
import datetime
import argparse
from pathlib import Path
from typing import Optional

import requests

# ── Import du pipeline RAG existant ──────────────────────────────────────────
# On importe ask() depuis le fichier principal (ajuster le nom si besoin)
try:
    from rag_query import ask, call_ollama, OLLAMA_EXTRACTOR
    RAG_IMPORTED = True
except ImportError:
    RAG_IMPORTED = False
    print("⚠️  Impossible d'importer rag_pipeline.py — mode simulation activé")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_EXTRACTOR = "qwen2.5:7b"
K_VALUES         = [5, 10]          # Recall/Precision@K à calculer
MATCH_THRESHOLD  = 0.70             # Score LLM au-dessus duquel = match (0-1)
OUTPUT_DIR       = Path("eval_results")


# ══════════════════════════════════════════════════════════════════════════════
# MATCHING LLM — CŒUR DE L'ÉVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def call_ollama_raw(system: str, user: str) -> str:
    payload = {
        "model":  OLLAMA_EXTRACTOR,
        "prompt": f"{system}\n\n{user}",
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"]
    except Exception as e:
        return f'{{"error": "{e}"}}'


def parse_json_safe(raw: str) -> Optional[dict]:
    clean = re.sub(r"```json|```", "", raw.strip()).strip()
    start = clean.find("{")
    if start == -1:
        return None
    depth, end = 0, -1
    for i, ch in enumerate(clean[start:], start):
        if ch == "{": depth += 1
        elif ch == "}": depth -= 1
        if depth == 0:
            end = i + 1
            break
    if end == -1:
        return None
    try:
        return json.loads(clean[start:end])
    except json.JSONDecodeError:
        return None


def llm_match_projects(
    expected_list: list[str],
    retrieved_list: list[str],
    verbose: bool = False
) -> dict[str, dict]:
    """
    Pour chaque projet attendu, demande à l'LLM si l'un des projets
    récupérés lui correspond (matching sémantique flou, bilingue FR/EN).

    Retourne : {expected_name: {"matched": bool, "best_match": str|None, "score": float, "reason": str}}
    """
    if not expected_list or not retrieved_list:
        return {e: {"matched": False, "best_match": None, "score": 0.0, "reason": "liste vide"} for e in expected_list}

    system = """/no_think
Tu es un expert en matching de noms de projets de consulting.
Ta tâche : pour chaque projet attendu, trouver le projet récupéré qui correspond le mieux,
même si les noms sont formulés différemment, abrégés, ou dans une autre langue (FR/EN).

RÈGLES DE MATCHING :
- Même projet = même mission / même client / même thème central, même s'il y a des variations de libellé
- Un acronyme peut matcher le titre complet
- Ignore les articles, prépositions et légères différences orthographiques
- Si le projet n'a aucun équivalent dans la liste récupérée → score 0.0

Réponds UNIQUEMENT en JSON valide :
{
  "matches": [
    {
      "expected": "nom exact du projet attendu",
      "best_retrieved": "nom du projet récupéré correspondant ou null",
      "score": 0.85,
      "reason": "courte explication (max 10 mots)"
    }
  ]
}
"""

    # On limite pour ne pas dépasser le contexte
    retrieved_str = "\n".join([f"  [{i+1}] {name}" for i, name in enumerate(retrieved_list[:30])])
    expected_str  = "\n".join([f"  - {name}" for name in expected_list])

    user = f"""PROJETS ATTENDUS :
{expected_str}

PROJETS RÉCUPÉRÉS PAR LE SYSTÈME :
{retrieved_str}

Pour chaque projet attendu, donne le meilleur match parmi les projets récupérés."""

    raw    = call_ollama_raw(system, user)
    result = parse_json_safe(raw)

    if not result or "matches" not in result:
        if verbose:
            print(f"   ⚠️  Parsing LLM échoué — raw: {raw[:200]}")
        return {e: {"matched": False, "best_match": None, "score": 0.0, "reason": "parsing LLM échoué"} for e in expected_list}

    matching_results = {}
    for m in result.get("matches", []):
        expected     = m.get("expected", "")
        best         = m.get("best_retrieved")
        score        = float(m.get("score", 0.0))
        reason       = m.get("reason", "")
        matched      = score >= MATCH_THRESHOLD and best is not None

        # Cherche la clé la plus proche dans expected_list (tolérance casse)
        key = expected
        for e in expected_list:
            if e.lower().strip() == expected.lower().strip():
                key = e
                break

        matching_results[key] = {
            "matched"   : matched,
            "best_match": best,
            "score"     : round(score, 3),
            "reason"    : reason,
        }

    # Fallback pour les projets non traités par le LLM
    for e in expected_list:
        found = False
        for key in matching_results:
            if key.lower().strip() == e.lower().strip():
                found = True
                break
        if not found:
            matching_results[e] = {"matched": False, "best_match": None, "score": 0.0, "reason": "non traité"}

    return matching_results


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    expected_projects : list[str],
    retrieved_projects: list[str],
    matching_results  : dict,
    k_values          : list[int] = K_VALUES,
) -> dict:
    n_expected = len(expected_projects)
    metrics    = {}

    # ── Construire retrieved_is_tp avec matching insensible à la casse ──────
    retrieved_is_tp: dict[str, bool] = {}

    # Index normalisé des projets récupérés → nom original
    retrieved_normalized = {p.lower().strip(): p for p in retrieved_projects}

    for info in matching_results.values():
        if info["matched"] and info["best_match"]:
            best = info["best_match"]
            best_norm = best.lower().strip()

            # 1) Correspondance exacte (insensible à la casse)
            if best_norm in retrieved_normalized:
                original = retrieved_normalized[best_norm]
                retrieved_is_tp[original] = True
                continue

            # 2) Correspondance partielle : best est contenu dans un retrieved
            #    ou vice-versa (pour les titres tronqués/abrégés)
            matched_any = False
            for r_norm, r_orig in retrieved_normalized.items():
                if best_norm in r_norm or r_norm in best_norm:
                    retrieved_is_tp[r_orig] = True
                    matched_any = True
                    break

            # 3) Fallback : le LLM a parfois renvoyé seulement les N premiers mots
            if not matched_any:
                best_words = set(best_norm.split())
                for r_norm, r_orig in retrieved_normalized.items():
                    r_words = set(r_norm.split())
                    overlap = len(best_words & r_words)
                    union   = len(best_words | r_words)
                    jaccard = overlap / union if union > 0 else 0
                    if jaccard >= 0.5:           # ≥50% de mots en commun
                        retrieved_is_tp[r_orig] = True
                        break

    # ── MRR ──────────────────────────────────────────────────────────────────
    mrr_rank = None
    for rank, proj in enumerate(retrieved_projects, start=1):
        if retrieved_is_tp.get(proj, False):
            mrr_rank = rank
            break
    metrics["MRR"] = round(1.0 / mrr_rank, 4) if mrr_rank else 0.0

    # ── Hit Rate ──────────────────────────────────────────────────────────────
    metrics["HitRate"] = 1 if any(
        retrieved_is_tp.get(p, False) for p in retrieved_projects
    ) else 0

    # ── Recall@K, Precision@K, F1@K ─────────────────────────────────────────
    for k in k_values:
        top_k     = retrieved_projects[:k]
        tp_at_k   = sum(1 for p in top_k if retrieved_is_tp.get(p, False))
        recall    = round(tp_at_k / n_expected, 4) if n_expected > 0 else 0.0
        precision = round(tp_at_k / k,          4) if k > 0          else 0.0
        f1        = round(
            2 * precision * recall / (precision + recall), 4
        ) if (precision + recall) > 0 else 0.0
        metrics[f"Recall@{k}"]    = recall
        metrics[f"Precision@{k}"] = precision
        metrics[f"F1@{k}"]        = f1

    metrics["n_expected"]  = n_expected
    metrics["n_retrieved"] = len(retrieved_projects)
    metrics["n_matched"]   = sum(1 for info in matching_results.values() if info["matched"])

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC PAR QUESTION
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_failures(
    expected_projects : list[str],
    retrieved_projects: list[str],
    matching_results  : dict,
) -> dict:
    """
    Identifie les projets attendus non trouvés et classe les causes probables.
    """
    not_found     = []
    found_late    = []  # trouvé mais pas dans top-5
    found_ok      = []

    top5 = set(retrieved_projects[:5])

    for expected, info in matching_results.items():
        if not info["matched"]:
            not_found.append({
                "project" : expected,
                "reason"  : info.get("reason", ""),
                "best_llm": info.get("best_match"),
                "score"   : info.get("score", 0.0),
            })
        else:
            best = info["best_match"]
            if best and best not in top5:
                found_late.append({
                    "project"  : expected,
                    "found_as" : best,
                    "position" : retrieved_projects.index(best) + 1 if best in retrieved_projects else -1,
                })
            else:
                found_ok.append({
                    "project" : expected,
                    "found_as": best,
                })

    # Inférence des causes de non-trouvaille
    for item in not_found:
        score = item["score"]
        if score >= 0.5:
            item["probable_cause"] = "low_rank"         # trouvé mais mal classé
        elif item["best_llm"]:
            item["probable_cause"] = "partial_match"    # vague correspondance
        else:
            item["probable_cause"] = "not_in_db_or_naming"  # absent ou nommage très différent

    return {
        "found_ok"  : found_ok,
        "found_late": found_late,
        "not_found" : not_found,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    test_file : str,
    output_dir: Path = OUTPUT_DIR,
    verbose   : bool = True,
    dry_run   : bool = False,
) -> dict:
    """
    Lance l'évaluation complète sur le fichier JSON de test.
    dry_run=True : génère des projets fictifs (pour tester le pipeline sans RAG).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"\n{'='*60}")
    print(f"  ÉVALUATION RAG — {len(test_cases)} questions")
    print(f"{'='*60}\n")

    all_results = []

    for idx, case in enumerate(test_cases, start=1):
        case_id   = case.get("id", idx)
        query     = case["query"]
        expected  = case["expected_projects"]

        print(f"\n[{idx:02d}/{len(test_cases)}] Q{case_id}: {query[:80]}...")
        print(f"       {len(expected)} projets attendus")

        # ── Appel au système RAG ──────────────────────────────────────────────
        t0 = time.time()

        if dry_run or not RAG_IMPORTED:
            # Simulation : on retourne des projets fictifs pour tester le pipeline
            import random
            fake = expected[:5] + [f"Projet fictif {i}" for i in range(5)]
            random.shuffle(fake)
            rag_result = {
                "table": {
                    "projets": [{"titre": p, "annee": "2022", "pays": "Tunisie",
                                 "pertinence": round(random.uniform(50, 95), 1)} for p in fake]
                },
                "answer": f"Simulation : {len(fake)} projets"
            }
        else:
            try:
                rag_result = ask(query, verbose=False)
            except Exception as e:
                print(f"   ❌ Erreur RAG : {e}")
                rag_result = {"table": {"projets": []}, "answer": str(e)}

        elapsed = round(time.time() - t0, 2)

        # ── Extraction des projets retournés (ordonnés par pertinence) ───────
        projets_raw = rag_result.get("table", {}).get("projets", [])
        retrieved   = [p["titre"] for p in projets_raw]

        print(f"       {len(retrieved)} projets récupérés en {elapsed}s")

        # ── Matching LLM ──────────────────────────────────────────────────────
        print(f"       Matching LLM ({len(expected)} × {len(retrieved)})...")
        matching = llm_match_projects(expected, retrieved, verbose=verbose)

        n_matched = sum(1 for v in matching.values() if v["matched"])
        print(f"       {n_matched}/{len(expected)} projets matchés (seuil={MATCH_THRESHOLD})")

        # ── Métriques ─────────────────────────────────────────────────────────
        metrics   = compute_metrics(expected, retrieved, matching)
        diagnosis = diagnose_failures(expected, retrieved, matching)

        if verbose:
            for k in K_VALUES:
                print(f"       Recall@{k}={metrics[f'Recall@{k}']:.2f}  "
                      f"Precision@{k}={metrics[f'Precision@{k}']:.2f}  "
                      f"F1@{k}={metrics[f'F1@{k}']:.2f}")
            print(f"       MRR={metrics['MRR']:.3f}  HitRate={metrics['HitRate']}")

            if diagnosis["not_found"]:
                print(f"       ⚠️  Non trouvés ({len(diagnosis['not_found'])}) :")
                for item in diagnosis["not_found"][:3]:
                    print(f"          - {item['project'][:60]} [{item['probable_cause']}]")

        # ── Projets récupérés avec détail ─────────────────────────────────────
        retrieved_detail = []
        for p in projets_raw:
            retrieved_detail.append({
                "titre"     : p["titre"],
                "annee"     : p.get("annee", ""),
                "pays"      : p.get("pays", ""),
                "pertinence": p.get("pertinence", 0),
            })

        all_results.append({
            "id"              : case_id,
            "query"           : query,
            "expected_count"  : len(expected),
            "expected_projects": expected,
            "retrieved_count" : len(retrieved),
            "retrieved_detail": retrieved_detail,
            "matching"        : matching,
            "metrics"         : metrics,
            "diagnosis"       : diagnosis,
            "elapsed_s"       : elapsed,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # AGRÉGATION GLOBALE
    # ══════════════════════════════════════════════════════════════════════════
    n = len(all_results)

    def avg(metric_key):
        vals = [r["metrics"].get(metric_key, 0) for r in all_results]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    global_metrics = {
        "n_questions" : n,
        "HitRate"     : avg("HitRate"),
        "MRR"         : avg("MRR"),
    }
    for k in K_VALUES:
        global_metrics[f"Recall@{k}"]    = avg(f"Recall@{k}")
        global_metrics[f"Precision@{k}"] = avg(f"Precision@{k}")
        global_metrics[f"F1@{k}"]        = avg(f"F1@{k}")

    # Diagnostic global : causes de non-trouvaille
    all_not_found = []
    cause_counts  = {}
    for r in all_results:
        for item in r["diagnosis"]["not_found"]:
            all_not_found.append({"query_id": r["id"], **item})
            c = item.get("probable_cause", "unknown")
            cause_counts[c] = cause_counts.get(c, 0) + 1

    global_metrics["total_not_found"]  = len(all_not_found)
    global_metrics["failure_causes"]   = cause_counts

    # ── Affichage résumé ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RÉSULTATS GLOBAUX ({n} questions)")
    print(f"{'='*60}")
    print(f"  Hit Rate     : {global_metrics['HitRate']:.2%}")
    print(f"  MRR          : {global_metrics['MRR']:.4f}")
    for k in K_VALUES:
        print(f"  Recall@{k}     : {global_metrics[f'Recall@{k}']:.2%}")
        print(f"  Precision@{k}  : {global_metrics[f'Precision@{k}']:.2%}")
        print(f"  F1@{k}         : {global_metrics[f'F1@{k}']:.2%}")
    print(f"\n  Projets non trouvés : {global_metrics['total_not_found']}")
    print(f"  Causes principales  : {cause_counts}")
    print(f"{'='*60}\n")

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp"     : timestamp,
        "config"        : {
            "match_threshold": MATCH_THRESHOLD,
            "k_values"       : K_VALUES,
            "ollama_model"   : OLLAMA_EXTRACTOR,
        },
        "global_metrics": global_metrics,
        "per_question"  : all_results,
    }

    out_path = output_dir / f"eval_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"📁 Résultats sauvegardés → {out_path}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# RAPPORT MARKDOWN
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(eval_result: dict, output_dir: Path = OUTPUT_DIR) -> Path:
    """Génère un rapport Markdown lisible depuis les résultats d'évaluation."""
    gm   = eval_result["global_metrics"]
    ts   = eval_result["timestamp"]
    cfg  = eval_result["config"]

    lines = [
        f"# Rapport d'évaluation RAG — {ts}",
        "",
        "## Configuration",
        f"- Modèle matching : `{cfg['ollama_model']}`",
        f"- Seuil de match  : `{cfg['match_threshold']}`",
        f"- K évalués       : `{cfg['k_values']}`",
        "",
        "## Résultats globaux",
        "",
        f"| Métrique | Score |",
        f"|----------|-------|",
        f"| Hit Rate | **{gm['HitRate']:.2%}** |",
        f"| MRR | {gm['MRR']:.4f} |",
    ]
    for k in cfg["k_values"]:
        lines += [
            f"| Recall@{k} | {gm[f'Recall@{k}']:.2%} |",
            f"| Precision@{k} | {gm[f'Precision@{k}']:.2%} |",
            f"| F1@{k} | {gm[f'F1@{k}']:.2%} |",
        ]
    lines += [
        "",
        f"**Projets non trouvés (total)** : {gm['total_not_found']}",
        "",
        "### Causes de non-trouvaille",
        "",
    ]
    for cause, count in gm.get("failure_causes", {}).items():
        pct = count / max(gm['total_not_found'], 1) * 100
        desc = {
            "low_rank"            : "🔻 Projet présent mais mal classé (problème reranker/retrieval)",
            "partial_match"       : "🔶 Correspondance partielle (nommage différent, score LLM 0.5-0.7)",
            "not_in_db_or_naming" : "❌ Absent de la DB ou nommage trop différent",
        }.get(cause, cause)
        lines.append(f"- **{cause}** ({count} cas, {pct:.0f}%) : {desc}")

    lines += ["", "---", "", "## Détail par question", ""]

    for r in eval_result["per_question"]:
        m    = r["metrics"]
        diag = r["diagnosis"]
        lines += [
            f"### Q{r['id']} — {r['query'][:70]}",
            "",
            f"- Attendus : {r['expected_count']} | Récupérés : {r['retrieved_count']} | "
            f"Matchés : {m['n_matched']} | Durée : {r['elapsed_s']}s",
        ]
        for k in cfg["k_values"]:
            lines.append(f"- Recall@{k}={m[f'Recall@{k}']:.2%}  "
                         f"Precision@{k}={m[f'Precision@{k}']:.2%}  "
                         f"F1@{k}={m[f'F1@{k}']:.2%}")
        lines.append(f"- MRR={m['MRR']:.3f}  HitRate={m['HitRate']}")

        if diag["found_ok"]:
            lines.append(f"\n**Trouvés correctement** ({len(diag['found_ok'])}) :")
            for item in diag["found_ok"][:5]:
                lines.append(f"  - ✅ {item['project'][:60]}")

        if diag["found_late"]:
            lines.append(f"\n**Trouvés tardivement** ({len(diag['found_late'])}) :")
            for item in diag["found_late"]:
                lines.append(f"  - 🔻 Pos.{item['position']} : {item['project'][:55]}")

        if diag["not_found"]:
            lines.append(f"\n**Non trouvés** ({len(diag['not_found'])}) :")
            for item in diag["not_found"]:
                cause_emoji = {"low_rank": "🔻", "partial_match": "🔶",
                               "not_in_db_or_naming": "❌"}.get(item.get("probable_cause"), "❓")
                lines.append(f"  - {cause_emoji} [{item.get('probable_cause','')}] {item['project'][:55]}")
                if item.get("best_llm"):
                    lines.append(f"       → LLM suggère : {item['best_llm'][:50]} (score={item['score']:.2f})")

        lines.append("")

    report_path = output_dir / f"report_{ts}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📄 Rapport Markdown → {report_path}")
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation du système RAG Matine Mind")
    parser.add_argument("test_file",         help="Chemin vers le JSON de test")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Mode simulation (pas d'appel RAG réel)")
    parser.add_argument("--output-dir",      default="eval_results",
                        help="Dossier de sortie des résultats")
    parser.add_argument("--threshold",       type=float, default=0.70,
                        help="Seuil de matching LLM (0-1, défaut=0.70)")
    parser.add_argument("--no-report",       action="store_true",
                        help="Ne génère pas le rapport Markdown")
    parser.add_argument("-v", "--verbose",   action="store_true", default=True)
    args = parser.parse_args()

    MATCH_THRESHOLD = args.threshold
    out_dir         = Path(args.output_dir)

    result = run_evaluation(
        test_file  = args.test_file,
        output_dir = out_dir,
        verbose    = args.verbose,
        dry_run    = args.dry_run,
    )

    if not args.no_report:
        generate_report(result, out_dir)
print("SCRIPT STARTED")        