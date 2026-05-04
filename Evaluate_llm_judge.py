print("🚀 Démarrage...")
import json
import re
import requests
from rag_query import ask

# ───────────── CONFIG ─────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
JUDGE_MODEL  = "gemma2:9b"   # ou qwen2.5:14b si tu l'as
OUTPUT_PATH  = r"C:\Users\Chayma MAJJEDI\Desktop\chatbot_web\new pipeline\eval_results.json"

# ───────────── Questions de test ─────────────
# Pas besoin de réponses attendues pour le LLM-as-a-judge
TEST_QUESTIONS = [
    "Avons-nous des références sur les études de marché TIC en Afrique ?",
    "Quels projets avons-nous réalisés pour la Banque Mondiale ?",
    "Donnez-moi des références sur la formation professionnelle et l'emploi des jeunes en Tunisie.",
    "Avons-nous travaillé sur des projets liés à l'hydrogène vert ou aux énergies renouvelables ?",
    "Quels sont nos projets dans le secteur agricole au Maghreb ?",
    "Avons-nous des références sur la stratégie nationale pour l'emploi ?",
    "Donnez-moi des projets liés au diagnostic du marché du travail.",
    "Avons-nous travaillé sur des projets de due diligence commerciale ?",
    "Quels projets avons-nous réalisés en Arabie Saoudite ?",
    "Donnez-moi des références sur la transformation digitale et les fintechs.",
]

# ───────────── Appel LLM juge ─────────────
def call_judge(prompt: str) -> str:
    payload = {
        "model"  : JUDGE_MODEL,
        "prompt" : prompt,
        "stream" : False,
        "options": {"temperature": 0}
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=360)
        resp.raise_for_status()
        return resp.json()["response"]
    except Exception as e:
        return f"❌ Erreur juge : {e}"

# ───────────── Prompt du juge ─────────────
def build_judge_prompt(question: str, answer: str) -> str:
    return f"""Tu es un expert évaluateur pour un cabinet de consulting.
Tu dois évaluer la qualité de la réponse d'un chatbot de recherche de références projets.

QUESTION POSÉE :
{question}

RÉPONSE DU CHATBOT :
{answer}

Évalue la réponse sur ces 4 critères. Pour chaque critère, donne un score entre 0 et 1 et une justification courte.

CRITÈRES :
1. pertinence     : Les projets retournés sont-ils pertinents par rapport à la question ?
2. complétude     : Le chatbot a-t-il retourné suffisamment de projets pertinents ?
3. format         : La réponse est-elle bien structurée et lisible ?
4. hallucination  : Le chatbot a-t-il inventé des informations (0 = beaucoup d'hallucinations, 1 = aucune) ?

Réponds UNIQUEMENT avec un JSON valide, sans texte avant ou après :
{{
  "pertinence"   : {{"score": 0.0, "justification": "..."}},
  "complétude"   : {{"score": 0.0, "justification": "..."}},
  "format"       : {{"score": 0.0, "justification": "..."}},
  "hallucination": {{"score": 0.0, "justification": "..."}},
  "score_global" : 0.0,
  "commentaire"  : "..."
}}
"""

# ───────────── Parsing JSON juge ─────────────
def parse_judge_response(raw: str) -> dict:
    try:
        clean = re.sub(r"```json|```", "", raw.strip()).strip()
        return json.loads(clean)
    except Exception:
        # Fallback : cherche le JSON dans la réponse
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return {
            "pertinence"   : {"score": 0, "justification": "parsing échoué"},
            "complétude"   : {"score": 0, "justification": "parsing échoué"},
            "format"       : {"score": 0, "justification": "parsing échoué"},
            "hallucination": {"score": 0, "justification": "parsing échoué"},
            "score_global" : 0,
            "commentaire"  : raw[:200]
        }

# ───────────── Évaluation principale ─────────────
def evaluate():
    results  = []
    scores   = {"pertinence": [], "complétude": [], "format": [], "hallucination": [], "global": []}

    print("🤖 Démarrage de l'évaluation LLM-as-a-judge\n")
    print("=" * 60)

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] Question : {question[:70]}...")

        # ── Appel RAG ──
        print("  ⏳ Appel RAG...")
        try:
            answer = ask(question, verbose=False)
        except Exception as e:
            answer = f"❌ Erreur RAG : {e}"
        print(f"  ✅ Réponse obtenue ({len(answer)} chars)")

        # ── Jugement ──
        print("  🧑‍⚖️  Jugement en cours...")
        judge_prompt    = build_judge_prompt(question, answer)
        judge_raw       = call_judge(judge_prompt)
        judge_result    = parse_judge_response(judge_raw)

        # ── Calcul score global si absent ──
        if not judge_result.get("score_global"):
            s = judge_result
            judge_result["score_global"] = round((
                s.get("pertinence",    {}).get("score", 0) * 0.35 +
                s.get("complétude",    {}).get("score", 0) * 0.25 +
                s.get("format",        {}).get("score", 0) * 0.20 +
                s.get("hallucination", {}).get("score", 0) * 0.20
            ), 2)

        # ── Affichage ──
        sg = judge_result.get("score_global", 0)
        print(f"  📊 Score global : {sg:.2f}/1.0")
        print(f"     pertinence    : {judge_result.get('pertinence',    {}).get('score', '?')}")
        print(f"     complétude    : {judge_result.get('complétude',    {}).get('score', '?')}")
        print(f"     format        : {judge_result.get('format',        {}).get('score', '?')}")
        print(f"     hallucination : {judge_result.get('hallucination', {}).get('score', '?')}")
        if judge_result.get("commentaire"):
            print(f"     commentaire   : {judge_result['commentaire'][:100]}")

        # ── Stockage ──
        scores["pertinence"].append(judge_result.get("pertinence",    {}).get("score", 0))
        scores["complétude"].append(judge_result.get("complétude",    {}).get("score", 0))
        scores["format"].append(judge_result.get("format",            {}).get("score", 0))
        scores["hallucination"].append(judge_result.get("hallucination", {}).get("score", 0))
        scores["global"].append(sg)

        results.append({
            "question"     : question,
            "answer"       : answer,
            "judge_result" : judge_result,
        })

    # ── Résumé final ──
    print("\n" + "=" * 60)
    print("📈 RÉSUMÉ DE L'ÉVALUATION\n")
    print(f"  Pertinence    moyenne : {sum(scores['pertinence'])    / len(scores['pertinence']):.2f}")
    print(f"  Complétude    moyenne : {sum(scores['complétude'])    / len(scores['complétude']):.2f}")
    print(f"  Format        moyen   : {sum(scores['format'])        / len(scores['format']):.2f}")
    print(f"  Hallucination moyenne : {sum(scores['hallucination']) / len(scores['hallucination']):.2f}")
    print(f"\n  ⭐ Score global moyen  : {sum(scores['global'])       / len(scores['global']):.2f} / 1.0")

    # ── Sauvegarde ──
    output = {
        "summary": {
            "pertinence_avg"    : round(sum(scores["pertinence"])    / len(scores["pertinence"]),    2),
            "complétude_avg"    : round(sum(scores["complétude"])    / len(scores["complétude"]),    2),
            "format_avg"        : round(sum(scores["format"])        / len(scores["format"]),        2),
            "hallucination_avg" : round(sum(scores["hallucination"]) / len(scores["hallucination"]), 2),
            "global_avg"        : round(sum(scores["global"])        / len(scores["global"]),        2),
        },
        "results": results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Résultats sauvegardés : {OUTPUT_PATH}")
    return output

# ───────────── Main ─────────────
if __name__ == "__main__":
    evaluate()