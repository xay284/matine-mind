from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
import json
import re
import requests
import time

# ───────────── CONFIG ─────────────
DOCX_PATH        = r"C:\Users\Chayma MAJJEDI\Desktop\chatbot_web\new pipeline\Projets2.docx"
OUTPUT_PATH      = r"C:\Users\Chayma MAJJEDI\Desktop\chatbot_web\new pipeline\output.json"
MATCH_THRESHOLD  = 0.15
verbose_matching = True

# LLM pour normalisation des fiches
OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_MODEL     = "qwen2.5:7b"   # modèle rapide pour l'extraction
LLM_BATCH_SIZE   = 1              # traitement fiche par fiche (safe)

doc = Document(DOCX_PATH)

# ───────────── Mapping manuel ─────────────
MANUAL_MAPPING = {
    "Actualisation des plans d'études des filières STEM":
        "Actualisation des plans d'études des filières STEM",
    "Formulation SNE à horizon 2030":
        "Assistance à la définition de la Stratégie de l'Emploi à Horizon 2030",
    "Étude approfondie et évaluation du marché des chaînes de val":
        "Étude approfondie et évaluation du marché des chaînes de valeur de la noix de cajou",
    "Services pour la mise en œuvre du plan d'action pour l'appui":
        "Services pour la mise en œuvre du plan d'action pour l'appui technique aux ministères",
    "Scénarios expansion – industriel acier (MENA)":
        "Evaluation de 4 scenarios d'expansion et élaboration d'une planification opérationnelle pour un industriel de l'acier au moyen orient",
    "Études faisabilité mobilité électrique":
        "Études faisabilité mobilité électrique",
}

# ───────────── Secteurs et sous-secteurs ─────────────
SECTEURS_HIERARCHY = {
    "Développement socio-économique": [],
    "Enseignement, Emploi & Entrepreneuriat": ["Enseignement", "Emploi", "Entrepreneuriat"],
    "Services financiers": [],
    "Agriculture & Agribusiness": [],
    "TIC & innovation": [],
    "Tourisme & Culture": [],
    "Services & Industries manufacturières": [],
    "Santé et Pharmaceutique": [],
    "Energy & Mining": [],
    "Logistique & Mobilité": []
}

# ───────────── Utilitaires texte ─────────────
def clean_text(text):
    return " ".join(text.split())

def detect_secteur(text):
    t = text.strip().lower()
    for secteur, sous_liste in SECTEURS_HIERARCHY.items():
        if secteur.lower() in t:
            return secteur, None
        for sous in sous_liste:
            if sous.lower() == t:
                return secteur, sous
    return None, None

def iter_block_items(doc):
    body = doc.element.body
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            yield ("paragraph", Paragraph(child, doc))
        elif tag == "tbl":
            yield ("table", Table(child, doc))

# ───────────── Matching Jaccard ─────────────
STOPWORDS = {
    'de', 'du', 'des', 'le', 'la', 'les', 'un', 'une', 'et', 'en',
    'à', 'au', 'aux', 'pour', 'par', 'sur', 'dans', 'avec', 'ou',
    'est', 'que', 'qui', 'se', 'sa', 'son', 'ses', 'nous', 'vous',
    'd', 'l', 'j', 'n', 'y', 'the', 'of', 'in', 'for', 'a'
}

def normalize(s: str) -> set:
    s = s.lower()
    s = re.sub(r"[«»\"\"'''\-–—/&]", ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    words = set(s.split())
    return words - STOPWORDS

def jaccard_similarity(s1: str, s2: str) -> float:
    w1 = normalize(s1)
    w2 = normalize(s2)
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

def combined_score(projet: dict, fiche: dict) -> float:
    def to_str(val):
        return val if isinstance(val, str) else ""
    score_titre  = jaccard_similarity(to_str(projet.get("nom_projet")), to_str(fiche.get("nom_projet_fiche")))
    score_pays   = jaccard_similarity(to_str(projet.get("pays")),       to_str(fiche.get("pays_fiche")))
    score_client = jaccard_similarity(to_str(projet.get("client")),     to_str(fiche.get("client_fiche")))
    return score_titre * 0.6 + score_pays * 0.2 + score_client * 0.2

# ───────────── Détection type de tableau ─────────────
def is_recap_table(table) -> bool:
    if len(table.columns) != 5:
        return False
    rows = list(table.rows)
    if not rows:
        return False
    header = [c.text.strip().lower() for c in rows[0].cells]
    return any("réf" in h or "ref" in h or "titre" in h for h in header)

def is_fiche_table(table) -> bool:
    return len(table.columns) == 2 and len(list(table.rows)) >= 5

# ───────────── Parser tableau récapitulatif ─────────────
def extract_table_projects(table, current_secteur, current_sous_secteur):
    projets = []
    rows = list(table.rows)
    if len(rows) < 2:
        return projets

    header = [clean_text(c.text).lower() for c in rows[0].cells]
    col = {}
    for i, h in enumerate(header):
        if "réf" in h or "ref" in h:
            col["ref"] = i
        elif "titre" in h or "nom" in h:
            col["nom_projet"] = i
        elif "client" in h:
            col["client"] = i
        elif "valeur" in h or "montant" in h:
            col["valeur"] = i
        elif "pays" in h:
            col["pays"] = i

    if not col:
        col = {"ref": 0, "nom_projet": 1, "client": 2, "valeur": 3, "pays": 4}

    for row in rows[1:]:
        cells = [clean_text(c.text) for c in row.cells]
        if not any(cells):
            continue

        def get(field):
            idx = col.get(field)
            return cells[idx] if idx is not None and idx < len(cells) else ""

        nom = get("nom_projet")
        ref = get("ref")
        if not nom and not ref:
            continue

        projets.append({
            "secteur"      : current_secteur,
            "sous_secteur" : current_sous_secteur,
            "ref"          : ref,
            "nom_projet"   : nom,
            "client"       : get("client"),
            "valeur"       : get("valeur"),
            "pays"         : get("pays"),
            "full_text"    : "",
            # Champs remplis par le LLM
            "annee"        : "",
            "budget"       : "",
            "description"  : "",
            "services"     : [],
            "impacts"      : [],
            "chiffres_cles": [],
            "mots_cles"    : [],
            # Rétrocompatibilité
            "missions"     : [],
            "livrables"    : [],
            "expertises"   : [],
        })

    return projets
def table_to_markdown(table) -> str:
    """
    Transforme un tableau docx en format Markdown.
    C'est crucial pour ne pas perdre la structure des données 'implicites'.
    """
    md_content = ""
    for i, row in enumerate(table.rows):
        # On nettoie chaque cellule
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        # Création de la ligne Markdown
        md_line = "| " + " | ".join(cells) + " |\n"
        md_content += md_line
        
        # Ajout de la ligne de séparation Markdown après l'en-tête (la 1ère ligne)
        if i == 0:
            separator = "| " + " | ".join(["---"] * len(cells)) + " |\n"
            md_content += separator
    return md_content
# ───────────── Parser fiche (extraction brute) ─────────────
def parse_fiche_raw(table) -> dict:
    """
    Extraction brute de la fiche utilisant le format Markdown 
    pour conserver TOUTE l'information.
    """
    # 1. On transforme le tableau complet en Markdown
    full_markdown = table_to_markdown(table)
    
    # 2. On garde votre logique pour extraire le Nom, Pays et Client 
    # (nécessaire pour le matching Jaccard plus tard)
    rows = list(table.rows)
    nom_raw = rows[0].cells[0].text.strip()
    # Nettoyer le label si présent
    if "nom du projet" in nom_raw.lower() and ":" in nom_raw:
        nom = nom_raw.split(":", 1)[1].strip()
    else:
        nom = nom_raw
    nom = " ".join(nom.split())

    # Extraction rapide pour les métadonnées (matching)
    temp_text = full_markdown.lower()
    
    def find_val_from_cells(rows, keyword):
        for row in rows:
            for i, cell in enumerate(row.cells):
                if keyword.lower() in cell.text.lower():
                    # La valeur est soit dans la même cellule après ":", soit dans la cellule suivante
                    text = cell.text
                    if ":" in text:
                        val = text.split(":", 1)[1].strip()
                        if val:
                            return val.split("\n")[0]
                    # Chercher dans la cellule d'à côté
                    if i + 1 < len(row.cells):
                        return row.cells[i+1].text.strip().split("\n")[0]
        return ""

    pays   = find_val_from_cells(rows, "pays")
    client = find_val_from_cells(rows, "client")

    return {
        "nom_projet_fiche": nom,
        "pays_fiche": pays,
        "client_fiche": client,
        "full_text": full_markdown, # <-- C'est ici que le Markdown est stocké
        "raw_pairs": full_markdown.split("\n")
    }

# ───────────── LLM : Normalisation intelligente d'une fiche ─────────────
LLM_SYSTEM = """/no_think
Tu es un assistant spécialisé dans l'extraction structurée de données pour un cabinet de consulting (Matine Consulting).
Tu vas recevoir une fiche projet extraite d'un document Word convertie au format Markdown.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TON RÔLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Analyser le tableau Markdown pour extraire les informations clés.
2. Si une information est répartie sur plusieurs lignes ou cellules du tableau, fusionne-la intelligemment.
3. Ne perds aucun détail important, même si l'information est nichée dans une description longue.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAMPS À EXTRAIRE (en JSON strict) :
- nom_projet : Le titre complet.
- pays       : Pays d'exécution.
- annee : Année de début.
- budget : Montant exact avec devise.
- client : Entité cliente complète.
- description : Un résumé structuré de 3 à 5 phrases capturant l'essence du projet.
- services : Liste bullet des principaux services rendus et actions menées.
- impacts : Liste des résultats concrets et succès.
- chiffres_cles : Tous les chiffres quantifiés présents dans la fiche (participants, mesures, emplois, budgets partiels, etc.).
- mots_cles     : Termes sectoriels, acronymes, noms de programmes ou méthodologies clés.
RÈGLE D'OR : 
Le contenu reçu est un tableau. Les colonnes de gauche sont généralement les labels (ex: 'Description du projet') et les colonnes de droite le contenu. Lis bien horizontalement.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÈGLES STRICTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
R1. Réponds UNIQUEMENT avec un JSON valide. Aucun texte avant ou après. Aucune balise markdown.
R2. Si un champ est absent ou introuvable → "" pour les strings, [] pour les listes.
R3. Pour "annee" : cherche "Date de démarrage", "Date de début", ou toute mention d'année (ex: "mars 2022" → "2022", "2019-2021" → "2019").
R4. Pour "budget" : prends la "Valeur approximative des services" ou toute mention de montant. Garde la devise originale. Ne convertis pas.
R5. Pour "description" : résume en 3-5 phrases le contexte + objectif principal. Ne copie pas mot pour mot.
R6. Pour "services"  : extrais TOUTES les activités clés réalisées, sans limite de nombre. Format : liste de strings. 
R7. Pour "impacts"   : extrais les résultats concrets obtenus. Si la section est vide dans la fiche → [].
R8. Pour "pays" : si plusieurs pays mentionnés comme lieux d'exécution, liste-les séparés par des virgules.
R9. Langue de sortie : toujours en français.
R10. Pour "chiffres_cles" : extrais TOUS les chiffres quantifiés de la fiche, qu'ils soient dans la description,
     les services ou les impacts. Chaque entrée doit être une phrase courte avec le chiffre et son contexte.
     Exemples : "150 participants aux ateliers", "229 mesures approuvées en conseil des ministres",
     "43 mesures d'urgence validées", "20 départements analysés", "3 800 étudiants bénéficiaires".
R11. Pour "mots_cles" : extrais les termes techniques, acronymes, noms de programmes et méthodologies.
     Exemples : "DEPTH", "PMO SMART", "SCAPP", "TIA", "Delivery Unit", "EU4Youth", "Maghroum'IN",
     "iDICE", "Compact with Africa", "STEM", "DigComp", "ESCO", "dialogue public-privé".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT JSON ATTENDU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "nom_projet"  : "Titre complet du projet",
  "pays"        : "Tunisie", 
  "annee"       : "2022",
  "budget"      : "85 006,20 USD",
  "client"      : "Nom du client",
  "description" : "Résumé du contexte et des objectifs en 3-5 phrases.",
  "services"    : [
    "Service clé 1 réalisé",
    "Service clé 2 réalisé"
  ],
  "impacts"     : [
    "Impact ou résultat concret 1",
    "Impact ou résultat concret 2"
  ],
  "chiffres_cles": [
    "150 participants aux ateliers sectoriels",
    "229 mesures approuvées en conseil des ministres"
  ],
  "mots_cles"    : [
    "dialogue public-privé",
    "PMO SMART",
    "Delivery Unit"
  ]
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXEMPLES DE SORTIE ATTENDUE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── EXEMPLE 1 ──
Fiche brute (extrait) :
  Nom du projet : Country Economic Transformation Outlook (CETO) : Tunisie
  Pays : Tunisia | Date de démarrage : 01/2025 | Valeur : 85 006,20 USD
  Client : African Center for Economic Transformation (ACET)
  Description : Le CETO est une initiative de l'ACET basée sur le cadre DEPTH (Diversification, Export competitiveness, Productivity, Technological upgrading, Human well-being), enrichi par la durabilité environnementale et l'inclusion du genre...
  Services : Diagnostic de la transformation économique sur 20 ans, analyse de 3 leviers stratégiques (RE, MPME, numérique), démarche participative, feuille de route...
  Impacts : Diagnostic DEPTH consolidé, recommandations opérationnelles, analyses thématiques approfondies, appropriation renforcée...

Sortie JSON attendue :
{
  "nom_projet"  : "Country Economic Transformation Outlook (CETO) : Tunisie",
  "pays"        : "Tunisie",
  "annee"       : "2025",
  "budget"      : "85 006,20 USD",
  "client"      : "African Center for Economic Transformation (ACET)",
  "description" : "Le CETO, initiative de l'ACET, analyse la transformation économique à travers le concept de Growth with DEPTH, articulé autour de cinq piliers : diversification, compétitivité des exportations, productivité, modernisation technologique et bien-être humain. Ce cadre est renforcé par la durabilité environnementale et l'inclusion du genre comme leviers transversaux.",
  "services"    : [
    "Diagnostic de la transformation économique sur 20 ans selon le cadre DEPTH, avec identification des obstacles et opportunités.",
    "Analyse approfondie de trois leviers stratégiques : énergies renouvelables, financement des MPME et économie numérique.",
    "Démarche participative impliquant acteurs publics, privés et société civile pour co-construire les recommandations.",
    "Élaboration d'une feuille de route combinant actions à court et long terme, adaptée aux spécificités régionales."
  ],
  "impacts"     : [
    "Diagnostic DEPTH consolidé : identification claire des obstacles et opportunités de transformation.",
    "Recommandations opérationnelles alignées sur les priorités nationales.",
    "Analyses thématiques approfondies sur les leviers stratégiques (RE, MPME, numérique).",
    "Appropriation renforcée grâce à l'implication continue des parties prenantes."
  ],
    "chiffres_cles": [
    "20 ans de trajectoire économique analysés",
    "3 leviers stratégiques approfondis : RE, MPME, numérique",
    "Plus de 150 parties prenantes impliquées dans la démarche participative"
  ],
  "mots_cles"    : [
    "CETO",
    "DEPTH",
    "Growth with DEPTH",
    "ACET",
    "transformation économique",
    "énergies renouvelables",
    "MPME",
    "économie numérique",
    "durabilité environnementale",
    "inclusion du genre"
  ]
}

── EXEMPLE 2 ──
Fiche brute (extrait) :
  Nom du projet : Dialogue Public-Privé – Assises de l'Innovation
  Pays : Tunisie | Date de démarrage : Octobre 2022 | Date d'achèvement : Février 2023 | Durée : 5 mois
  Valeur : 49 500 USD | Client : IFC / Ministère de l'Économie et de la Planification
  Description : Appui au Gouvernement tunisien pour la réalisation des Assises de l'Innovation...
  Résultats : Feuille de route validée en conseil des ministres, 100+ projets innovants, 126 mesures sectorielles,
              25 mesures transverses, ~20 mesures pour faire évoluer le Start-up Act, 11 secteurs couverts,
              150+ participants (ministres, DG, startups, associations)

Sortie JSON attendue :
{
  "nom_projet"   : "Dialogue Public-Privé pour une feuille de route de l'Innovation (Les Assises de l'Innovation)",
  "pays"         : "Tunisie",
  "annee"        : "2022",
  "budget"       : "49 500 USD",
  "client"       : "International Financial Corporation (IFC) / Ministère de l'Économie et de la Planification",
  "description"  : "Matine Consulting a appuyé le Gouvernement tunisien dans la réalisation des Assises de l'Innovation, sessions de dialogue entre acteurs publics et privés couvrant 11 secteurs. L'objectif était d'identifier les opportunités, priorités et barrières à l'innovation en Tunisie et de produire une feuille de route nationale validée en conseil des ministres.",
  "services"     : [
    "Conception de la méthodologie et identification des secteurs et thématiques clés.",
    "Identification et mobilisation des acteurs publics et privés (ministres, DG, startups, associations).",
    "Modération et animation des ateliers sectoriels et thématiques.",
    "Identification des opportunités d'innovation et des prérequis par secteur.",
    "Consolidation des contributions et restitution finale au gouvernement."
  ],
  "impacts"      : [
    "Feuille de route pour l'innovation validée en conseil des ministres et publiée par la présidence.",
    "Plus de 100 projets innovants identifiés.",
    "126 mesures sectorielles et 25 mesures transverses définies.",
    "Environ 20 mesures proposées pour faire évoluer le Start-up Act tunisien.",
    "Réformes réglementaires approuvées et pilotées par la Delivery Unit."
  ],
  "chiffres_cles": [
    "Plus de 150 participants (ministres, directeurs généraux, startups, experts internationaux)",
    "11 secteurs d'activités couverts",
    "126 mesures sectorielles produites",
    "25 mesures transverses pour améliorer le climat de l'innovation",
    "Plus de 100 projets innovants identifiés",
    "Environ 20 mesures pour faire évoluer le Start-up Act"
  ],
  "mots_cles"    : [
    "dialogue public-privé",
    "Assises de l'Innovation",
    "feuille de route innovation",
    "Start-up Act",
    "Delivery Unit",
    "IFC",
    "réformes réglementaires",
    "entrepreneuriat",
    "climat de l'innovation"
  ]
}

── EXEMPLE 3 ──
Fiche brute (extrait) :
  Nom du projet : Identification, Formulation, priorisation et suivi des réformes économiques d'urgence - Delivery unit
  Pays : Tunisie | Date de démarrage : 07/2022 | Date d'achèvement : 2022 | Durée : 6 mois
  Valeur : Confidentiel | Client : Ministère de l'Économie et de la Planification
  Description : 43 mesures d'urgence validées dans le cadre du plan national de relance.
  Services : Mise en place d'un PMO SMART, entretiens parties prenantes, ateliers, suivi.
  Impacts : (section vide)

Sortie JSON attendue :
{
  "nom_projet"   : "Identification, Formulation, priorisation et suivi de la mise en œuvre de réformes économiques d'urgence en Tunisie en mode dialogue public-public - Delivery unit",
  "pays"         : "Tunisie",
  "annee"        : "2022",
  "budget"       : "Confidentiel",
  "client"       : "Ministère de l'Économie et de la Planification",
  "description"  : "Dans le cadre du plan national de relance économique à court terme, 43 mesures d'urgence ont été validées. Cette mission visait à coordonner les acteurs concernés et assurer le suivi de leur mise en œuvre via un PMO SMART.",
  "services"     : [
    "Mise en place d'un PMO SMART pour une gestion efficace du projet.",
    "Conduite d'entretiens avec les parties prenantes concernées.",
    "Organisation d'ateliers de travail interactifs.",
    "Suivi rigoureux de la mise en œuvre des mesures d'urgence."
  ],
  "impacts"      : [],
  "chiffres_cles": [
    "43 mesures d'urgence validées dans le cadre du plan national de relance"
  ],
  "mots_cles"    : [
    "PMO SMART",
    "Delivery Unit",
    "réformes économiques d'urgence",
    "dialogue public-public",
    "plan national de relance"
  ]
}
"""

def call_ollama_llm(fiche_text: str) -> dict | None:
    # Tronquer à 600 mots pour ne pas saturer le contexte
    words = fiche_text.split()
    if len(words) > 600:
        fiche_text = " ".join(words[:600]) + "\n[... texte tronqué ...]"

    payload = {
        "model"  : OLLAMA_MODEL,
        "system" : LLM_SYSTEM,       # ← system séparé du prompt utilisateur
        "prompt" : f"VOICI LE TABLEAU MARKDOWN À ANALYSER :\n\n{fiche_text}",
        "stream" : False,
        "options": {
            "temperature" : 0.1,
            "num_predict" : 1200,    # ← limite la réponse JSON
            "num_ctx"     : 8192,    # ← contexte explicite
        }
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        if not raw:
            print(f"   ⚠️  LLM a retourné une réponse vide")
            return None

        if len(raw) < 20:
            print(f"   ⚠️  Réponse trop courte : {repr(raw)}")
            return None

        clean = re.sub(r"```json|```", "", raw).strip()
        start = clean.find("{")
        end   = clean.rfind("}") + 1

        if start == -1 or end <= start:
            print(f"   ⚠️  Aucun JSON dans la réponse : {repr(raw[:200])}")
            return None

        clean = clean[start:end]
        return json.loads(clean)

    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout — fiche trop longue ou modèle trop lent")
        return None
    except requests.exceptions.ConnectionError:
        print("   ❌ Ollama non disponible")
        return None
    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON invalide ({e})")
        print(f"        Réponse brute : {repr(raw[:300])}")
        return None
    except Exception as e:
        print(f"   ⚠️  Erreur LLM ({e})")
        return None

def extract_annee_fallback(text: str) -> str:
    """Extraction d'année par regex si LLM indisponible."""
    # Cherche une année entre 1990 et 2030
    matches = re.findall(r'\b(20[0-2]\d|199\d)\b', text)
    return matches[0] if matches else ""

def normalize_fiche_with_llm(raw_fiche: dict, verbose: bool = True) -> dict:
    fiche_text = raw_fiche["full_text"]
    nom        = raw_fiche["nom_projet_fiche"]

    if verbose:
        print(f"   🧠 LLM normalise : {nom[:55]}...")

    llm_result = call_ollama_llm(fiche_text)
    def safe_str(val, fallback=""):
        """Garantit une string même si le LLM retourne un dict ou une liste."""
        if isinstance(val, str):
            return val.strip()
        if isinstance(val, list):
            return " ".join(str(v) for v in val).strip()
        if isinstance(val, dict):
            # Essaie d'extraire une valeur textuelle du dict
            for key in ("nom", "name", "value", "text", "label"):
                if key in val and isinstance(val[key], str):
                    return val[key].strip()
            return " ".join(str(v) for v in val.values()).strip()
        return fallback

    def safe_list(val):
        """Garantit une liste de strings même si le LLM retourne autre chose."""
        if isinstance(val, list):
            return [str(v) for v in val if v]
        if isinstance(val, str) and val:
            return [val]
        return []
# ✅ CORRECT — appeler safe_str et safe_list dans le return
    if llm_result:
        return {
            "nom_projet_fiche"   : safe_str(llm_result.get("nom_projet"),    nom) or nom,
            "pays_fiche"         : safe_str(llm_result.get("pays"),          raw_fiche["pays_fiche"]) or raw_fiche["pays_fiche"],
            "client_fiche"       : safe_str(llm_result.get("client"),        raw_fiche["client_fiche"]) or raw_fiche["client_fiche"],
            "annee_fiche"        : safe_str(llm_result.get("annee"),         "") or extract_annee_fallback(fiche_text),
            "budget_fiche"       : safe_str(llm_result.get("budget"),        ""),
            "description_fiche"  : safe_str(llm_result.get("description"),   ""),
            "services_fiche"     : safe_list(llm_result.get("services")),
            "impacts_fiche"      : safe_list(llm_result.get("impacts")),
            "chiffres_cles_fiche": safe_list(llm_result.get("chiffres_cles")),
            "mots_cles_fiche"    : safe_list(llm_result.get("mots_cles")),
            "full_text"          : fiche_text,
        }
    else:
        return {
            "nom_projet_fiche"  : nom,
            "pays_fiche"        : raw_fiche["pays_fiche"],
            "client_fiche"      : raw_fiche["client_fiche"],
            "annee_fiche"       : extract_annee_fallback(fiche_text),
            "budget_fiche"      : "",
            "description_fiche" : "",
            "services_fiche"    : [],
            "impacts_fiche"     : [],
            "chiffres_cles_fiche": [],
            "mots_cles_fiche"    : [],
            "full_text"         : fiche_text,
        }

# ───────────── Application fiche → projet ─────────────
def apply_fiche_to_projet(projet: dict, fiche: dict):
    """
    Enrichit le projet avec toutes les données extraites de la fiche.
    Priorité aux données de la fiche si le projet a des champs vides.
    """
    projet["full_text"] = fiche["full_text"]
    if fiche.get("nom_projet_fiche") and len(fiche["nom_projet_fiche"]) > len(projet.get("nom_projet", "")):
        projet["nom_projet"] = fiche["nom_projet_fiche"]

    # Champs simples : fiche prioritaire si projet vide
    for field_p, field_f in [
        ("pays",    "pays_fiche"),
        ("client",  "client_fiche"),
        ("annee",   "annee_fiche"),
        ("budget",  "budget_fiche"),
    ]:
        if not projet.get(field_p) and fiche.get(field_f):
            projet[field_p] = fiche[field_f]

    if not projet.get("valeur") and fiche.get("budget_fiche"):
        projet["valeur"] = fiche["budget_fiche"]

    # Champs enrichis (depuis la fiche)
    projet["description"] = fiche.get("description_fiche", "")
    projet["services"]    = fiche.get("services_fiche",    [])
    projet["impacts"]     = fiche.get("impacts_fiche",     [])
    projet["chiffres_cles"] = fiche.get("chiffres_cles_fiche",  [])  
    projet["mots_cles"]     = fiche.get("mots_cles_fiche",      [])
    
    # Rétrocompatibilité (anciennes clés conservées si besoin downstream)
    projet["missions"]    = fiche.get("services_fiche",    [])
    projet["livrables"]   = []
    projet["expertises"]  = []


# ───────────── Parcours séquentiel du document ─────────────
current_secteur      = None
current_sous_secteur = None
all_projects         = []
tableau_count        = 0
fiche_count          = 0
fiche_matched        = 0
fiche_unmatched      = 0

print("📖 Parcours du document Word...\n")

for block_type, block in iter_block_items(doc):

    if block_type == "paragraph":
        text = block.text.strip()
        if not text:
            continue
        secteur, sous = detect_secteur(text)
        if secteur:
            current_secteur      = secteur
            current_sous_secteur = sous
            print(f"\n✅ Secteur : {secteur}" + (f" > {sous}" if sous else ""))

    elif block_type == "table":
        if current_secteur is None:
            continue

        # ── Tableau récapitulatif ──
        if is_recap_table(block):
            projets = extract_table_projects(block, current_secteur, current_sous_secteur)
            if projets:
                all_projects.extend(projets)
                tableau_count += 1
                print(f"   📋 Récap #{tableau_count} → {len(projets)} projets ({current_secteur})")

        # ── Fiche détaillée ──
        elif is_fiche_table(block):
            fiche_count += 1

            # 1. Extraction brute
            raw_fiche = parse_fiche_raw(block)
            nom_fiche = raw_fiche["nom_projet_fiche"]

            if not nom_fiche or not raw_fiche["full_text"]:
                print(f"   ⚠️  Fiche #{fiche_count} vide — ignorée")
                continue

            # 2. Normalisation intelligente avec LLM
            fiche = normalize_fiche_with_llm(raw_fiche, verbose=verbose_matching)

            best_projet = None

            # ── Priorité 1 : mapping manuel ──
            for nom_recap, debut_fiche in MANUAL_MAPPING.items():
                if jaccard_similarity(debut_fiche, nom_fiche) >= 0.3:
                    for projet in all_projects:
                        if not projet.get("full_text"):
                            if jaccard_similarity(nom_recap, projet.get("nom_projet", "")) >= 0.3:
                                best_projet = projet
                                if verbose_matching:
                                    print(f"   🗺️  Manuel : {nom_fiche[:50]}")
                                    print(f"           → {projet['nom_projet'][:50]}")
                                break
                    if best_projet:
                        break

            # ── Priorité 2 : matching multi-critères ──
            if not best_projet:
                best_score = 0.0
                for projet in all_projects:
                    if projet.get("full_text"):
                        continue
                    # Score enrichi avec l'année LLM
                    score = combined_score(projet, fiche)
                    if score > best_score:
                        best_score  = score
                        best_projet = projet if score >= MATCH_THRESHOLD else None

            # ── Application ──
            if best_projet:
                apply_fiche_to_projet(best_projet, fiche)
                fiche_matched += 1
                if verbose_matching:
                    annee_info = f" [{fiche.get('annee_fiche', '?')}]"
                    print(f"   ✅ Match{annee_info} : {nom_fiche[:50]}")
                    print(f"          → {best_projet['nom_projet'][:50]}")
            else:
                fiche_unmatched += 1
                print(f"   ❌ Sans match : {nom_fiche[:60]}")

# ───────────── Résultats ─────────────
print(f"\n{'='*55}")
print(f"Total projets          : {len(all_projects)}")
print(f"Tableaux récap parsés  : {tableau_count}")
print(f"Fiches parsées         : {fiche_count}")
print(f"  ✅ Fiches matchées   : {fiche_matched}")
print(f"  ❌ Fiches sans match : {fiche_unmatched}")

avec_fiche = sum(1 for p in all_projects if p.get("full_text"))
sans_fiche = sum(1 for p in all_projects if not p.get("full_text"))
avec_annee = sum(1 for p in all_projects if p.get("annee"))
print(f"Projets avec full_text : {avec_fiche} / {len(all_projects)}")
print(f"Projets sans full_text : {sans_fiche} / {len(all_projects)}")
print(f"Projets avec année     : {avec_annee} / {len(all_projects)}")

if sans_fiche:
    print(f"\n=== Projets sans full_text ===")
    for p in all_projects:
        if not p.get("full_text"):
            print(f"  ❌ [{p.get('secteur','')}] {p.get('ref','')} | {p.get('nom_projet','')[:60]}")

print(f"\n=== Aperçu 3 premiers projets ===")
for p in all_projects[:3]:
    print(f"\n  [{p['secteur']}] {p.get('ref','')} | {p['nom_projet'][:50]}")
    print(f"  pays       : {p.get('pays','')}")
    print(f"  client     : {p.get('client','')}")
    print(f"  année      : {p.get('annee','')}")
    print(f"  budget     : {p.get('valeur','')}")
    print(f"  durée      : {p.get('duree','')}")
    print(f"  services   : {p.get('services',[])[:2]}")
    print(f"  chiffres_cles : {p.get('chiffres_cles',[])[:3]}")
    print(f"  mots_cles     : {p.get('mots_cles',[])[:5]}")
    print(f"  expertises : {p.get('expertises',[])[:3]}")
    ft = p.get('full_text', '')
    print(f"  full_text  : {ft[:150]}...")

# ───────────── Sauvegarde JSON ─────────────
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_projects, f, ensure_ascii=False, indent=2)

print(f"\n✅ JSON sauvegardé : {OUTPUT_PATH}")
print(f"   Champs disponibles : nom_projet, pays, client, annee, valeur, duree,")
print(f"                        description, services, livrables, expertises, full_text")
