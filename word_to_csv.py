from docx import Document
import pandas as pd
import re

DOCX_PATH = r"C:\Users\Chayma MAJJEDI\Desktop\chatbot_web\new pipeline\Projets2.docx"
OUTPUT_CSV = "projets_structures.csv"

doc = Document(DOCX_PATH)

# ──────────────────────────────────────────────
# NORMALISATION TEXTE
# ──────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ──────────────────────────────────────────────
# EXTRACTION FLEXIBLE DES CHAMPS
# ──────────────────────────────────────────────
def extract_field(text, keywords):
    for kw in keywords:
        pattern = rf"{kw}\s*:\s*([^\n]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return clean_text(match.group(1))
    return ""

# ──────────────────────────────────────────────
# EXTRACTION DESCRIPTION (bloc long)
# ──────────────────────────────────────────────
def extract_section(text, start_keywords):
    for kw in start_keywords:
        pattern = rf"{kw}\s*:\s*(.+?)(?:\n[A-Z][^\n]+:|\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return clean_text(match.group(1))
    return ""

# ──────────────────────────────────────────────
# PARSE UN TABLEAU = 1 PROJET
# ──────────────────────────────────────────────
def parse_project_table(table):

    # récupérer tout le texte
    all_text = "\n".join(
        cell.text.strip()
        for row in table.rows
        for cell in row.cells
        if cell.text.strip()
    )

    if len(all_text) < 50:
        return None

    # nom projet = première cellule
    try:
        nom = clean_text(table.rows[0].cells[0].text)
    except:
        nom = ""

    projet = {
        "nom_projet": nom,

        "pays": extract_field(all_text, [
            "Pays"
        ]),
        "lieu": extract_field(all_text, [
            "Lieu"
        ]),
        "client": extract_field(all_text, [
            "Nom du Client",
            "Client"

        ]),

        "periode": extract_field(all_text, [
            "Période",
            "Durée",
            "Nombre de mois de travail, durée du projet ",
            "Duration of assignment (months)"
            
        ]),

        "valeur": extract_field(all_text, [
            "Valeur approximative",
            "Valeur",
            "Budget",
            "Valeur approximative des services en USD ",
            "Environ. valeur du contrat (en dollars américains courants) ",
            "Valeur approximative des services en dollars américains ",
            "Environ. Valeur du contrat (en US$ courants) ",
            "Valeur approximative des services en TND ",
            "Valeur approximative des services (en US$)"
        ]),

        "description": extract_section(all_text, [
            "Description du projet"
        ]),

        "services": extract_section(all_text, [
            "Description des services effectivement rendus par notre personnel ",
            "Description des prestations réelles fournies par notre personnel ",
            "Description des prestations effectivement réalisées par l’équipe ",
            "Description des services réellement fournis par notre personnel",
            "Description des services fournis par notre personnel ",
            "Description des services réels fournis par notre personnel dans le cadre de la mission",
            "Description des services fournis par notre équipe",
            "Prestations"
        ]),
        "consultants": extract_section(all_text, [
            "Nom des Consultants associés /partenaires éventuels ",
            "Nom des Consultants associés, le cas échéant (experts)",
            "Nom des consultants associés, le cas échéant",
            "Name of associated consultants / possible partners ",
            "Nom des consultants associés",
            "Nom des consultants associés si applicable"
        ]),
        "impacts": extract_section(all_text, [
            "Principaux résultats & impacts",
            "Impacts Majeurs",
            "Principaux résultats et impacts",
            "Impacts et résultats ",
            "Impacts",
            

        ])
    }

    return projet

# ──────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ──────────────────────────────────────────────
projects = []

for i, table in enumerate(doc.tables):

    projet = parse_project_table(table)

    if projet and projet["nom_projet"]:
        projects.append(projet)
        print(f"✅ Projet extrait : {projet['nom_projet'][:60]}")

print(f"\n📊 Total projets extraits : {len(projects)}")

# ──────────────────────────────────────────────
# EXPORT CSV
# ──────────────────────────────────────────────
df = pd.DataFrame(projects)

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n✅ CSV généré : {OUTPUT_CSV}")