from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

DOCX_PATH = r"C:\Users\Chayma MAJJEDI\Desktop\chatbot_web\new pipeline\Projets2.docx"
doc = Document(DOCX_PATH)

def iter_block_items(doc):
    body = doc.element.body
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            yield ("paragraph", Paragraph(child, doc))
        elif tag == "tbl":
            yield ("table", Table(child, doc))

# Affiche le contenu complet du tableau à l'index 8 (1ère fiche)
for i, (btype, block) in enumerate(iter_block_items(doc)):
    if i == 8:
        print(f"=== FICHE (bloc {i}) ===")
        for r, row in enumerate(block.rows):
            col0 = row.cells[0].text.strip()[:60]
            col1 = row.cells[1].text.strip()[:60]
            print(f"  ligne {r} | col0: '{col0}' | col1: '{col1}'")
        break