print("start")
from tdrparser import parse

pages = parse(r"C:\Users\Chayma MAJJEDI\Downloads\TdR .pdf")
print(f"{len(pages)} pages extraites")
print(pages[0]["text"][:300])
print("done")