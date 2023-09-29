from glob import glob
from os import chdir

from pdfminer.high_level import extract_text

pdf_dir = "C:/Users/peter/Repos/czech_readability_corpus/orig/"

for fname in glob(pdf_dir + "unlocked*.pdf"):
    print(fname)
    title = fname[60:-4]
    text = extract_text(fname)

    with open(f"pdf_texts/{title}.txt", "w", encoding="UTF8") as f:
        f.write(text)
