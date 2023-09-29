from glob import glob
import os
from pathlib import Path

work_dir = Path("czech_readability_corpus")

bad_files = ["a1-a2_chram-matky-bozi-v-parizi.txt", "a1-a2_pohadky-bratri-grimmu.txt", "a1-a2_rimske-baje.txt",
             "a1-a2_ruske-byliny.txt", "a1-a2_talianske-rozpravky.txt", "a1-a2-aj-cj_maly-princ.txt"]

# os.chdir("C:/Users/peter/Repos/czech_readability_corpus/cz_full/")m
id = 0
for fname in Path('cz_full').glob("*.txt"):
    # get the level and book title from file name
    if(fname.name in bad_files):
        # somehow the text for these two books got mixed up and both of them are the b2 text
        continue
    
    book = fname.name
    parts = book.split("_")
    level = parts[0]
    title = parts[1][:-4]
    

    with open(fname, "r", encoding="UTF8") as curr_text:
        text = curr_text.read()
        # split the text on '\x0c' form feed markers for each chapter

        # TODO: check for no-space breaks and soft hyphens and replace them
        text = text.replace('\xa0', ' ')
        text = text.replace('\xad', '')

        split_text = text.split(sep='\x0c')
        i = 0
        for chapter in split_text:
            id += 1
            i += 1
            chap_id = str(i).zfill(3) 
            chap_file_name = str(id).zfill(4) + "_Edika_" + level + "_" + title + "_" + chap_id
            with open(Path(f'chapters/{chap_file_name}.txt'), "w", encoding="UTF8") as fout:
                fout.write(chapter)
