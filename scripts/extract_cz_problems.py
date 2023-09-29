# this is to extract the czech from the problematic files, in this list: 
    # a1-a1_chram-matky-bozi-v-parizi.txt 
    # French/Czech texts are intermingled and cause problems, need to make sure it's all there
    # a1-a2_pohadky-bratri-grimmu.txt - German and Czech texts mixed together
    # a1-a2_rimske-baje.txt - Italian and Czech mixed together
    # a1-a2_ruske-byliny.txt - Russian and Czech mixed together, then almost nothing
    # a1-a2_talianske-rozpravky.txt - no text there
    # a1-a2-aj-cj_maly-princ.txt - English and Czech mixed together
# this should be very similar to the last one, but for only these texts, and checking the language by line instead of page
import langdetect
from pathlib import Path

problem_files = ["chram-matky-bozi-v-parizi-a1-a2.txt", "maly-princ-a1-a2-aj-cj.txt", "pohadky-bratri-grimmu-a1-a2.txt",
                 "rimske-baje-a1-a2.txt", "ruske-byliny-a1-a2.txt", "talianske-rozpravky-a1-a2.txt"]


txt_dir = "pdf_texts/"
for fname in Path("pdf_texts").glob("*.txt"): 
    name = fname.name
    if name not in problem_files:
        continue
    # get information for the new label at end 
    # need the title, level, chapter, publisher
    level_index = 0
    end_index = -4

    if(name.find('a1') != -1):
        level_index = name.find('a1')
    elif(name.find('b1') != -1):
        level_index = name.find('b1')
    
    title_end_index = level_index - 1

    level = name[level_index:end_index]    
    title = name[:title_end_index]

    
    with open(fname, "r", encoding="UTF8") as curr_text:
        text = curr_text.readlines()
    # split the text on '\x0c' form feed markers for each page
    # check the language with detect for each page
    cz_text = "" # possibly use a string instead of array?
    for line in text:
        try:
            if langdetect.detect(line) == "cs":
                cz_text = cz_text + (line)
        except langdetect.lang_detect_exception.LangDetectException:
            continue
        # catch the exception when the page has no features to be detected
        # i.e. this should catch empty pages with '' or pages like '8282\n\n\n'

    with open(Path("test_files/" + level + "_" + title + ".txt"), "w", encoding="UTF8") as fout:
        fout.write(cz_text)
    # look at the Little Prince, a1-a2 - text is split on every other line eng-ces