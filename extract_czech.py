from glob import glob
import langdetect
from os import chdir

chdir("C:/Users/peter/Repos/czech_readability_corpus")
txt_dir = "pdf_texts/"
for fname in glob(txt_dir + "*.txt"): 
    chdir("C:/Users/peter/Repos/czech_readability_corpus")
    # get information for the new label at end 
    # need the title, level, chapter, publisher
    level_index = 0
    end_index = -4

    if(fname.find('a1') != -1):
        level_index = fname.find('a1')
    elif(fname.find('b1') != -1):
        level_index = fname.find('b1')
    
    title_end_index = level_index - 1

    level = fname[level_index:end_index]    
    title = fname[10:title_end_index]

    
    with open(fname, "r", encoding="UTF8") as curr_text:
        text = curr_text.read()
    # split the text on '\x0c' form feed markers for each page
    # check the language with detect for each page
    split_text = text.split(sep='\x0c')
    czech_pages = "" # possibly use a string instead of array?
    for page in split_text:
        try:
            if langdetect.detect(page) == "cs":
                czech_pages = czech_pages + (page)
        except langdetect.lang_detect_exception.LangDetectException:
            continue
            # catch the exception when the page has no features to be detected
            # i.e. this should catch empty pages with '' or pages like '8282\n\n\n'

    # it looks like there's too much variety in chapter headings across pages and the splitting on chapters will be manual
    # TODO: add the output file
    chdir("C:/Users/peter/Repos/czech_readability_corpus/cz_full")
    with open(level + "_" + title + ".txt", "w", encoding="UTF8") as fout:
        fout.write(czech_pages)
    # look at the Little Prince, a1-a2 - text is split on every other line eng-ces
