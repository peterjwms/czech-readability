import re
import sys
from collections import Counter
from pathlib import Path
import numpy

import pandas as pd
import stanza
from matplotlib import pyplot as plt
from numpy import mean
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

stanza.download('cs')
nlp = stanza.Pipeline('cs')

doc = nlp("Já bych hrál hru.")

words = []
for sent in doc.sentences:
    for word in sent.words:
        # print(word)
        words.append(word)


def words_per_sentence(doc) -> float:
    '''Returns the average number of tokens per sentence in a Doc.

    Parameters
    ----------

    doc: Doc
        Spacy Doc containing tokenized and sentencized text.
    '''
    num_words = doc.num_words
    num_sents = len(doc.sentences)
    return num_words/num_sents


def words_per_file(doc) -> float:
    '''Returns the total number of tokens in a Doc.

    Parameters
    ----------

    doc: Doc
        Spacy Doc containing tokenized and sentencized text.
    '''
    return doc.num_words

def sents_per_file(doc):
    '''Returns number of sentences in a Doc.

    Parameters
    ----------

    doc: Doc
        Spacy Doc containing tokenized and sentencized text.
    '''
    return len(doc.sentences)

def mean_chars_per_word(words):
    return mean(([len(word.text) for word in words]))

def max_chars_per_word(words):
    return max([len(word.text) for word in words])


def total_punctuation(words) -> float:
    '''Returns the total number of punctuation marks in a Doc.

    Parameters
    ----------

    doc: Doc
        Spacy Doc containing tokenized and sentencized text.
    '''
    total_punct = [word for word in words if word.upos == "PUNCT"]
    return len(total_punct)

def punct_per_sentence(doc, words) -> float:
    '''Returns the average number of punctuation marks per sentence in a Doc.

    Parameters
    ----------

    doc: Doc
        Spacy Doc containing tokenized and sentencized text.
    '''
    total_punct = total_punctuation(words)
    return total_punct/len(doc.sentences)

def punct_token_ratio(words):
    return total_punctuation(words)/len(words)

def non_sent_end_punctuation(words) -> float:
    '''Returns the ratio of non-sentence ending punctuation marks out of total
    punctuation marks in a Doc.

    Parameters
    ----------

    doc: Doc
        Spacy Doc containing tokenized and sentencized text.
    '''
    sent_end_punct = {'.', '?', '!'}
    total_punct = total_punctuation(words)
    punct = [word for word in words if word.upos == "PUNCT"]
    if total_punct == 0:
        return "NA"
    odd_punct = [word for word in punct if word.text not in sent_end_punct]
    return len(odd_punct)/total_punct

def dialogue_punctuation_ratio(words):
    # regex to look for ""?
    dialogue_markers = ["„", "\"", "“"]
    dialogue_punct = [word for word in words if word.text in dialogue_markers]
    return len(dialogue_punct)/len(words)
# potentially normalize dialogue punct over sentences 
# could add sth w/ sents in quotation marks 


def type_token_ratio(words):
    return total_num_types(words)/len(words)
# TODO: corrected or root

# leave these out for the actual feature set
def total_num_types(words):
    types = {word.text for word in words}
    return len(types)

def total_num_lemmas(words):
    lemmas = [word.lemma for word in words]
    lemma_types = set(lemmas)
    return len(lemma_types)


# TODO: TTR, the other TTR measurements, vowel/consonant ratio


def pro_tr(words):
    """Compute pronoun-token ratio for input Text.

    tokens -- list of strings
    """
        
    # TODO: currently looking at total pronoun ratio
    regex = r'PronType=Prs'
    pro_count = len([word for word in words if (word.upos == "PRON" and re.search(regex, word.feats, re.I))])
    # print(pro_count)
    # look at pos tag ratio
    return pro_count / len(words)


def pro1_tr(words):
    """Compute 1st person singular pronoun-token ratio for input Text.

    tokens -- list of strings
    """

    pron_type = r'PronType=Prs'
    person = r'Person=1'
    number = r'Number=Sing'
    pro1_count = len([word for word in words if (word.upos == "PRON" and re.search(pron_type, word.feats, re.I) 
                                                 and re.search(person, word.feats) and re.search(number, word.feats))])
    # look at pos tag ratio
    return pro1_count / len(words)

# TODO: this one is marking ty as a plural nominative demonstrative pronoun
def pro2_tr(words):
    """Compute 2nd person singular pronoun-token ratio for input Text.

    tokens -- list of strings
    """

    pron_type = r'PronType=Prs'
    person = r'Person=2'
    number = r'Number=Sing'
    pro2_count = len([word for word in words if (word.upos == "PRON" and re.search(pron_type, word.feats, re.I) 
                                                 and re.search(person, word.feats) and re.search(number, word.feats))])
    return pro2_count / len(words)


def pro3_tr(words):
    """Compute 3rd person singular pronoun-token ratio for input Text.

    tokens -- list of strings
    """
    
    pron_type = r'PronType=Prs'
    person = r'Person=3'
    number = r'Number=Sing'
    pro3_count = len([word for word in words if (word.upos == "PRON" and re.search(pron_type, word.feats, re.I) 
                                                 and re.search(person, word.feats) and re.search(number, word.feats))])
    return pro3_count / len(words)


def pro1pl_tr(words):
    """Compute 1st person plural pronoun-token ratio for input Text.

    tokens -- list of strings
    """

    pron_type = r'PronType=Prs'
    person = r'Person=1'
    number = r'Number=Plur'
    pro1_count = len([word for word in words if (word.upos == "PRON" and re.search(pron_type, word.feats, re.I) 
                                                 and re.search(person, word.feats) and re.search(number, word.feats))])
    # look at pos tag ratio
    return pro1_count / len(words)


def pro2pl_tr(words):
    """Compute 2nd person plural pronoun-token ratio for input Text.

    tokens -- list of strings
    """

    pron_type = r'PronType=Prs'
    person = r'Person=2'
    number = r'Number=Plur'
    pro2_count = len([word for word in words if (word.upos == "PRON" and re.search(pron_type, word.feats, re.I) 
                                                 and re.search(person, word.feats) and re.search(number, word.feats))])
    return pro2_count / len(words)


def pro3pl_tr(words):
    """Compute 3rd person plural pronoun-token ratio for input Text.

    tokens -- list of strings
    """
    
    pron_type = r'PronType=Prs'
    person = r'Person=3'
    number = r'Number=Plur'
    pro3_count = len([word for word in words if (word.upos == "PRON" and re.search(pron_type, word.feats, re.I) 
                                                 and re.search(person, word.feats) and re.search(number, word.feats))])
    return pro3_count / len(words)


def verbs_per_sent(doc, words):
    '''Returns number of verbs per sentence

    Parameters
    ----------

    doc: Doc
        Spacy Doc containing tokenized and sentencized text.
    '''
    verbs = [word for word in words if word.upos == "VERB" or word.upos == "AUX"]
    return len(verbs)/len(doc.sentences)


def aux_verbs_per_sent(doc, words):
    verbs = [word for word in words if word.upos == "AUX"]
    return len(verbs)/len(doc.sentences)


def ind_verbs_per_sent(doc, words):
    # indicatives per sentence
    # TODO: also look at conditional
    ind_mood = r'Mood=Ind'
    ind_verbs = [word for word in words if word.upos == "VERB" and re.search(ind_mood, word.feats)]
    return len(ind_verbs)/len(doc.sentences)


def cond_verbs_per_sent(doc, words):
    cond_mood = r'Mood=Cnd'
    cond_verbs = [word for word in words if word.upos == "AUX" and re.search(cond_mood, word.feats)]
    return len(cond_verbs)/len(doc.sentences)


def verbs_tr(doc):
    '''Compute verb-token ratio for input Text.

    Parameters
    ----------

    doc: Doc
        Spacy Doc containing tokenized and sentencized text.
    '''
    verbs = [word for word in words if word.upos == "VERB" or word.upos == "AUX"]
    return len(verbs)/doc.num_words


def word_length(words):
    """Compute average word length for input Text.

    tokens -- list of strings
    """
    lengths = [len(word.text) for word in words if not word.upos == "PUNCT"]
    average = sum(lengths)/len(lengths)
    return average


def nouns_per_sent(doc, words):
    '''Returns average number of nouns per sentence

    Parameters
    ----------

    doc: Doc
        Spacy Doc containing tokenized and sentencized text.
    '''
    nouns = [word for word in words if word.upos == "NOUN"]
    return len(nouns)/len(doc.sentences)


def adjs_per_sent(doc, words):
    adjs = [word for word in words if word.upos == "ADJ"]
    return len(adjs)/len(doc.sentences)


def adps_per_sent(doc, words):
    adps = [word for word in words if word.upos == "ADP"]
    return len(adps)/len(doc.sentences)


# TODO: make the list once to be able to pass it back for each one
def mean_lemma_freq(words):
    global lemma_freq
    # TODO: make this happen once somewhere else
    doc_freqs = list()
    for word in words:
        try:
            doc_freqs.append(lemma_freq[word.lemma])
        except:
            continue
    return mean(doc_freqs)

def min_lemma_freq(words):
    global lemma_freq
    doc_freqs = list()
    for word in words:
        try:
            doc_freqs.append(lemma_freq[word.lemma])
        except:
            continue
    return min(doc_freqs)

def mean_lemma_rank(words):
    global lemma_rank
    doc_ranks = list()
    for word in words:
        try:
            doc_ranks.append(lemma_rank[word.lemma])
        except:
            continue
    return mean(doc_ranks)

def max_lemma_rank(words):
    global lemma_rank
    doc_ranks = list()
    for word in words:
        try:
            doc_ranks.append(lemma_rank[word.lemma])
        except:
            continue
    return max(doc_ranks)


def mean_word_freq(words):
    global word_freq
    # TODO: make this happen once somewhere else
    doc_freqs = list()
    for word in words:
        try:
            doc_freqs.append(word_freq[word.text])
        except:
            continue
    return mean(doc_freqs)

def min_word_freq(words):
    global word_freq
    doc_freqs = list()
    for word in words:
        try:
            doc_freqs.append(word_freq[word.text])
        except:
            continue
    return min(doc_freqs)

def mean_word_rank(words):
    global word_rank
    doc_ranks = list()
    for word in words:
        try:
            doc_ranks.append(word_rank[word.text])
        except:
            continue
    return mean(doc_ranks)

def max_word_rank(words):
    global word_rank
    doc_ranks = list()
    for word in words:
        try:
            doc_ranks.append(word_rank[word.text])
        except:
            continue
    return max(doc_ranks)


def create_freq_dist(file, type):
    columns = ["rank", f'{type}', "freq", "recalc_freq", "fict_freq", "branch_freq", "journal_freq", "freq_char"]
    # df.set_index('rank')

    df = pd.read_csv(file, sep='\t', names=columns)
    ranks = df['rank'].tolist()
    
    last_seen_num = 0
    real_ranks = list()
    for rank in ranks:
        if numpy.isnan(rank):
            rank = last_seen_num
        else:
            last_seen_num = rank
        real_ranks.append(rank)
    
    df['rank'] = real_ranks

    return df

lemma_freq_df = create_freq_dist(Path('freq_dists/syn2015_lemma_utf8.tsv'), type = "lemma")
word_freq_df = create_freq_dist(Path('freq_dists/syn2015_word_utf8.tsv'), type = "word")

# zip and dict the lists in each column
lemma_rank = dict(zip(lemma_freq_df["lemma"], lemma_freq_df['rank']))
lemma_freq = dict(zip(lemma_freq_df['lemma'], lemma_freq_df['freq']))
word_rank = dict(zip(word_freq_df['word'], word_freq_df['rank']))
word_freq = dict(zip(word_freq_df['word'], word_freq_df['freq']))

# TODO: feats to add:
# words_over_n_syllables, pos function, consonant/vowel ratio
# words_over_n_chars, 
# look up tools for morphological analysis (Morfessor)
# median word/lemma freqs, pos-tr 
# want some more syntax stuff - dependency
# also need to add syllable features - possibly just a regex to look for and count syllables
# add readability measures  
#TODO: meta-function that you pass in the part of speech


def mean_dep_length(words):
    dep_lengths = [(abs(word.id - word.head)) for word in words]
    return mean(dep_lengths)

def max_dep_length(words):
    dep_lengths = [(abs(word.id - word.head)) for word in words]
    return max(dep_lengths)

def min_dep_length(words):
    dep_lengths = [(abs(word.id - word.head)) for word in words]
    return min(dep_lengths)

def mean_dep_length_pos(words, pos):
    dep_lengths = [(abs(word.id - word.head)) for word in words if word.upos == pos]
    if len(dep_lengths) != 0:
        return mean(dep_lengths)
    else:
        return 0

def max_dep_length_pos(words, pos):
    dep_lengths = [(abs(word.id - word.head)) for word in words if word.upos == pos]
    if len(dep_lengths) != 0:
        return max(dep_lengths)
    else:
        return 0


# readability measures

def avg_syllables_per_token(words):
    '''
    Returns the mean number of syllables per token in a doc

    Code adapted from: https://github.com/vanickovak/ReadabilityFormula/blob/main/counter_czech_from_txt.py#L86-L100
    '''
    total_syllables = 0
    for word in words:
        word_syllables = count_syllables(word.text)
        total_syllables += word_syllables
    return total_syllables/len(words)

def count_syllables(word):
    V = "[aeiuoyáéíóúůýěäëïöü]"
    word = re.sub(r"(r|l|m)\1+", r"\1", word.lower())
    word = word.replace("neuro", "nero")
    if word in ["sedm", "osm", "vosm"]:
        return 2
    elif word == "hm":
        return 1
    return len(re.findall(r"(^eu|^au|ou|{V}|(?<!{V})[rl](?!{V}))".format(V=V), word))

def flesch_ease(words, doc):
    # this requires the mean number of syllables per token - can't do syllables currently
    score = 206.935 - (words_per_sentence(doc) * 1.672) - (62.183 * avg_syllables_per_token(words))
    return score

def flesch_kincaid_grade(words, doc):
    # this requires mean number of syllables per token
    level = (words_per_sentence(doc) * 0.52) + (9.133 * avg_syllables_per_token(words)) - 16.393
    return level

def auto_read_index(words, doc):
    level = (3.666 * words_per_sentence(doc)) + (0.631 * mean_chars_per_word(words)) - 19.491
    return level

def coleman_liau_index(words, doc):
    # this is supposed to use mean number of chars per 100 tokens, and mean number tokens per 100 sents
    level = (0.047 * mean_chars_per_word(words) * 100) - (0.286 * (1/words_per_sentence(doc)) * 100) - 12.9
    return level

# extract the level label and make that the last column in the row
feat_names = ['name', 
              'words/sent', 
            #   'words/file', 'sents/file', 
              'mean_chars/wd', 'max_chars/wd',
            #   'total_punct', 
              'punct/sent', 'punct_tr', 'non_sent_end_punct', 'dialog_punct_tr',
              'ttr', #'total_types', 'total_lemmas', 
              'pro_tr', '1sing_pro_tr', '3sing_pro_tr',
              '1pl_pro_tr', '2pl_pro_tr', '3pl_pro_tr',
              "Verbs/sent", 'aux/sent', 'ind_vbs/sent', 'cond_vbs/sent',
              "Verbs_tr",  "word_length",  "nouns/sent",
              'adjs/sent', 'adps/sent',
              'mean_lemma_freq', 'min_lemma_freq', 'mean_lemma_rank', 'min_lemma_rank',
              'mean_word_freq', 'min_word_freq', 'mean_word_rank', 'min_word_rank',
              'mean_dep_len', 'min_dep_len', 'max_dep_len', 
              'mean_dep_len_nouns', 'max_dep_len_nouns', 
              'mean_dep_len_adjs', 'max_dep_len_adjs',
              'mean_dep_len_advs', 'max_dep_len_advs',
              'mean_dep_len_vbs', 'max_dep_len_vbs',
              'mean_dep_len_intjs', 'max_dep_len_intjs', 
            #   'flesch_ease', 'flesch_kincaid', 'auto_read_index', 'coleman_liau',
              'group', 'label']


ds = pd.read_csv(Path('datasets/balanced_dataset.csv'))

ds = ds.sample(frac=1, random_state=42)

feats_list = []

def get_title(name: str) -> str:
    parts = name.split('_')
    id_len = len(parts[0])
    if id_len == 4:
        title = parts[2] + '_' + parts[3]
        return title
    elif id_len == 5:
        return 'CTDC'

book_groups = dict() # a dict with the book name (level and title) as key and group number as the array
book_group_num = 0
ctdc_group_counter = 31
first = True

def get_book_group(name: str) -> int:
    
    title = get_title(name)
    print(title)
    # this is if it's a book
    if title != 'CTDC':
        if title not in book_groups.keys():
            global book_group_num
            book_groups[title] = book_group_num
            book_group_num += 1
        print(book_groups[title])
        return book_groups[title]
    else:
        # if it's from CTDC - loop through a certain number of groups to 'randomly' assign each file to a group
        global ctdc_group_counter
        global first
        if first:
            first = False
        elif ctdc_group_counter == 40: # reset it once we have ten groups
            ctdc_group_counter = 31
        else: # increment normally to simulate randomness
            ctdc_group_counter += 1
        print(ctdc_group_counter)
        return ctdc_group_counter
    
    
j = 0

for i, row in tqdm(ds.iterrows()):
    # preprocess
    doc = nlp(row['text'])
    words = []
    for sent in doc.sentences:
        for word in sent.words:
            words.append(word)

    # call functions HERE
    feats_list.append([row['name'], 
            words_per_sentence(doc), 
            # words_per_file(doc), sents_per_file(doc),
            mean_chars_per_word(words), max_chars_per_word(words),
            # total_punctuation(words), 
            punct_per_sentence(doc, words), punct_token_ratio(words), non_sent_end_punctuation(words), dialogue_punctuation_ratio(words),
            type_token_ratio(words), # total_num_types(words), total_num_lemmas(words),
            pro_tr(words), pro1_tr(words), pro3_tr(words),
            pro1pl_tr(words), pro2pl_tr(words), pro3pl_tr(words),
            verbs_per_sent(doc, words), aux_verbs_per_sent(doc, words),
            ind_verbs_per_sent(doc, words), cond_verbs_per_sent(doc, words),
            verbs_tr(doc), word_length(words), nouns_per_sent(doc, words),
            adjs_per_sent(doc, words), adps_per_sent(doc, words),
            mean_lemma_freq(words), min_lemma_freq(words), mean_lemma_rank(words), max_lemma_rank(words),
            mean_word_freq(words), min_word_freq(words), mean_word_rank(words), max_word_rank(words),
            mean_dep_length(words), min_dep_length(words), max_dep_length(words),
            mean_dep_length_pos(words, 'NOUN'), max_dep_length_pos(words, 'NOUN'),
            mean_dep_length_pos(words, 'ADJ'), max_dep_length_pos(words, 'ADJ'),
            mean_dep_length_pos(words, 'ADV'), max_dep_length_pos(words, 'ADV'),
            mean_dep_length_pos(words, 'VERB'), max_dep_length_pos(words, 'VERB'),
            mean_dep_length_pos(words, 'INTJ'), max_dep_length_pos(words, 'INTJ'),
            # flesch_ease(words, doc), flesch_kincaid_grade(words, doc), auto_read_index(words, doc), coleman_liau_index(words, doc),
            get_book_group(row['name']), row['label']])
    
    # j += 1
    # if j == 16:
    #    break # test for just some files
    # make this a bool that I can set to true/false for testing, used to overwrite balanced_with_feats(_test).csv

        
feats_df = pd.DataFrame(feats_list, columns=feat_names)
feats_df.set_index('name')
feats_df.to_csv(Path('datasets/balanced_feats_rand_groups.csv'), encoding='UTF8', sep=',', index=False)
