import regex
import spacy
from spacy.symbols import nsubj, dobj, pobj, VERB
from memory_profiler import profile

# nlp_eng = spacy.load("en_core_web_md")
# nlp_eng = spacy.load("en_core_web_lg")
nlp_eng_spacy = spacy.load("en_core_web_sm")


def split_claim_spacy(text):
    # doc_multi = nlp_multi(text)
    doc_noun = nlp_eng_spacy(text)
    nouns_chunks = [chunk.text for chunk in doc_noun.noun_chunks]
    ents = ([(ent.text, ent.label_) for ent in doc_noun.ents])
    # print(nouns)
    # print(ents)
    return nouns_chunks, ents

# @profile
def count_words(sent):
    doc_w = nlp_eng_spacy(sent)
    l = len(doc_w)
    del doc_w
    return l


def split_claim_regex(text):
    # get capital phrases
    # REGEX = r'(?<![.])([A-Z]+[\w]*\s)*([A-Z][\w]+)'
    REGEX = r'([a-z0-9]*[A-Z]+[\w]*(\s)*(\'s)?' \
            r'(of\s)*(to\s)*(for\s)*(at\s)*(in\s)*(on\s)*(from\s)*(and\s)*(with\s)*(the\s)*' \
            r'(-?)(\d*\s)*)*(?<!-\s)([A-Z]+[\w]*(\s\d+)*)'
    regexp = regex.compile(REGEX)
    matches = [m for m in regexp.finditer(text)]
    tokens = [matches[i].group() for i in range(len(matches))]
    return tokens


def merge_phrases_as_span(sent, phrase_l):
    doc_to_merge = nlp_eng_spacy(sent)
    for ph in phrase_l:
        phrase = nlp_eng_spacy(ph)
        doc_tokens = [token.text for token in doc_to_merge]
        phrase_tokens = [token.text for token in phrase]
        try:
            phrase_idx = get_phrase_token_indice(doc_tokens, phrase_tokens)
            if len(phrase_idx) > 0:
                idx = phrase_idx[0]
                with doc_to_merge.retokenize() as retokenizer:
                    retokenizer.merge(doc_to_merge[idx[0]:idx[1]], attrs={"LEMMA": ph})
            else:
                continue
        except Exception as err:
            print(err)
    return doc_to_merge


def get_dependent_verb(sent, phrase_l):
    doc_merged = merge_phrases_as_span(sent, phrase_l)
    # displacy.serve(doc_merged, style='dep')
    # svg = displacy.render(doc_merged, style="dep")
    # output_path = config.LOG_PATH / 'sentence.svg'
    # output_path.open("w", encoding="utf-8").write(svg)
    phs = dict()
    for ph in phrase_l:
        for possible_phrase in doc_merged:
            if possible_phrase.text == ph:
                one_p = {'dep': '', 'verb': ''}
                if possible_phrase.dep == nsubj:
                    one_p['dep'] = 'subj'
                if possible_phrase.dep == dobj or possible_phrase.dep == pobj:
                    one_p['dep'] = 'obj'
                if possible_phrase.head.pos == VERB:
                    one_p['verb'] = possible_phrase.head.text
                else:
                    if possible_phrase.head.head.pos == VERB:
                        one_p['verb'] = possible_phrase.head.head.text
                    else:
                        try:
                            next_w = next(possible_phrase.head.rights)
                            if next_w.pos == VERB:
                                one_p['verb'] = next_w.text
                        except StopIteration:
                            pass
                phs[ph] = one_p
    return phs


def get_phrase_token_indice(sent_token_l, phrase_token_l):
    i = 0
    j = 0
    idx = []
    while i < len(sent_token_l):
        tmp_i = i
        while j < len(phrase_token_l) and i < len(sent_token_l):
            if not sent_token_l[i] == phrase_token_l[j]:
                i = i + 1
                j = 0
            else:
                i = i + 1
                j = j + 1
        if j == len(phrase_token_l):
            idx.append((i - len(phrase_token_l), i))
            j = 0
        else:
            i = tmp_i + 1
            j = 0
    return idx


if __name__ == '__main__':

    # d_l = ['two', 'three', 'four', 'six', 'one', 'two', 'thirty', 'one', 'two']
    # p_l = ['one', 'two']
    # get_phrase_token_indice(d_l, p_l)

    text = "Bessie Smith's Tale was married on April 15, 1894."
    text1 = "L.A. Reid has served as the president of a record label."
    ph = ['Bessie Smith', 'April 15, 1894']

    print(split_claim_regex(text))
    split_claim_spacy(text)
    # split_claim_nltk(text)
    # print(n)
    # print(e)
    doc = nlp_eng_spacy(text)
    for tok in doc:
        print(tok, tok.dep_)
