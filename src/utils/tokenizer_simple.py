import regex
import spacy
from spacy.symbols import nsubj, dobj, pobj, VERB, nsubjpass
import config
from memory_profiler import profile

# nlp_eng = spacy.load("en_core_web_md")
# nlp_eng = spacy.load("en_core_web_lg")
nlp_eng_spacy = spacy.load("en_core_web_md")


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


REGEX = r'([a-z0-9]*[A-Z]+[\w-]*(\')?(\s)*(\'s)?' \
            r'(of\s)*(to\s)*(for\s)*(at\s)*(in\s)*(on\s)*(from\s)*(by\s)*(and\s)*(with\s)*(the\s)*(a\s)*(an\s)*' \
            r'(-?)(&?)(\.?)(:?)(\d*\s)*)*(?<!-\s)(:?([A-Z0-9]+[\w]*(\s\d+[a-zA-Z]*(,?))*)|(of\s\d+))(\s\(.*\))*'

REGEX2 = r'([a-z0-9]*[A-Z]+[\w]*(\')?(\s)*' \
         r'(for\s)*(on\s)*(from\s)*(with\s)*(the\s)*(a\s)*(an\s)*' \
         r'(-?)(&?)(\.?)(:?)(\d*\s)*)*(?<!-\s)(:?([A-Z]+[\w]*(\s\d+[a-zA-Z]*)*))'


REGEX3 = r'([a-z0-9]*[A-Z]+[\w-]*(\')?(\s)*(\'s)?(of\s)*(to\s)*' \
         r'(for\s)*(at\s)*(in\s)*(on\s)*(from\s)*(by\s)*(with\s)*' \
         r'(the\s)*(a\s)*(an\s)*(-?)(&?)(\.?)(:?)(\d*\s)*)+(?<!-\s)(:?([A-Z0-9]+[\w]*(\s\d+[a-zA-Z]*)*))'


def split_claim_regex(text):
    # get capital phrases
    # REGEX = r'(?<![.])([A-Z]+[\w]*\s)*([A-Z][\w]+)'
    regexp = regex.compile(REGEX)
    matches = [m for m in regexp.finditer(text)]
    tokens = [matches[i].group() for i in range(len(matches))]
    to_delete = []
    tmp_tokens = []
    for t in tokens:
        if t.count(' ') > 7 and (t.count(',') > 0 or t.count(' and ') > 0):
            regexp = regex.compile(REGEX3)
            matches = [m for m in regexp.finditer(text)]
            new_tokens = [matches[i].group() for i in range(len(matches))]
            to_delete.append(t)
            tmp_tokens.extend(new_tokens)
    tokens.extend(tmp_tokens)
    for i in to_delete:
        tokens.remove(i)
    return list(set(tokens))


def split_combinations(text):
    regexp = regex.compile(REGEX2)
    matches = [m for m in regexp.finditer(text)]
    tokens = [matches[i].group() for i in range(len(matches))]
    return tokens


def is_capitalized(text):
    r = split_claim_regex(text)
    if len(r) > 0 and r[0] == text:
        return True
    else:
        return False


def is_person(phrase):
    t = nlp_eng_spacy(phrase)
    for tt in t.ents:
        if tt.label_ == 'PERSON':
            return True
    return False


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
    phrase_to_merge_in_dep_tree = [i for i in phrase_l if is_capitalized(i)]
    doc_merged = merge_phrases_as_span(sent, phrase_to_merge_in_dep_tree)
    # displacy.serve(doc_merged, style='dep')
    # svg = displacy.render(doc_merged, style="dep")
    # output_path = config.LOG_PATH / 'sentence.svg'
    # output_path.open("w", encoding="utf-8").write(svg)
    phs = dict()
    for possible_phrase in doc_merged:
        if possible_phrase.text in phrase_l:
            one_p = {'dep': '', 'verb': ''}
            if possible_phrase.dep in [nsubj, nsubjpass] :
                one_p['dep'] = 'subj'
            if possible_phrase.dep == dobj or possible_phrase.dep == pobj:
                one_p['dep'] = 'obj'
            if possible_phrase.head.pos == VERB or possible_phrase.head.dep_ == 'ROOT':
                one_p['verb'] = possible_phrase.head.text
            else:
                if possible_phrase.head.head.pos == VERB or possible_phrase.head.head.dep_ == 'ROOT':
                    one_p['verb'] = possible_phrase.head.head.text
                else:
                    try:
                        next_w = next(possible_phrase.head.rights)
                        if next_w.pos == VERB or next_w.dep_ == 'ROOT':
                            one_p['verb'] = next_w.text
                    except StopIteration:
                        pass
            phs[possible_phrase.text] = one_p
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


def get_lemma(text):
    tokens = nlp_eng_spacy(text)
    lemmas = [t.lemma_ for t in tokens]
    return lemmas


if __name__ == '__main__':
    # print(get_lemma('starring'))
    # ss1 = "Michelle Obama's husband was born in Kenya"
    # verbs = get_dependent_verb(ss1, ['Michelle Obama', 'husband', 'Kenya'])

    # d_l = ['two', 'three', 'four', 'six', 'one', 'two', 'thirty', 'one', 'two']
    # p_l = ['one', 'two']
    # get_phrase_token_indice(d_l, p_l)

    # text = "Bessie Smith's Tale was married on April 15, 1894."
    text = 'South African Communist Party is a partner of an alliance between the African National Congress (ANC), the Congress of South African Trade Unions (COSATU) and the South African Communist Party (SACP).'
    # ph = ['Bessie Smith', 'April 15, 1894']
    print(split_claim_regex(text))
    # split_claim_nltk(text)
    # print(n)
    # print(e)
    # doc = nlp_eng_spacy(text)
    # for tok in doc:
    #     print(tok, tok.dep_)
