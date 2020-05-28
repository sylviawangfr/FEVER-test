import spacy
import regex

# nlp_eng = spacy.load("en_core_web_md")
# nlp_eng = spacy.load("en_core_web_lg")
nlp_eng_spacy = spacy.load("en_core_web_sm")


def split_claim_spacy(text):
    # doc_multi = nlp_multi(text)
    doc_noun = nlp_eng_spacy(text)
    nouns = [chunk.text for chunk in doc_noun.noun_chunks]
    ents = ([(ent.text, ent.label_) for ent in doc_noun.ents])
    # print(nouns)
    # print(ents)
    return nouns, ents


def split_claim_regex(text):
    # get capital phrases
    # REGEX = r'(?<![.])([A-Z]+[\w]*\s)*([A-Z][\w]+)'
    REGEX = r'([A-Z]+[\w]*[.]*(\s)*(of\s)*(to\s)*(for\s)*(a\s)*(in\s)*(on\s)*(from\s)*(and\s)*(with\s)*(the\s)*)*(?<!-\s)([A-Z]+[\w]*)'
    regexp = regex.compile(REGEX)
    matches = [m for m in regexp.finditer(text)]
    tokens = [matches[i].group() for i in range(len(matches))]
    return tokens


if __name__ == '__main__':
    text = "Tool has won three Oscars"
    print(split_claim_regex(text))
    split_claim_spacy(text)
    # split_claim_nltk(text)
    # print(n)
    # print(e)
    doc = nlp_eng_spacy(text)
    for tok in doc:
        print(tok, tok.dep_)
