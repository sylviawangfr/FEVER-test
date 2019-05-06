import spacy

# nlp_eng = spacy.load("en_core_web_md")
# nlp_eng = spacy.load("en_core_web_lg")
nlp_eng = spacy.load("en_core_web_sm")


def split_claim_spacy(text):
    # doc_multi = nlp_multi(text)
    doc_noun = nlp_eng(text)
    nouns = [chunk.text for chunk in doc_noun.noun_chunks]
    ents = ([(ent.text, ent.label_) for ent in doc_noun.ents])
    # print(nouns)
    # print(ents)
    return nouns, ents


if __name__ == '__main__':
    n, e = split_claim_spacy("Hot Right Now is mistakenly attributed to DJ Fresh.")
    print(n)
    print(e)