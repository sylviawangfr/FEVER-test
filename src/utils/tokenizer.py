import spacy

# nlp_multi = spacy.load("xx_ent_wiki_sm")
nlp_eng = spacy.load("en_core_web_sm")

def split_claim(text):
    # doc_multi = nlp_multi(text)
    doc_noun = nlp_eng(text)
    nouns = [chunk.text for chunk in doc_noun.noun_chunks]
    ents = ([(ent.text, ent.label_) for ent in doc_noun.ents])
    # print(nouns)
    # print(ents)
    return nouns, ents


if __name__ == '__main__':
    n, e = split_claim("Colin Kaepernick became a starting quarterback during the 49ers 63rd season in the National Football League.")
    print(n)
    print(e)