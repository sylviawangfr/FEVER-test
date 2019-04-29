import spacy


nlp = spacy.load("xx_ent_wiki_sm")


def split_claim(text):
    claim = nlp(text)
    nouns = [chunk.text for chunk in claim.noun_chunks]
    entities = claim.ents
    return {'nouns':nouns, 'entities':entities}