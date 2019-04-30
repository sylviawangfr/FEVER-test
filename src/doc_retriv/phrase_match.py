from utils.tokenizer import *


def retrieve_docs(claim):
    claim_chunks = split_claim(claim)

    # ['Colin Kaepernick', 'a starting quarterback', 'the 49ers', '63rd season', 'the National Football League']
    # [('Colin Kaepernick', 'PERSON'), ('the 49ers 63rd season', 'DATE'), ('the National Football League', 'ORG')]
    nouns = claim.get('nouns')
    entities = claim.get('entities')
