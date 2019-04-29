from utils.tokenizer import *


def retrieve_docs(claim):
    claim_chunks = split_claim(claim)
    nouns = claim.get('nouns')
    entities = claim.get('entities')