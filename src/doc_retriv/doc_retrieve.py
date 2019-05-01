from utils.tokenizer import *
from ES.es_search import search_and_merge
from utils.fever_db import *


def retrieve_docs(claim):
    claim_chunks = split_claim(claim)

    # ['Colin Kaepernick', 'a starting quarterback', 'the 49ers', '63rd season', 'the National Football League']
    # [('Colin Kaepernick', 'PERSON'), ('the 49ers 63rd season', 'DATE'), ('the National Football League', 'ORG')]
    nouns = claim.get('nouns')
    entities = claim.get('entities')
    result = search_and_merge(entities, nouns)
    result.sort(key=lambda x: x.get('score'))
    if len(result) > 10:
        result = result[:10]
    reshape = [x.update({'claim_id': claim}) for x in result]
    return reshape


def save_retrs(records):
    conn = sqlite3.connect(str(config.FEVER_DB))
    c = conn.cursor()
    save_doc_retr(c, records)
    conn.commit()
    conn.close()

