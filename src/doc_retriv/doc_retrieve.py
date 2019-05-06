from utils.tokenizer import *
from ES.es_search import search_and_merge
from utils.fever_db import *
from utils.spcl import *
from utils.c_scorer import *


def retrieve_docs(claim):
    nouns, entities = split_claim_spacy(claim)

    # ['Colin Kaepernick', 'a starting quarterback', 'the 49ers', '63rd season', 'the National Football League']
    # [('Colin Kaepernick', 'PERSON'), ('the 49ers 63rd season', 'DATE'), ('the National Football League', 'ORG')]
    result = search_and_merge(entities, nouns)
    result.sort(key=lambda x: x.get('score'), reverse=True)
    if len(result) > 10:
        result = result[:10]
    # reshape = [x.update({'claim_id': claim}) for x in result]
    # return reshape
    return result


def get_doc_ids_and_fever_score(in_file, top_k=5):
    d_list = read_json_rows(in_file)[:500]
    for i, item in enumerate(spcl(d_list)):
        claim = item.get('claim')
        print(claim)
        docs = retrieve_docs(claim)
        item['predicted_docids'] = [j.get('id') for j in docs][:top_k]

    print(fever_doc_only(d_list, d_list))
    eval_mode = {'check_doc_id_correct': True, 'standard': False}
    print(fever_score(d_list, d_list, mode=eval_mode, error_analysis_file=config.PRO_ROOT / 'log/test.log'))


def save_retrs(records):
    conn = sqlite3.connect(str(config.FEVER_DB))
    c = conn.cursor()
    save_doc_retr(c, records)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    # print(search_doc(['Fox 2000 Pictures', 'Soul Food']))
    get_doc_ids_and_fever_score(config.FEVER_DEV_JSONL)
