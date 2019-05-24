from utils.tokenizer_simple import *
from ES.es_search import search_and_merge
from utils.fever_db import *
from utils.spcl import *
from utils.c_scorer import *
from utils.fever_db import *
import json


def retrieve_docs(claim):
    nouns, entities = split_claim_spacy(claim)
    cap_phrases = split_claim_regex(claim)
    nouns = list(set(nouns) | set(cap_phrases))


    # ['Colin Kaepernick', 'a starting quarterback', 'the 49ers', '63rd season', 'the National Football League']
    # [('Colin Kaepernick', 'PERSON'), ('the 49ers 63rd season', 'DATE'), ('the National Football League', 'ORG')]
    result = search_and_merge(entities, nouns)
    result.sort(key=lambda x: x.get('score'), reverse=True)
    if len(result) > 10:
        result = result[:10]
    # reshape = [x.update({'claim_id': claim}) for x in result]
    # return reshape
    return result


def retri_doc_and_update_item(item):
    claim = item.get('claim')
    docs = retrieve_docs(claim)
    if len(docs) < 1:
        print("failed claim:", item.get('id'))

    item['predicted_docids'] = [j.get('id') for j in docs][:10]
    return item


def get_doc_ids_and_fever_score(in_file, out_file, top_k=10):
    d_list = read_json_rows(in_file)
    retri_list = []
    # cursor = get_cursor()
    thread_number = 8
    thread_exe(retri_doc_and_update_item, iter(d_list), thread_number, "query wiki pages")
    save_intermidiate_results(d_list, out_file)
    print(fever_doc_only(d_list, d_list))
    eval_mode = {'check_doc_id_correct': True, 'standard': False}
    out_fname = config.LOG_PATH / f"{utils.get_current_time_str()}_analyze_doc_retri.log"
    print(fever_score(d_list, d_list, mode=eval_mode, error_analysis_file=out_fname))




def save_retrs(records):
    conn = sqlite3.connect(str(config.FEVER_DB))
    c = conn.cursor()
    save_doc_retr(c, records)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    # print(search_doc(['Fox 2000 Pictures', 'Soul Food']))
    # get_doc_ids_and_fever_score(config.FEVER_DEV_JSONL, config.DOC_RETRV_DEV)
    get_doc_ids_and_fever_score(config.FEVER_TRAIN_JSONL, config.DOC_RETRV_TRAIN)
    # retrieve_docs("Telemundo is a English-language television network.")
