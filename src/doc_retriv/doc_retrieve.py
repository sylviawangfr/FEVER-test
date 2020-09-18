from ES.es_search import search_and_merge, search_doc_id
from utils.c_scorer import *
from utils.common import thread_exe
from utils.fever_db import *
from utils.file_loader import read_json_rows, get_current_time_str
from dbpedia_sampler.dbpedia_triple_linker import link_sent_to_resources_multi
from dbpedia_sampler.dbpedia_virtuoso import get_resource_wiki_page
from dbpedia_sampler.sentence_util import get_phrases
import difflib
from utils.text_clean import convert_brc


def retrieve_docs(claim):
    entities, nouns = get_phrases(claim)
    result_es = search_and_merge(entities, nouns)
    result_dbpedia = search_dbpedia(claim)
    result = merge_es_and_dbpedia(result_es, result_dbpedia)
    if len(result) > 10:
        result = result[:10]
    return result


def merge_es_and_dbpedia(r_es, r_db):
    r_es_ids = [i['id'] for i in r_es]
    r_db_ids = [i['id'] for i in r_db]
    for idx_i, i in enumerate(r_es_ids):
        for idx_j, j in enumerate(r_db_ids):
            if i == j:
                if len(r_es[idx_i]['phrases']) > 1:
                    r_es[idx_i]['score'] += r_db[idx_j]['score']
                else:
                    p = r_db[idx_j]['phrases'][0].lower()
                    doc_id = convert_brc(r_db[idx_j]['id']).replace('_', ' ').lower()
                    ratio = difflib.SequenceMatcher(None, p, doc_id).ratio()
                    if ratio > 0.8:
                        r_es[idx_i]['score'] += r_db[idx_j]['score'] * 0.5
    merged = r_es
    for idx, i in enumerate(r_db_ids):
        if i not in r_es_ids:
            p = r_db[idx]['phrases'][0].lower()
            doc_id = convert_brc(r_db[idx]['id']).replace('_', ' ').lower()
            ratio = difflib.SequenceMatcher(None, p, doc_id).ratio()
            if ratio > 0.8:
                r_db[idx]['score'] *= 2
            merged.append(r_db[idx])
    merged.sort(key=lambda x: x.get('score'), reverse=True)
    return merged


def search_dbpedia(claim):
    not_linked_phrases_l, linked_phrases_l = link_sent_to_resources_multi(claim)
    docs = []
    for resource in linked_phrases_l:
        resource_uri = resource['URI']
        wiki_links = get_resource_wiki_page(resource_uri)
        if wiki_links is None or len(wiki_links) < 1:
            continue
        for item in wiki_links:
            possible_doc_id = item.split('/')[-1]
            verified_id_es = search_doc_id(possible_doc_id)
            for r_es in verified_id_es:
                if len(list(filter(lambda x: (x['id'] == r_es['id']), docs))) < 1:
                    docs.append({'id': r_es['id'], 'score': r_es['score'], 'phrases': [resource['text']]})
    return docs


def retri_doc_and_update_item(item):
    claim = item.get('claim')
    docs = retrieve_docs(claim)
    if len(docs) < 1:
        print("failed claim:", item.get('id'))
        item['predicted_docids'] = []
    else:
        item['predicted_docids'] = [j.get('id') for j in docs][:10]
    return item


def get_doc_ids_and_fever_score(in_file, out_file, top_k=10, eval=True, log_file=None):
    d_list = read_json_rows(in_file)

    print("total items: ", len(d_list))
    for i in tqdm(d_list):
        retri_doc_and_update_item(i)
    # thread_number = 2
    # thread_exe(retri_doc_and_update_item, iter(d_list), thread_number, "query wiki pages")
    save_intermidiate_results(d_list, out_file)
    if eval:
        eval_doc_preds(d_list, top_k, log_file)
    return d_list


def eval_doc_preds(doc_list, top_k, log_file):
    print(fever_doc_only(doc_list, doc_list, max_evidence=top_k,
                         analysis_log=config.LOG_PATH / f"{get_current_time_str()}_doc_retri_no_hits.jsonl"))
    eval_mode = {'check_doc_id_correct': True, 'standard': False}
    if log_file is None:
        log_file = config.LOG_PATH / f"{utils.get_current_time_str()}_analyze_doc_retri.log"
    print(fever_score(doc_list, doc_list, mode=eval_mode, error_analysis_file=log_file))


def rerun_failed_items(full_retri_doc, failed_list, updated_file_name):
    r_list = read_json_rows(full_retri_doc)
    for i in r_list:
        if i['id'] in failed_list:
            retri_doc_and_update_item(i)
    save_intermidiate_results(r_list, updated_file_name)


if __name__ == '__main__':
    # i = retrieve_docs("The Dark Tower is a fantasy film.")
    # print(i)
    # j = retrieve_docs("Trouble with the Curve")
    # print(j)
    # get_doc_ids_and_fever_score(config.LOG_PATH / "test.jsonl", config.RESULT_PATH / f"{get_current_time_str()}_train_doc_retrive.jsonl")
    # get_doc_ids_and_fever_score(config.FEVER_DEV_JSONL,
    #                             config.RESULT_PATH / f"{get_current_time_str()}_train_doc_retrive.jsonl", top_k=10)
    #
    # get_doc_ids_and_fever_score(config.FEVER_TRAIN_JSONL, config.DOC_RETRV_TRAIN)
    get_doc_ids_and_fever_score(config.FEVER_DEV_JSONL, config.RESULT_PATH / f"doc_dev_{get_current_time_str()}.jsonl")
    # get_doc_ids_and_fever_score(config.FEVER_TEST_JSONL, config.DOC_RETRV_TEST, eval=False)
    # print(retrieve_docs("Brian Wilson was part of the Beach Boys."))
    # get_doc_ids_and_fever_score(config.FEVER_TEST_JSONL, config.DOC_RETRV_TEST / get_current_time_str())
    # a_list = read_json_rows(config.DOC_RETRV_DEV)
    # fever_doc_only(a_list, a_list, analysis_log=config.LOG_PATH / f"{get_current_time_str()}_doc_retri_no_hits_.jsonl")
    # rerun_failed_items(config.DOC_RETRV_TEST, [49649, 24225, 149500,202840,64863], config.RESULT_PATH / 'test_update.jsonl')