from utils.tokenizer_simple import *
from ES.es_search import search_and_merge
from utils.c_scorer import *
from utils.fever_db import *
from utils.file_loader import read_json_rows, get_current_time_str
from utils.common import thread_exe


def retrieve_docs(claim):
    nouns, entities = split_claim_spacy(claim)
    cap_phrases = split_claim_regex(claim)
    ents_list = [i[0] for i in entities]
    nouns = list(set(nouns) | set(cap_phrases))
    # print(claim)
    # print(nouns)

    # ['Colin Kaepernick', 'a starting quarterback', 'the 49ers', '63rd season', 'the National Football League']
    # [('Colin Kaepernick', 'PERSON'), ('the 49ers 63rd season', 'DATE'), ('the National Football League', 'ORG')]
    result = search_and_merge(ents_list, nouns)
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
        item['predicted_docids'] = []
    else:
        item['predicted_docids'] = [j.get('id') for j in docs][:10]
    return item


def get_doc_ids_and_fever_score(in_file, out_file, top_k=10, eval=True, log_file=None):
    d_list = read_json_rows(in_file)[10000:20000]
    thread_number = 10
    print("total items: ", len(d_list))
    thread_exe(retri_doc_and_update_item, iter(d_list), thread_number, "query wiki pages")
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


if __name__ == '__main__':
    # i = retrieve_docs("A German-American fashion model starred in The Host.")
    j = retrieve_docs("Trouble with the Curve is a television show.")
    print(j)
    # print(search_doc(['Fox 2000 Pictures', 'Soul Food']))
    # get_doc_ids_and_fever_score(config.LOG_PATH / "test.jsonl", config.RESULT_PATH / f"{get_current_time_str()}_train_doc_retrive.jsonl")
    # get_doc_ids_and_fever_score(config.FEVER_DEV_JSONL,
    #                             config.RESULT_PATH / f"{get_current_time_str()}_train_doc_retrive.jsonl", top_k=10)
    #
    # get_doc_ids_and_fever_score(config.FEVER_TRAIN_JSONL, config.DOC_RETRV_TRAIN)
    # get_doc_ids_and_fever_score(config.FEVER_DEV_JSONL, config.DOC_RETRV_DEV)
    # get_doc_ids_and_fever_score(config.FEVER_TEST_JSONL, config.DOC_RETRV_TEST, eval=False)
    # print(retrieve_docs("Brian Wilson was part of the Beach Boys."))
    # get_doc_ids_and_fever_score(config.FEVER_TEST_JSONL, config.DOC_RETRV_TEST / get_current_time_str())
    # a_list = read_json_rows(config.DOC_RETRV_DEV)
    # fever_doc_only(a_list, a_list, analysis_log=config.LOG_PATH / f"{get_current_time_str()}_doc_retri_no_hits_.jsonl")