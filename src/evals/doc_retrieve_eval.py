from doc_retriv.doc_retrieve import *

def eval_doc_retrieve(upstrem_data, origin_data):
    # eval_mode = {'check_doc_id_correct': True, 'standard': False}
    fever_doc_only(upstrem_data, actual=upstrem_data, max_evidence=5,
                   analysis_log=config.LOG_PATH / 'aug05_dev_doc_top5.jsonl')
    fever_doc_only(upstrem_data, actual=upstrem_data, max_evidence=10,
                   analysis_log=config.LOG_PATH / 'aug05_dev_doc_top10.jsonl')
    # fever_score(upstrem_data,
    #             origin_data,
    #             mode=eval_mode,
    #             error_analysis_file=config.LOG_PATH / 'aug05_dev_doc_error_items.log')
    #

def eval_claim(upstream_data, claim_id):
    pass


if __name__ == "__main__":
    upstream_data = read_json_rows(config.DOC_RETRV_DEV)
    eval_doc_retrieve(upstream_data, upstream_data)