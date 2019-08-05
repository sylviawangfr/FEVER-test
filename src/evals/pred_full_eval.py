from doc_retriv.doc_retrieve import *

def eval_pred(upstrem_data, origin_data):
    eval_mode = {'check_sent_id_correct': True, 'standard': True}

    fever_score(upstrem_data,
                origin_data,
                mode=eval_mode,
                error_analysis_file=config.LOG_PATH / 'aug05_dev_pred_error_items.log')


def eval_claim(upstream_data, claim_id):
    pass


if __name__ == "__main__":
    upstream_data = read_json_rows(config.RESULT_PATH / 'eval_data_nli_dev_0.5_top5.jsonl')
    original_data = read_json_rows(config.FEVER_DEV_JSONL)
    eval_pred(upstream_data, upstream_data)