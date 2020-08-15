from utils.c_scorer import *

time = "aug05_top10_"


def create_hard_set():
    # doc retrieve
    # time = get_current_time_str()
    tfidf_upstream = read_json_rows(config.RESULT_PATH / 'dev_s_tfidf_retrieve.jsonl')
    orgin_data = read_json_rows(config.FEVER_DEV_JSONL)
    eval_mode = {'check_sent_id_correct': True, 'standard': False}
    fever_score(tfidf_upstream, orgin_data, max_evidence=10, mode=eval_mode,
                error_analysis_file=config.LOG_PATH / f'{time}_hardset_from_tfidf.jsonl',)

    print("done with creating dev tfidf hardset")


def clean_hard_set(hard_set):
    new_set = []
    for i in hard_set:
        new_set.append({'id': i['id'],
                        "verifiable": i['verifiable'],
                        "label": i["label"],
                        "claim": i["claim"],
                        "evidence": i["evidence"]})
    save_intermidiate_results(new_set, config.RESULT_PATH / 'hardset.jsonl')

if __name__ == "__main__":
    hard_set = read_json_rows(config.LOG_PATH / 'aug05_top5_hardset_from_tfidf.jsonl')
    clean_hard_set(hard_set)

