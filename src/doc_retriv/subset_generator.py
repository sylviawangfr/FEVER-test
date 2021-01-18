from utils.c_scorer import *
from utils.fever_db import *
from utils.file_loader import read_json_rows, get_current_time_str, read_all_files, save_and_append_results

from utils.check_sentences import Evidences
from utils.tokenizer_simple import *
from dbpedia_sampler.sentence_util import get_ents_and_phrases

def generate_es_error_items():
    result_has_combination_evidence = []
    data = read_json_rows(config.RESULT_PATH / "errors/es_doc_10_1.log")
    for item in data:
        has_multi = False
        has_single = False
        e_list = utils.check_sentences.check_and_clean_evidence(item)
        for e in e_list:
            if len(e) > 1:
                has_multi = True
            if len(e) == 1:
                has_single = True
        if not has_multi:
            result_has_combination_evidence.append(item)
    save_intermidiate_results(result_has_combination_evidence, config.RESULT_PATH / "errors/es_errors.jsonl")


def generate_subset_multi_evidence_articals():
    data = read_json_rows(config.FEVER_DEV_JSONL)
    result_combination_evidence_only = []
    result_has_combination_evidence = []
    result_has_more_than_one_docs = []
    for item in data:
        has_multi = False
        has_single = False
        has_more_docs = False
        e_list = utils.check_sentences.check_and_clean_evidence(item)
        for e in e_list:
            if len(e) > 1:
                has_multi = True
            if len(e) == 1:
                has_single = True
            if len(set(i[0] for i in e.evidences_list)) > 1:
                has_more_docs = True
        if has_multi:
            result_has_combination_evidence.append(item)
        if has_multi and not has_single:
            result_combination_evidence_only.append(item)
        if has_more_docs:
            result_has_more_than_one_docs.append(item)
    print(f"multi-sentence:{len(result_has_combination_evidence)}, multi_docs:{len(result_has_more_than_one_docs)}")

    save_intermidiate_results(result_combination_evidence_only, config.RESULT_PATH / "hardset2021/dev_has_multi_evidence_only.jsonl")
    save_intermidiate_results(result_has_combination_evidence, config.DATA_ROOT / "hardset2021/dev_has_multi_evidence.jsonl")
    save_intermidiate_results(result_has_more_than_one_docs, config.DATA_ROOT / "hardset2021/dev_has_multi_doc_evidence.jsonl")


def generate_subset_no_capital():
    data = read_json_rows(config.FEVER_DEV_JSONL)
    result = []
    for item in data:
        sentence = item['claim']
        capitalized_phrased = list(set(split_claim_regex(sentence)))
        if len(capitalized_phrased) == 1 \
                and capitalized_phrased[0].count(' ') == 0 \
                and sentence.startswith(capitalized_phrased[0]):
            get_ents_and_phrases(sentence)
    # save_intermidiate_results(result, config.DATA_ROOT / "dev_no_capitals.jsonl")


if __name__ == '__main__':
    generate_subset_multi_evidence_articals()
    # generate_subset_no_capital()
    # t = read_json_rows(config.DATA_ROOT / "dev_has_multi_evidence.jsonl")
    # generate_es_error_items()