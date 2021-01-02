from ES.es_search import search_and_merge, search_doc_id, search_and_merge2, search_and_merge4, merge_result, search_doc_id_and_keywords, search_doc_id_and_keywords_in_sentences
from utils.c_scorer import *
from utils.common import thread_exe
from utils.fever_db import *
from utils.file_loader import read_json_rows, get_current_time_str, read_all_files, save_and_append_results
from dbpedia_sampler.dbpedia_triple_linker import link_sentence
from dbpedia_sampler.dbpedia_virtuoso import get_resource_wiki_page
from dbpedia_sampler.sentence_util import get_ents_and_phrases, get_phrases_and_nouns_merged
import difflib
from utils.text_clean import convert_brc
from dbpedia_sampler.dbpedia_subgraph import construct_subgraph_for_claim, construct_subgraph_for_candidate
from dbpedia_sampler.uri_util import isURI
from dbpedia_sampler.dbpedia_virtuoso import get_categories2
from bert_serving.client import BertClient
from doc_retriv.SentenceEvidence import *
from utils.check_sentences import Evidences
import copy
from typing import List
import itertools
from __future__ import absolute_import, division, print_function
from BERT_test.ss_eval import *



def filter_bert_claim_vs_sents(claim, docs):
    pass


def prepare_candidate_sents_bert_dev(original_data, data_with_candidate_docs, output_folder):
    paras = bert_para.PipelineParas()
    paras.pred = True
    paras.mode = 'dev'
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_3s_full2019_07_17_04:00:55"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_3s_full2019_07_17_04:00:55"
    paras.output_folder = output_folder
    paras.original_data = original_data
    paras.upstream_data = data_with_candidate_docs
    paras.sample_n = 20
    paras.top_n = [10, 5]
    paras.prob_thresholds = [0.1, 0.4]
    pred_ss_and_save(paras)


if __name__ == '__main__':
    folder = config.RESULT_PATH / "extend_20201231"
    org_data = read_json_rows(config.FEVER_DEV_JSONL)
    candidate_docs = read_json_rows(folder / "candidate_docs.jsonl")
    prepare_candidate_sents_bert_dev(original_data, candidate_docs, folder)

