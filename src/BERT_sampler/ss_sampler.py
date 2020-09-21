"""
This file contains scripts to build or sample data for neural sentence selector.

Neural sentence selector aimed to fine-select sentence for NLI models since NLI models are sensitive to data.
"""


import itertools
from collections import Counter

import numpy as np

import log_util
import utils.check_sentences
import utils.common_types as bert_para
from data_util.data_preperation.tokenize_fever import easy_tokenize
from utils import common, c_scorer
from utils.file_loader import *
from utils.text_clean import convert_brc
from memory_profiler import profile


logger = log_util.get_logger('ss_sampler')


def get_full_list_sample(paras: bert_para.PipelineParas):
    """
    This method will select all the sentence from upstream doc retrieval and label the correct evident as true
    :param tokenized_data_file: Remember this is tokenized data with original format containing 'evidence'
    :param additional_data_file:    This is the data after document retrieval.
                                    This file need to contain *"predicted_docids"* field.
    :return:
    """
    if isinstance(paras.upstream_data, list):
        d_list = paras.upstream_data
    else:
        d_list = read_json_rows(paras.upstream_data)

    if paras.sample_n is not None:
        logger.info(f"Upstream document number truncate to: {paras.sample_n}")
        trucate_item(d_list, top_k=paras.sample_n)

    full_data_list = []

    cursor, conn = fever_db.get_cursor()
    err_log_f = config.LOG_PATH / f"{utils.get_current_time_str()}_analyze_sample.log"
    for item in tqdm(d_list, desc="Sampling:"):
        doc_ids = item["predicted_docids"]

        if not paras.pred:
            if item['evidence'] is not None:
                # ground truth
                e_list = utils.check_sentences.check_and_clean_evidence(item)
                all_evidence_set = set(itertools.chain.from_iterable([evids.evidences_list for evids in e_list]))
            else:
                all_evidence_set = None
            # print(all_evidence_set)
            r_list = []
            id_list = []

            if all_evidence_set is not None:
                for doc_id, ln in all_evidence_set:
                    _, text, _ = fever_db.get_evidence(cursor, doc_id, ln)
                    r_list.append(text)
                    id_list.append(doc_id + '(-.-)' + str(ln))

        else:            # If pred, then reset to not containing ground truth evidence.
            all_evidence_set = None
            r_list = []
            id_list = []

        for doc_id in doc_ids:
            cur_r_list, cur_id_list = fever_db.get_all_sent_by_doc_id(cursor, doc_id, with_h_links=False)
            # Merging to data list and removing duplicate
            for i in range(len(cur_r_list)):
                if cur_id_list[i] in id_list:
                    continue
                else:
                    r_list.append(cur_r_list[i])
                    id_list.append(cur_id_list[i])

        # assert len(id_list) == len(set(id_list))  # check duplicate
        # assert len(r_list) == len(id_list)
        if not (len(id_list) == len(set(id_list)) or len(r_list) == len(id_list)):
            utils.get_adv_print_func(err_log_f)

        zipped_s_id_list = list(zip(r_list, id_list))
        # Sort using id
        # sorted(evidences_set, key=lambda x: (x[0], x[1]))
        zipped_s_id_list = sorted(zipped_s_id_list, key=lambda x: (x[1][0], x[1][1]))

        all_sent_list = convert_to_formatted_sent(zipped_s_id_list, all_evidence_set, contain_head=True,
                                                  id_tokenized=True)
        cur_id = item['id']
        for i, sent_item in enumerate(all_sent_list):
            sent_item['selection_id'] = str(cur_id) + "<##>" + str(sent_item['sid'])
            sent_item['query'] = item['claim']

            if 'label' in item.keys():
                sent_item['claim_label'] = item['label']

            full_data_list.append(sent_item)

    cursor.close()
    conn.close()
    if not paras.post_filter_prob == 1:
        return post_filter(full_data_list, keep_prob=paras.post_filter_prob, seed=12)
    else:
        return full_data_list


def trucate_item(d_list, top_k=None):
    for item in d_list:
        if top_k is not None and len(item['predicted_docids']) > top_k:
            item['predicted_docids'] = item['predicted_docids'][:top_k]
    return

# @profile
def convert_to_formatted_sent(zipped_s_id_list, evidence_set, contain_head=True, id_tokenized=True):
    sent_list = []
    for sent, sid in zipped_s_id_list:
        sent_item = dict()

        cur_sent = sent
        doc_id, ln = sid.split('(-.-)')[0], int(sid.split('(-.-)')[1])
        # print(sent, doc_id, ln)
        if contain_head:
            if not id_tokenized:
                doc_id_natural_format = convert_brc(doc_id).replace('_', ' ')
                t_doc_id_natural_format = ' '.join(easy_tokenize(doc_id_natural_format, common.tokenizer_spacy))
            else:
                t_doc_id_natural_format = common.doc_id_to_tokenized_text(doc_id)

            if ln != 0 and t_doc_id_natural_format.lower() not in sent.lower():
                cur_sent = f"{t_doc_id_natural_format}{c_scorer.SENT_DOC_TITLE}" + sent

            sent_item['text'] = cur_sent
            sent_item['sid'] = doc_id + c_scorer.SENT_LINE + str(ln)
            # sid is '[doc_id]<SENT_LINE>[line_number]'
            if evidence_set is not None:
                if (doc_id, ln) in evidence_set:
                    sent_item['selection_label'] = "true"
                else:
                    sent_item['selection_label'] = "false"
            else:
                sent_item['selection_label'] = "false"

            sent_list.append(sent_item)
        else:
            sent_list.append(sent_item)

    # for s in sent_list:
    # print(s['text'][:20], s['selection_label'])
    return sent_list


def post_filter(d_list, keep_prob=0.75, seed=12):
    np.random.seed(seed)
    r_list = []
    for item in d_list:
        if item['selection_label'] == 'false':
            if np.random.random(1) >= keep_prob:
                continue
        r_list.append(item)
    return r_list

# def get_formatted_sent(cursor, doc_id):
#     fever_db.get_all_sent_by_doc_id(cursor, doc_id)


def get_tfidf_sample(paras: bert_para.PipelineParas):
    """
    This method will select all the sentence from upstream tfidf ss retrieval and label the correct evident as true for nn model
    :param tfidf_ss_data_file: Remember this is result of tfidf ss data with original format containing 'evidence' and 'predicted_evidence'

    :return:
    """

    if not isinstance(paras.upstream_data, list):
        d_list = read_json_rows(paras.upstream_data)
    else:
        d_list = paras.upstream_data

    full_sample_list = []

    cursor, conn = fever_db.get_cursor()
    err_log_f = config.LOG_PATH / f"{get_current_time_str()}_analyze_sample.log"
    count_truth = []
    for item in tqdm(d_list):
        predicted_evidence = item["predicted_sentids"]
        ground_truth = item['evidence']
        if not paras.pred:
            if ground_truth is not None and len(ground_truth) > 0:
                e_list = utils.check_sentences.check_and_clean_evidence(item)
                all_evidence_set = set(itertools.chain.from_iterable([evids.evidences_list for evids in e_list]))
            else:
                all_evidence_set = None
            # print(all_evidence_set)
            r_list = []
            id_list = []

            if all_evidence_set is not None:
                for doc_id, ln in all_evidence_set:
                    _, text, _ = fever_db.get_evidence(cursor, doc_id, ln)
                    r_list.append(text)
                    id_list.append(doc_id + '(-.-)' + str(ln))

        else:            # If pred, then reset to not containing ground truth evidence.
            all_evidence_set = None
            r_list = []
            id_list = []

        num_envs = 0 if all_evidence_set is None else len(all_evidence_set)
        count_truth.append(num_envs)
        for pred_item in predicted_evidence:
            if num_envs >= paras.sample_n:
                break
            doc_id, ln = pred_item.split(c_scorer.SENT_LINE)[0], int(pred_item.split(c_scorer.SENT_LINE)[1])
            tmp_id = doc_id + '(-.-)' + str(ln)
            if not tmp_id in id_list:
                _, text, _ = fever_db.get_evidence(cursor, doc_id, ln)
                r_list.append(text)
                id_list.append(tmp_id)
                num_envs = num_envs + 1


        # assert len(id_list) == len(set(id_list))  # check duplicate
        # assert len(r_list) == len(id_list)
        if not (len(id_list) == len(set(id_list)) or len(r_list) == len(id_list)):
            utils.get_adv_print_func(err_log_f)

        zipped_s_id_list = list(zip(r_list, id_list))
        # Sort using id
        # sorted(evidences_set, key=lambda x: (x[0], x[1]))
        zipped_s_id_list = sorted(zipped_s_id_list, key=lambda x: (x[1][0], x[1][1]))

        all_sent_list = convert_to_formatted_sent(zipped_s_id_list, all_evidence_set, contain_head=True,
                                                  id_tokenized=True)
        cur_id = item['id']
        for i, sent_item in enumerate(all_sent_list):
            sent_item['selection_id'] = str(cur_id) + "<##>" + str(sent_item['sid'])
            sent_item['query'] = item['claim']

            if 'label' in item.keys():
                sent_item['claim_label'] = item['label']

            full_sample_list.append(sent_item)

    cursor.close()
    conn.close()
    count_truth_examples(full_sample_list)
    logger.info(np.sum(count_truth))
    return full_sample_list


def count_truth_examples(sample_list):
    count_hit = 0
    for item in sample_list:
        # print(item)
        if item['selection_label'] == 'true':
            count_hit += 1
    print(f"truth_count/total_count/rate: , {count_hit}/{len(sample_list)}/{count_hit / len(sample_list)}")
    return


def eval_sample_length(samples):
    count = Counter()
    length_list = []
    for item in samples:
        length_list.extend([len(item['text'].split(' '))])

    count.update(length_list)
    # print(f"most common: {count.most_common}")
    print(f"most_common: {sorted(list(count.most_common()), key=lambda x: -x[0])}")
    print(f"max_length: {np.max(length_list)}")
    print(f"mean: {np.mean(length_list)}")
    print(f"std: {np.std(length_list)}")
    return


if __name__ == '__main__':
    logger.info("test")
    paras = bert_para.PipelineParas()
    paras.upstream_data = read_json_rows(config.RESULT_PATH / "dev_s_tfidf_retrieve.jsonl")[5:10]
    paras.sample_n = 3
    paras.pred = False
    sample_tfidf = get_tfidf_sample(paras)
    eval_sample_length(sample_tfidf)
    count_truth_examples(sample_tfidf)

    paras2 = bert_para.PipelineParas()
    dev_upstream_data = read_json_rows(config.DOC_RETRV_DEV)[5:10]
    paras2.upstream_data = dev_upstream_data
    paras2.pred = False
    paras2.post_filter_prob = 0.5
    complete_upstream_train_data = get_full_list_sample(paras2)
    filtered_train_data = complete_upstream_train_data
    full_list = complete_upstream_train_data
    eval_sample_length(full_list)
    count_truth_examples(full_list)


