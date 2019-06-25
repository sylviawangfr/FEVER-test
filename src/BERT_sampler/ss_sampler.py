"""
This file contains scripts to build or sample data for neural sentence selector.

Neural sentence selector aimed to fine-select sentence for NLI models since NLI models are sensitive to data.
"""

import json

from sample_for_nli.tf_idf_sample_v1_0 import convert_evidence2scoring_format
from utils import fever_db, common, c_scorer
from utils.file_loader import read_json_rows, get_current_time_str
from tqdm import tqdm
import config
from data_util.data_preperation.tokenize_fever import easy_tokenize
import utils.check_sentences
import itertools
import numpy as np


def get_full_list_sample_for_nn(doc_retrieve_data, pred=False, top_k=None):
    """
    This method will select all the sentence from upstream doc retrieval and label the correct evident as true
    :param tokenized_data_file: Remember this is tokenized data with original format containing 'evidence'
    :param additional_data_file:    This is the data after document retrieval.
                                    This file need to contain *"predicted_docids"* field.
    :return:
    """
    if isinstance(doc_retrieve_data, list):
        d_list = doc_retrieve_data
    else:
        d_list = read_json_rows(doc_retrieve_data)

    if top_k is not None:
        print("Upstream document number truncate to:", top_k)
        trucate_item(d_list, top_k=top_k)

    full_data_list = []

    cursor, conn = fever_db.get_cursor()
    err_log_f = config.LOG_PATH / f"{utils.get_current_time_str()}_analyze_sample.log"
    for item in tqdm(d_list):
        doc_ids = item["predicted_docids"]

        if not pred:
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
    return full_data_list


def trucate_item(d_list, top_k=None):
    for item in d_list:
        if top_k is not None and len(item['predicted_docids']) > top_k:
            item['predicted_docids'] = item['predicted_docids'][:top_k]


def get_full_list_from_list_d(tokenized_data_file, additional_data_file, pred=False, top_k=None):
    """
    This method will select all the sentence from upstream doc retrieval and label the correct evident as true
    :param tokenized_data_file: Remember this is tokenized data with original format containing 'evidence'
    :param additional_data_file:    This is the data after document retrieval.
                                    This file need to contain *"predicted_docids"* field.
    :return:
    """

    d_list = tokenized_data_file

    additional_d_list = additional_data_file

    if top_k is not None:
        print("Upstream document number truncate to:", top_k)
        trucate_item(additional_d_list, top_k=top_k)

    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[add_item['id']] = add_item

    full_data_list = []

    cursor, conn = fever_db.get_cursor()
    for item in tqdm(d_list):
        doc_ids = additional_data_dict[item['id']]["predicted_docids"]

        if not pred:
            if item['evidence'] is not None:
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

        assert len(id_list) == len(set(id_list))  # check duplicate
        assert len(r_list) == len(id_list)

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
            full_data_list.append(sent_item)

    conn.close()
    return full_data_list


def get_additional_list(tokenized_data_file, additional_data_file,
                        item_key='prioritized_docids_aside', top_k=6):
    """
    This method will select all the sentence from upstream doc retrieval and label the correct evident as true
    :param item_key: The item that specify the additional prioritized document ids.
    :param tokenized_data_file: Remember this is tokenized data with original format containing 'evidence'
    :param additional_data_file:    This is the data after document retrieval.
                                    This file need to contain *"predicted_docids"* field.
    :return:
    """
    cursor, conn = fever_db.get_cursor()
    d_list = read_json_rows(tokenized_data_file)

    additional_d_list = read_json_rows(additional_data_file)
    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[int(add_item['id'])] = add_item

    full_data_list = []

    for item in tqdm(d_list):
        doc_ids_p_list = additional_data_dict[int(item['id'])][item_key]
        doc_ids = list(set([k for k, v in sorted(doc_ids_p_list, key=lambda x: (-x[1], x[0]))][:top_k]))

        # if not pred:
        #     if item['evidence'] is not None:
        #         e_list = utils.check_sentences.check_and_clean_evidence(item)
        #         all_evidence_set = set(itertools.chain.from_iterable([evids.evidences_list for evids in e_list]))
        #     else:
        #         all_evidence_set = None
        #     # print(all_evidence_set)
        #     r_list = []
        #     id_list = []
        #
        #     if all_evidence_set is not None:
        #         for doc_id, ln in all_evidence_set:
        #             _, text, _ = fever_db.get_evidence(cursor, doc_id, ln)
        #             r_list.append(text)
        #             id_list.append(doc_id + '(-.-)' + str(ln))
        #
        # else:            # If pred, then reset to not containing ground truth evidence.

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

        assert len(id_list) == len(set(id_list))  # check duplicate
        assert len(r_list) == len(id_list)

        zipped_s_id_list = list(zip(r_list, id_list))
        # Sort using id
        # sorted(evidences_set, key=lambda x: (x[0], x[1]))
        zipped_s_id_list = sorted(zipped_s_id_list, key=lambda x: (x[1][0], x[1][1]))

        all_sent_list = convert_to_formatted_sent(zipped_s_id_list, all_evidence_set, contain_head=True,
                                                  id_tokenized=True)
        cur_id = item['id']
        for i, sent_item in enumerate(all_sent_list):
            sent_item['selection_id'] = str(cur_id) + "<##>" + str(sent_item['sid'])
            # selection_id is '[item_id<##>[doc_id]<SENT_LINE>[line_number]'
            sent_item['query'] = item['claim']
            full_data_list.append(sent_item)

    return full_data_list


def convert_to_formatted_sent(zipped_s_id_list, evidence_set, contain_head=True, id_tokenized=True):
    sent_list = []
    for sent, sid in zipped_s_id_list:
        sent_item = dict()

        cur_sent = sent
        doc_id, ln = sid.split('(-.-)')[0], int(sid.split('(-.-)')[1])
        # print(sent, doc_id, ln)
        if contain_head:
            if not id_tokenized:
                doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')
                t_doc_id_natural_format = ' '.join(easy_tokenize(doc_id_natural_format))
            else:
                t_doc_id_natural_format = common.doc_id_to_tokenized_text(doc_id)

            if ln != 0 and t_doc_id_natural_format.lower() not in sent.lower():
                cur_sent = f"{t_doc_id_natural_format} <t> " + sent

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


def get_tfidf_sample_for_nn(tfidf_ss_data_file, pred=False, top_k=8):
    """
    This method will select all the sentence from upstream tfidf ss retrieval and label the correct evident as true for nn model
    :param tfidf_ss_data_file: Remember this is result of tfidf ss data with original format containing 'evidence' and 'predicted_evidence'

    :return:
    """

    if not isinstance(tfidf_ss_data_file, list):
        d_list = read_json_rows(tfidf_ss_data_file)
    else:
        d_list = tfidf_ss_data_file

    d_list = d_list
    full_sample_list = []

    cursor, conn = fever_db.get_cursor()
    err_log_f = config.LOG_PATH / f"{get_current_time_str()}_analyze_sample.log"
    count_truth = []
    for item in tqdm(d_list):
        predicted_evidence = item["predicted_sentids"]
        ground_truth = item['evidence']
        if not pred:
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
            if num_envs >= top_k:
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
    print(np.sum(count_truth))
    return full_sample_list


def count_truth_examples(sample_list):
    count_hit = 0
    for item in sample_list:
        # print(item)
        if item['selection_label'] == 'true':
            count_hit += 1

    print(f"Truth count/total count: , {count_hit}/{len(sample_list)}/{count_hit / len(sample_list)}")


if __name__ == '__main__':
    # additional_file = config.RESULT_PATH / "tfidf/train_2019_06_15_15:48:58.jsonl"
    # full_list = get_tfidf_sample_for_nn(additional_file)

    # full_list = get_full_list(config.T_FEVER_DEV_JSONL,
    #                           config.RESULT_PATH / "doc_retri/2018_07_04_21:56:49_r/dev.jsonl",
    #                           pred=True)
    # full_list = get_full_list(config.T_FEVER_TRAIN_JSONL,
    # config.RESULT_PATH / "doc_retri/2018_07_04_21:56:49_r/train.jsonl")
    # train_upstream_file = config.RESULT_PATH / "doc_retri/2018_07_04_21:56:49_r/train.jsonl"
    dev_upstream_data = read_json_rows(config.DOC_RETRV_DEV)[0:3]
    complete_upstream_train_data = get_full_list_sample_for_nn(dev_upstream_data, pred=False)
    filtered_train_data = post_filter(complete_upstream_train_data, keep_prob=0.5, seed=12)
    full_list = complete_upstream_train_data

    print(len(full_list))
    print(len(filtered_train_data))
    count_hit = 0
    for item in full_list:
        # print(item)
        if item['selection_label'] == 'true':
            count_hit += 1

    print(count_hit, len(full_list), count_hit / len(full_list))

    # d_list = navie_results_builder_for_sanity_check(config.T_FEVER_DEV_JSONL, full_list)
    # # d_list = navie_results_builder_for_sanity_check(config.T_FEVER_TRAIN_JSONL, full_list)
    # eval_mode = {'check_sent_id_correct': True, 'standard': True}
    # print(c_scorer.fever_score(d_list, config.T_FEVER_DEV_JSONL, mode=eval_mode, verbose=False))
    #
    # total = len(d_list)
    # hit = eval_mode['check_sent_id_correct_hits']
    # tracking_score = hit / total
    # print("Tracking:", tracking_score)

    # for item in full_list:
    #     print(item)