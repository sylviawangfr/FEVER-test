import random
import copy

from utils import fever_db, check_sentences
import config

from tqdm import tqdm
from utils import c_scorer, common, text_clean

from data_util.data_preperation.tokenize_fever import easy_tokenize
from utils.file_loader import read_json_rows
from BERT_test.eval_util import convert_evidence2scoring_format
import copy
import random
import numpy as np
from collections import Counter
import math

from tqdm import tqdm

import config
from BERT_test.eval_util import convert_evidence2scoring_format
from data_util.data_preperation.tokenize_fever import easy_tokenize
from utils import c_scorer, common, text_clean
from utils import fever_db, check_sentences
from utils.file_loader import read_json_rows


def sample_data_for_item(item, pred=False):
    res_sentids_list = []
    flags = []
    if pred:
        e_set = check_sentences.get_predicted_evidence(item)
        # return functools.reduce(operator.concat, e_list)
        return list(e_set), None

    if item['verifiable'] == "VERIFIABLE":
        assert item['label'] == 'SUPPORTS' or item['label'] == 'REFUTES'
        e_set = check_sentences.check_and_clean_evidence(item)
        additional_data = item['predicted_sentids']
        # print(len(additional_data))

        for evidences in e_set:
            # print(evidences)
            new_evidences = copy.deepcopy(evidences)
            n_e = len(evidences)
            if n_e < 5:
                current_sample_num = random.randint(0, 5 - n_e)
                random.shuffle(additional_data)
                for sampled_e in additional_data[:current_sample_num]:
                    doc_ids = sampled_e.split(c_scorer.SENT_LINE)[0]
                    ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
                    new_evidences.add_sent(doc_ids, ln)

            if new_evidences != evidences:
                flag = f"verifiable.non_eq.{len(new_evidences) - len(evidences)}"
                flags.append(flag)
                pass
            else:
                flag = "verifiable.eq.0"
                flags.append(flag)
                pass
            res_sentids_list.append(new_evidences)

        assert len(res_sentids_list) == len(e_set)

    elif item['verifiable'] == "NOT VERIFIABLE":
        assert item['label'] == 'NOT ENOUGH INFO'

        e_set = check_sentences.check_and_clean_evidence(item)
        additional_data = item['predicted_sentids']
        # print(len(additional_data))
        random.shuffle(additional_data)
        current_sample_num = random.randint(2, 5)
        raw_evidences_list = []
        for sampled_e in additional_data[:current_sample_num]:
            doc_ids = sampled_e.split(c_scorer.SENT_LINE)[0]
            ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
            raw_evidences_list.append((doc_ids, ln))
        new_evidences = check_sentences.Evidences(raw_evidences_list)

        if len(new_evidences) == 0:
            flag = f"verifiable.eq.0"
            flags.append(flag)
            pass
        else:
            flag = f"not_verifiable.non_eq.{len(new_evidences)}"
            flags.append(flag)

        assert all(len(e) == 0 for e in e_set)
        res_sentids_list.append(new_evidences)
        assert len(res_sentids_list) == 1

    assert len(res_sentids_list) == len(flags)

    return res_sentids_list, flags


# mode in ['train', 'pred', 'eval']
def sample_data_for_item_extend(item, mode='train'):
    res_sentids_list = []
    extended_RS_list = []
    extended_NEI_list = []
    if mode == 'pred':
        e_set = check_sentences.get_predicted_evidence(item)
        # return functools.reduce(operator.concat, e_list)
        return list(e_set), None, None

    def sample_not_in_evidence_set(sampled_e, e_set):
        doc_id = sampled_e.split(c_scorer.SENT_LINE)[0]
        ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
        pick_flag = True
        for one_evidence in e_set:
            if one_evidence.contains(doc_id, ln):
                pick_flag = False
        return pick_flag
    if item['verifiable'] == "VERIFIABLE":
        assert item['label'] == 'SUPPORTS' or item['label'] == 'REFUTES'
        e_set = check_sentences.check_and_clean_evidence(item)
        for evidence in e_set:
            # exact evidence
            res_sentids_list.append(evidence)
            # extend evidence + 2
            if mode == 'train':
                additional_data = item['predicted_sentids']
                n_e = len(evidence)
                # extend_number >= 2
                extend_number = 2  # >= 2
                # extend S and R
                for i in range(extend_number):
                    extended_SR_evidences = copy.deepcopy(evidence)
                    if n_e < 5:
                        additional_sample_num = random.randint(1, 5 - n_e)
                        random.shuffle(additional_data)
                        for sampled_e in additional_data:
                            if additional_sample_num <= 0:
                                extended_RS_list.append(extended_SR_evidences)
                                break
                            if sample_not_in_evidence_set(sampled_e, e_set):
                                doc_ids = sampled_e.split(c_scorer.SENT_LINE)[0]
                                ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
                                extended_SR_evidences.add_sent(doc_ids, ln)
                                additional_sample_num -= 1

                # extend NEI
                for i in range(n_e):
                    extended_NEI_evidence = copy.deepcopy(evidence)
                    extended_NEI_evidence.pop_sent(i)
                    additional_sample_num = random.randint(2-n_e, 5 - n_e)
                    for sampled_e in additional_data:
                        if additional_sample_num <= 0 and extended_NEI_evidence not in extended_NEI_list:
                            extended_NEI_list.append(extended_NEI_evidence)
                            break
                        if sample_not_in_evidence_set(sampled_e, e_set):
                            doc_id = sampled_e.split(c_scorer.SENT_LINE)[0]
                            ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
                            extended_NEI_evidence.add_sent(doc_id, ln)
                            additional_sample_num -= 1
        assert len(res_sentids_list) == len(e_set)

    elif item['verifiable'] == "NOT VERIFIABLE":
        assert item['label'] == 'NOT ENOUGH INFO'
        additional_data = item['predicted_sentids']
        random.shuffle(additional_data)
        additional_sample_num = random.randint(1, 5)
        raw_evidences_list = []
        for sampled_e in additional_data[:additional_sample_num]:
            doc_ids = sampled_e.split(c_scorer.SENT_LINE)[0]
            ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
            raw_evidences_list.append((doc_ids, ln))
        new_evidences = check_sentences.Evidences(raw_evidences_list)
        res_sentids_list.append(new_evidences)
    return res_sentids_list, extended_RS_list, extended_NEI_list

# mode in ['train', 'pred', 'eval']
def get_sample_data(upstream_data, tokenized=False, mode='train'):
    cursor, conn = fever_db.get_cursor()
    if isinstance(upstream_data, list):
        d_list = upstream_data
    else:
        d_list = read_json_rows(upstream_data)

    sampled_data_list = []

    for item in tqdm(d_list, desc="Sampling"):
        # e_list = check_sentences.check_and_clean_evidence(item)
        sampled_e_list, extended_RS_list, extended_NEI_list = sample_data_for_item_extend(item, mode)
        # print(flags)
        for i, sampled_evidence in enumerate(sampled_e_list):
            new_item = dict()
            evidence_text = evidence_list_to_text(cursor, sampled_evidence,
                                                  contain_head=True, id_tokenized=tokenized)
            new_item['id'] = str(item['id']) + '#' + str(i)

            if tokenized:
                new_item['claim'] = item['claim']
            else:
                new_item['claim'] = ' '.join(easy_tokenize(item['claim']))
            new_item['evid'] = evidence_text

            if mode == 'pred':
                new_item['predicted_evidence'] = item['predicted_evidence']
                new_item['predicted_sentids'] = item['predicted_sentids']
                # not used, but to avoid example error
                new_item['label'] = 'NOT ENOUGH INFO'
            else:
                new_item['label'] = item['label']
            sampled_data_list.append(new_item)
        if mode == 'train':
            def init_extended_evidence(tmp_extended_evidence):
                extend_item = dict()
                evidence_text = evidence_list_to_text(cursor, tmp_extended_evidence,
                                                      contain_head=True, id_tokenized=tokenized)
                extend_item['id'] = str(item['id']) + '_' + str(i)
                if tokenized:
                    extend_item['claim'] = item['claim']
                else:
                    extend_item['claim'] = ' '.join(easy_tokenize(item['claim']))
                extend_item['evid'] = evidence_text
                return extend_item

            for idx, extended_evidence in enumerate(extended_RS_list):
                extend_item = init_extended_evidence(extended_evidence)
                extend_item['id'] = extend_item['id'] + '_rs'
                extend_item['label'] = item['label']
                sampled_data_list.append(extend_item)
            for idx, extended_evidence in enumerate(extended_NEI_list):
                extend_item = init_extended_evidence(extended_evidence)
                extend_item['id'] = extend_item['id'] + '_nei'
                extend_item['label'] = 'NOT ENOUGH INFO'
                sampled_data_list.append(extend_item)

    cursor.close()
    print(f"Sampled evidences: {len(sampled_data_list)}")
    return sampled_data_list


def evidence_list_to_text(cursor, evidences, contain_head=True, id_tokenized=False):
    current_evidence_text = []
    evidences = sorted(evidences, key=lambda x: (x[0], x[1]))

    cur_head = 'DO NOT INCLUDE THIS FLAG'

    for doc_id, line_num in evidences:

        _, e_text, _ = fever_db.get_evidence(cursor, doc_id, line_num)

        if contain_head and cur_head != doc_id:
            cur_head = doc_id

            if not id_tokenized:
                doc_id_natural_format = text_clean.convert_brc(doc_id).replace('_', ' ')
                t_doc_id_natural_format = ' '.join(easy_tokenize(doc_id_natural_format))
            else:
                t_doc_id_natural_format = common.doc_id_to_tokenized_text(doc_id)

            if line_num != 0:
                current_evidence_text.append(f"{t_doc_id_natural_format}{c_scorer.SENT_DOC_TITLE}")

        # Important change move one line below: July 16
        current_evidence_text.append(e_text)

    # print(current_evidence_text)

    return ' '.join(current_evidence_text)


def format_printing(item):
    print("-" * 50)
    print("Claim:", item['claim'])
    print("Evidence:", item['evid'])
    # print("Pred Label:", item['predicted_label'])
    # print("Pred Evid:", item['predicted_evidence'])
    # print("Pred Evid F:", item['predicted_sentids'])
    # print("Label:", item['label'])
    # print("Evid:", item['evidence'])
    print("-" * 50)
    return


def eval_samples(sampled_data):
    refused = list(filter(lambda x: (x['label'] == 'REFUTES'), sampled_data))
    nei = list(filter(lambda x: (x['label'] == 'NOT ENOUGH INFO'), sampled_data))
    support = list(filter(lambda x: (x['label'] == 'SUPPORTS'), sampled_data))
    print(f"REFUTES:{len(refused)}, nei:{len(nei)}, SUPPORTS:{len(support)}")


if __name__ == '__main__':
    additional_file = read_json_rows(config.RESULT_PATH / "dev_s_tfidf_retrieve.jsonl")
    t = get_sample_data(additional_file, tokenized=True, mode='train')
    eval_samples(t)

    t = get_sample_data(additional_file, tokenized=True, mode='eval')
    eval_samples(t)

    # complete_upstream_dev_data = additional_file
    # count = Counter()
    # length_list = []
    #
    # for item in complete_upstream_dev_data:
    #     # length_list.extend([len(item['evid'].split(' '))])
    #     length_list.extend([len(e) for e in item['evidence'] if item['verifiable'] == 'VERIFIABLE'])
    #
    # count.update(length_list)
    # print(count.most_common())
    # print(sorted(list(count.most_common()), key=lambda x: -x[0]))
    # print(np.max(length_list))
    # print(np.mean(length_list))
    # print(np.std(length_list))
    #
    # for item in complete_upstream_dev_data[:5]:
    #     format_printing(item)

    # 785
    # 79.13041644297876
    # 43.75476065765309
