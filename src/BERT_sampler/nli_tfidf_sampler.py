import random
import copy
from utils import check_sentences

from utils import c_scorer, common
from collections import Counter
import numpy as np
from utils.file_loader import *
from data_util.tokenizers import SpacyTokenizer
from BERT_test.eval_util import convert_evidence2scoring_format
import functools
import operator

tok = SpacyTokenizer()

random.seed = 12


def easy_tokenize(text):
    if tok.instance is None:
        tok.create_instance()
    return tok.tokenize(text_clean.normalize(text)).words()


def sample_data_for_item(item, pred=False):
    res_sentids_list = []
    flags = []
    if pred:
        e_list = check_sentences.get_predicted_evidence(item)
        return functools.reduce(operator.concat, e_list)

    if item['verifiable'] == "VERIFIABLE":
        assert item['label'] == 'SUPPORTS' or item['label'] == 'REFUTES'
        e_list = check_sentences.check_and_clean_evidence(item)
        additional_data = item['predicted_sentids']

        for evidences in e_list:
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

        assert len(res_sentids_list) == len(e_list)

    elif item['verifiable'] == "NOT VERIFIABLE":
        assert item['label'] == 'NOT ENOUGH INFO'

        e_list = check_sentences.check_and_clean_evidence(item)
        additional_data = item['predicted_sentids']
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

        assert all(len(e) == 0 for e in e_list)
        res_sentids_list.append(new_evidences)
        assert len(res_sentids_list) == 1

    assert len(res_sentids_list) == len(flags)
    return res_sentids_list, flags


def evidence_list_to_text(cursor, evidences, contain_head=True, id_tokenized=False):
    current_evidence_text = []
    evidences = sorted(evidences, key=lambda x: (x[0], x[1]))

    cur_head = 'DO NOT INCLUDE THIS FLAG'

    for doc_id, line_num in evidences:

        _, e_text, _ = fever_db.get_evidence(cursor, doc_id, line_num)

        if contain_head and cur_head != doc_id:
            cur_head = doc_id

            doc_id = normalize(doc_id)
            if not id_tokenized:
                doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')
                t_doc_id_natural_format = ' '.join(easy_tokenize(doc_id_natural_format))
            else:
                t_doc_id_natural_format = common.doc_id_to_tokenized_text(doc_id)

            if line_num != 0:
                current_evidence_text.append(f"{t_doc_id_natural_format} <t>")

        # Important change move one line below: July 16
        current_evidence_text.append(e_text)

    # print(current_evidence_text)

    return ' '.join(current_evidence_text)


def get_sample_data(upstream_data, tokenized=True, pred=False):
    cursor, conn = fever_db.get_cursor()
    if not isinstance(upstream_data, list):
        d_list = read_json_rows(upstream_data)
    else:
        d_list = upstream_data

    sampled_data_list = []

    for item in tqdm(d_list):
        # e_list = check_sentences.check_and_clean_evidence(item)
        sampled_e_list, flags = sample_data_for_item(item, pred=pred)
        # print(flags)
        for i, (sampled_evidence, flag) in enumerate(zip(sampled_e_list, flags)):
            new_item = dict()
            evidence_text = evidence_list_to_text(cursor, sampled_evidence,
                                                  contain_head=True, id_tokenized=tokenized)

            new_item['id'] = str(item['id']) + '#' + str(i)

            if tokenized:
                new_item['claim'] = item['claim']
            else:
                new_item['claim'] = ' '.join(easy_tokenize(item['claim']))

            new_item['evid'] = evidence_text
            new_item['predicted_sentids'] = item['predicted_sentids']
            new_item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])
            new_item['verifiable'] = item['verifiable']
            new_item['label'] = item['label']

            # print("C:", new_item['claim'])
            # print("E:", new_item['evid'])
            # print("L:", new_item['label'])
            # print()
            sampled_data_list.append(new_item)

    return sampled_data_list


def eval_sample_length():
    sampled_d_list = get_sample_data(config.RESULT_PATH / "dev_s_tfidf_retrieve.jsonl", tokenized=True)
    count = Counter()
    length_list = []
    for item in sampled_d_list:
        length_list.extend([len(item['evid'].split(' '))])

    count.update(length_list)
    print(count.most_common())
    print(sorted(list(count.most_common()), key=lambda x: -x[0]))
    print(np.max(length_list))
    print(np.mean(length_list))
    print(np.std(length_list))


if __name__ == '__main__':
    additional_file = read_json_rows(config.RESULT_PATH / "dev_s_tfidf_retrieve.jsonl")[0:3]
    t = get_sample_data(additional_file, tokenized=True)
    print(t)

