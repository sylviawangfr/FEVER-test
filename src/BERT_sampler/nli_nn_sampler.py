import copy
import random
from tqdm import tqdm
import config
from data_util.data_preperation.tokenize_fever import easy_tokenize
from utils import c_scorer, common, text_clean
from utils import fever_db, check_sentences
from utils.file_loader import read_json_rows, save_intermidiate_results
import utils.common_types as bert_para
from utils.resource_manager import FeverDBResource
import itertools

# def sample_data_for_item(item, pred=False):
#     res_sentids_list = []
#     flags = []
#     if pred:
#         e_set = check_sentences.get_predicted_evidence(item)
#         # return functools.reduce(operator.concat, e_list)
#         return list(e_set), None
#
#     if item['verifiable'] == "VERIFIABLE":
#         assert item['label'] == 'SUPPORTS' or item['label'] == 'REFUTES'
#         e_set = check_sentences.check_and_clean_evidence(item)
#         additional_data = item['predicted_sentids']
#         # print(len(additional_data))
#
#         for evidences in e_set:
#             # print(evidences)
#             new_evidences = copy.deepcopy(evidences)
#             n_e = len(evidences)
#             if n_e < 5:
#                 current_sample_num = random.randint(0, 5 - n_e)
#                 random.shuffle(additional_data)
#                 for sampled_e in additional_data[:current_sample_num]:
#                     doc_ids = sampled_e.split(c_scorer.SENT_LINE)[0]
#                     ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
#                     new_evidences.add_sent(doc_ids, ln)
#
#             if new_evidences != evidences:
#                 flag = f"verifiable.non_eq.{len(new_evidences) - len(evidences)}"
#                 flags.append(flag)
#                 pass
#             else:
#                 flag = "verifiable.eq.0"
#                 flags.append(flag)
#                 pass
#             res_sentids_list.append(new_evidences)
#
#         assert len(res_sentids_list) == len(e_set)
#
#     elif item['verifiable'] == "NOT VERIFIABLE":
#         assert item['label'] == 'NOT ENOUGH INFO'
#
#         e_set = check_sentences.check_and_clean_evidence(item)
#         additional_data = item['predicted_sentids']
#         # print(len(additional_data))
#         random.shuffle(additional_data)
#         current_sample_num = random.randint(2, 5)
#         raw_evidences_list = []
#         for sampled_e in additional_data[:current_sample_num]:
#             doc_ids = sampled_e.split(c_scorer.SENT_LINE)[0]
#             ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
#             raw_evidences_list.append((doc_ids, ln))
#         new_evidences = check_sentences.Evidences(raw_evidences_list)
#
#         if len(new_evidences) == 0:
#             flag = f"verifiable.eq.0"
#             flags.append(flag)
#             pass
#         else:
#             flag = f"not_verifiable.non_eq.{len(new_evidences)}"
#             flags.append(flag)
#
#         assert all(len(e) == 0 for e in e_set)
#         res_sentids_list.append(new_evidences)
#         assert len(res_sentids_list) == 1
#
#     assert len(res_sentids_list) == len(flags)
#
#     return res_sentids_list, flags


# mode in ['train', 'pred', 'eval']
def sample_data_for_item_extend(item, data_from_pred=False, mode='train'):
    res_sentids_list = []
    extended_RS_list = []
    extended_NEI_list = []
    if data_from_pred:
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

    def get_false_from_same_doc(docid, ground_truth_evi_tuple_set, candidate_sids_tuple_l):
        tmp_tuples = []
        for pred in candidate_sids_tuple_l:
            if pred[0] == docid and pred not in ground_truth_evi_tuple_set:
                tmp_tuples.append(pred)
        return tmp_tuples

    def get_false_from_diff_doc(ground_truth_docids, candidate_sids_tuple_l):
        tmp_tuples = []
        for pred in candidate_sids_tuple_l:
            if pred[0] not in ground_truth_docids:
                tmp_tuples.append(pred)
        return tmp_tuples

    if item['verifiable'] == "VERIFIABLE":
        assert item['label'] == 'SUPPORTS' or item['label'] == 'REFUTES'
        e_set = check_sentences.check_and_clean_evidence(item)
        for evidence in e_set:
            # exact evidence
            res_sentids_list.append(evidence)
            # extend evidence + 2
            if mode == 'train':
                additional_data = item['predicted_sentids']
                predicted_sents_tuples = [
                    (pred_item.split(c_scorer.SENT_LINE)[0], int(pred_item.split(c_scorer.SENT_LINE)[1])) for pred_item
                    in additional_data]
                all_evidence_set = list(set(itertools.chain.from_iterable([evids.evidences_list for evids in e_set])))
                n_e = len(evidence)
                if item['label'] == 'SUPPORTS':
                    extend_number = 2  # >= 2
                elif item['label'] == 'REFUTES':
                    extend_number = 4
                # extend S and R
                same_doc_false_sents = []
                ground_truth_docids = list(set([e[0] for e in all_evidence_set]))
                for doc_id in ground_truth_docids:
                    additional_extend_num = random.randint(1, extend_number)
                    tmp_same_doc_samples = get_false_from_same_doc(doc_id, all_evidence_set, predicted_sents_tuples)
                    len_tmp_same_doc_samples = len(tmp_same_doc_samples)
                    if len_tmp_same_doc_samples == 0:
                        continue
                    while additional_extend_num > 0:
                        extend_s_num = 1 if len_tmp_same_doc_samples < 2 else random.randint(1, len_tmp_same_doc_samples)
                        add_sample = tmp_same_doc_samples[:extend_s_num]
                        if add_sample not in same_doc_false_sents:
                            same_doc_false_sents.append(add_sample)
                            extended_SR_evidences = copy.deepcopy(evidence)
                            extended_SR_evidences.add_sents_tuples(add_sample)
                            extended_RS_list.append(extended_SR_evidences)
                        additional_extend_num -= 1
                        random.shuffle(tmp_same_doc_samples)

                additional_sample_num = random.randint(1, extend_number)
                diff_doc_false_sents = get_false_from_diff_doc(ground_truth_docids, predicted_sents_tuples)
                if len(diff_doc_false_sents) > 0:
                    for i in range(1, additional_sample_num):
                        random.shuffle(diff_doc_false_sents)
                        if len(evidence) < 5:
                            extended_SR_evidences = copy.deepcopy(evidence)
                            extended_SR_evidences.add_sents_tuples(diff_doc_false_sents[:random.randint(1, 3)])
                            extended_RS_list.append(extended_SR_evidences)
                # extend NEI
                for i in range(n_e):
                    #  false from same doc
                    extended_NEI_evidence = copy.deepcopy(evidence)
                    docid, ln = extended_NEI_evidence.pop_sent(i)
                    additional_sample_num = random.randint(0, 5 - n_e) if n_e <= 5 else 2
                    tmp_false_samples = get_false_from_same_doc(docid, all_evidence_set, predicted_sents_tuples)
                    extended_NEI_evidence.add_sents_tuples(tmp_false_samples[:additional_sample_num])
                    extended_NEI_list.append(extended_NEI_evidence)
                    #  false from different doc
                    extended_NEI_evidence = copy.deepcopy(evidence)
                    additional_sample_num = random.randint(0, 5 - n_e) if n_e <= 5 else 2
                    extended_NEI_evidence.pop_sent(i)
                    extended_NEI_evidence.add_sents_tuples(diff_doc_false_sents[:additional_sample_num])
                    extended_NEI_list.append(extended_NEI_evidence)

                # extend more random S and R
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

                # extend more random NEI
                for i in range(n_e):
                    extended_NEI_evidence = copy.deepcopy(evidence)
                    extended_NEI_evidence.pop_sent(i)
                    additional_sample_num = random.randint(2 - n_e, 3 - n_e)
                    for sampled_e in additional_data:
                        if additional_sample_num <= 0 and len(extended_NEI_evidence) > 0 and extended_NEI_evidence not in extended_NEI_list:
                            extended_NEI_list.append(extended_NEI_evidence)
                            break
                        if sample_not_in_evidence_set(sampled_e, e_set):
                            doc_id = sampled_e.split(c_scorer.SENT_LINE)[0]
                            ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
                            extended_NEI_evidence.add_sent(doc_id, ln)
                            additional_sample_num -= 1
        extended_RS_list = list(set(extended_RS_list))
        extended_NEI_list = list(set(extended_NEI_list))
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
def get_sample_data(upstream_data, data_from_pred=False, mode='train'):
    if isinstance(upstream_data, list):
        d_list = upstream_data
    else:
        d_list = read_json_rows(upstream_data)

    sampled_data_list = []

    for item in tqdm(d_list, desc="Sampling"):
        # e_list = check_sentences.check_and_clean_evidence(item)
        sampled_e_list, extended_RS_list, extended_NEI_list = sample_data_for_item_extend(item, data_from_pred=data_from_pred, mode=mode)
        # print(flags)
        for i, sampled_evidence in enumerate(sampled_e_list):
            new_item = dict()
            evidence_text = evidence_list_to_text(sampled_evidence,
                                                  contain_head=True)
            new_item['id'] = str(item['id']) + '#' + str(i)
            new_item['claim'] = item['claim']
            new_item['evid'] = evidence_text
            new_item['sids'] = sampled_evidence.to_sids()
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
                tmp_extend_item = dict()
                tmp_evidence_text = evidence_list_to_text(tmp_extended_evidence, contain_head=True)
                tmp_extend_item['id'] = str(item['id']) + '#' + 'EXTEND'
                tmp_extend_item['claim'] = item['claim']
                tmp_extend_item['evid'] = tmp_evidence_text
                tmp_extend_item['sids'] = tmp_extended_evidence.to_sids()
                return tmp_extend_item

            for idx, extended_evidence in enumerate(extended_RS_list):
                extend_item = init_extended_evidence(extended_evidence)
                extend_item['id'] = extend_item['id'] + '_RS_' + str(idx)
                extend_item['label'] = item['label']
                sampled_data_list.append(extend_item)
            for idx, extended_evidence in enumerate(extended_NEI_list):
                extend_item = init_extended_evidence(extended_evidence)
                extend_item['id'] = extend_item['id'] + '_NEI_' + str(idx)
                extend_item['label'] = 'NOT ENOUGH INFO'
                sampled_data_list.append(extend_item)
    print(f"Sampled evidences: {len(sampled_data_list)}")
    return sampled_data_list


def evidence_list_to_text(evidences, contain_head=True):
    current_evidence_text = []
    evidences = sorted(evidences, key=lambda x: (x[0], x[1]))
    cur_head = 'DO NOT INCLUDE THIS FLAG'
    db = FeverDBResource()
    cursor = db.get_cursor()
    for doc_id, line_num in evidences:
        _, e_text, _ = fever_db.get_evidence(cursor, doc_id, line_num)
        if contain_head and cur_head != doc_id:
            cur_head = doc_id
            # doc_id_natural_format = text_clean.convert_brc(doc_id).replace('_', ' ')
            # t_doc_id_natural_format = ' '.join(easy_tokenize(doc_id_natural_format))
            t_doc_id_natural_format = text_clean.convert_brc(text_clean.normalize(doc_id)).replace('_', ' ')
            if line_num != 0:
                current_evidence_text.append(f"{t_doc_id_natural_format}{c_scorer.SENT_DOC_TITLE}")
        current_evidence_text.append(e_text)
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


def create_train_pred(p1,p2, origin_d):
    new_items = []
    len_ori = len(origin_d)
    print(len(p1))
    print(len(p2))
    print(len(origin_d))
    assert (len(p1) == len(p2))
    while len(origin_d) > 0:
        item = origin_d.pop(0)
        item1 = p1.pop(0)
        item2 = p2.pop(0)
        assert (item['id'] == item1['id'])
        assert(item1['id'] == item2['id'])
        tmp_pred = item1['predicted_sentids']
        tmp_pred.extend(item2['predicted_sentids'])
        tmp_pred = list(set(tmp_pred))
        item['predicted_sentids'] = tmp_pred
        new_items.append(item)
    print(len(new_items))
    print(len(origin_d))
    assert(len(new_items) == len_ori)
    save_intermidiate_results(new_items, config.RESULT_PATH / 'train_2021/train_ss.jsonl')



if __name__ == '__main__':
    # tmp1 = read_json_rows(config.RESULT_PATH / 'train_2021/bert_ss_0.01_10_80000.jsonl')
    # tmp1.extend(read_json_rows(config.RESULT_PATH / 'train_2021/bert_ss_0.01_10.jsonl'))
    # tmp2 = read_json_rows(config.RESULT_PATH / 'tfidf/train_2019_06_15_15:48:58.jsonl')
    # ori = read_json_rows(config.FEVER_TRAIN_JSONL)
    # create_train_pred(tmp1, tmp2, ori)
    additional_file = read_json_rows(config.RESULT_PATH / 'train_2021/bert_ss_0.01_10_80000.jsonl')
    # additional_file = read_json_rows(config.RESULT_PATH / 'bert_ss_dev_10/eval_data_ss_10_dev.jsonl')
    t = get_sample_data(additional_file, data_from_pred=False, mode='train')
    eval_samples(t)

    # t = get_sample_data(additional_file, data_from_pred=False, mode='eval')
    # eval_samples(t)

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
