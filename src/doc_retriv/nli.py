from tqdm import tqdm
import config
from BERT_test.nli_eval import nli_pred_evi_score_only, eval_nli_examples
import utils.common_types as bert_para
from collections import Counter
from utils.file_loader import read_json_rows
from utils.check_sentences import Evidences, sids_to_doclnlist
from functools import reduce
from utils.c_scorer import get_macro_ss_recall_precision
import numpy as np

id2label = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT ENOUGH INFO"
}

def nli_vote(data_nli_with_score):
    hits = 0
    with tqdm(total=len(data_nli_with_score), desc=f"searching triple sentences") as pbar:
        for idx, example in enumerate(data_nli_with_score):
            if 'evi_nli' not in example:
                label = 2
                print(idx)
            else:
                preds = example['evi_nli']
                pred_labels = [p['predicted_label'] for p in preds]
                count = Counter()
                count.update(pred_labels)
                label_count = sorted(list(count.most_common()), key=lambda x: x[0])
                label_dict = {i[0]: i[1] for i in label_count}
                sc = label_dict['0'] if '0' in label_dict else 0
                rc = label_dict['1'] if '1' in label_dict else 0
                nei = label_dict['2'] if '2' in label_dict else 0
                if sc > 0 or rc > 0:
                    label = 0 if sc > rc else 1
                else:
                    label = 2
            pre_label = id2label[label]
            if pre_label == example['label']:
                hits += 1
            else:
                print(f"idx: {idx}, label: {example['label']}, sc: {sc}, rc: {rc}, nei:{nei}")
            pbar.update(1)
    print(hits / len(data_nli_with_score))


def vote_and_filter(data_nli_with_score, eval=True):
    filtered = []
    for i in tqdm(data_nli_with_score):
        filtered_example = vote_label_and_filter_example(i)
        filtered.append(filtered_example)
    if eval:
        get_macro_ss_recall_precision(filtered)


def vote_label_and_filter_example(example):
    if 'evi_nli' not in example:
        label = 2
    else:
        preds = example['evi_nli']
        pred_labels = [p['predicted_label'] for p in preds]
        count = Counter()
        count.update(pred_labels)
        label_count = sorted(list(count.most_common()), key=lambda x: x[0])
        label_dict = {int(i[0]): i[1] for i in label_count}
        sc = label_dict[0] if 0 in label_dict else 0
        rc = label_dict[1] if 1 in label_dict else 0
        # nei = label_dict[2] if 2 in label_dict else 0
        if sc > 0 or rc > 0:
            label = 0 if sc > rc else 1
        else:
            label = 2
    if label == 2:
        clean_item = {'id': example['id'],
                      'claim': example['claim'],
                      'predicted_evidence': [],
                      'predicted_label': id2label[label]}
    else:
        # {'example_idx': evids_id, 'predicted_label': str(evids_item["predicted_label"]),
        #  'score': evids_item["score"], 'prob': evids_item["prob"], 'sids': evids_item['sids']}
        all_evi = [p for p in preds if p['predicted_label'] == str(label)]
        all_sids = list(set([s for evi in all_evi for s in evi['sids']]))
        sid2weightedprob = {s : float(0) for s in all_sids}
        for evi in all_evi:
            sids = evi['sids']
            prob = evi['prob']
            weighted = prob / len(sids)
            for s in sids:
                sid2weightedprob[s] += weighted
        sids_scores = list(zip(sid2weightedprob.keys(), sid2weightedprob.values()))
        sids_scores.sort(key=lambda k: k[1], reverse=True)
        predicted_sids = sids_scores[:5]
        predicted_sids = [s[0] for s in predicted_sids]
        predicted_sids = sids_to_doclnlist(predicted_sids)
        clean_item = {'id': example['id'],
                      'claim': example['claim'],
                      'predicted_evidence': predicted_sids,
                      'predicted_label': id2label[label]}
    if 'label' in example:
        clean_item.update({'label': example['label'], 'evidence': example['evidence']})
    return clean_item


def nli_pred_evi_set(upstream_data, output_folder):
    paras = bert_para.PipelineParas()
    paras.mode = 'eval'
    paras.data_from_pred = True
    paras.upstream_data = upstream_data
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_train_81.4"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_train_81.4"
    # paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_nli_first_train2019"
    # paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_nli_first_train2019"
    paras.output_folder = output_folder
    paras.sampler = 'nli_evis'
    nli_pred_evi_score_only(paras)
    data_nli = read_json_rows(folder / "sids_nli_pred.jsonl")
    vote_and_filter(data_nli)


def nli_eval_vote(upstream_data, output_folder):
    paras = bert_para.PipelineParas()
    paras.mode = 'eval'
    paras.data_from_pred = False
    paras.upstream_data = upstream_data
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_train_81.4"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_train_81.4"
    paras.output_folder = output_folder
    paras.sampler = 'nli_evis'
    nli_pred_evi_score_only(paras)
    data_nli = read_json_rows(folder / "sids_nli_pred.jsonl")
    nli_vote(data_nli)


def nli_eval_top_rank(upstream_data, output_folder):
    paras = bert_para.PipelineParas()
    paras.mode = 'eval'
    paras.data_from_pred = False
    paras.upstream_data = upstream_data
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_train_81.4"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_train_81.4"
    paras.output_folder = output_folder
    paras.sampler = 'nli_nn'
    eval_nli_examples(paras)


def eval_samples(upstream_data):
    count = Counter()
    sid_count = [len(i['nli_sids']) for i in upstream_data]
    count.update(sid_count)
    print(f"most_common: {sorted(list(count.most_common()), key=lambda x: -x[0])}")
    print(f"max_length: {np.max(sid_count)}")
    print(f"mean: {np.mean(sid_count)}")
    print(f"std: {np.std(sid_count)}")
    too_many = [idx for idx, i in enumerate(sid_count) if i > 500]
    print(too_many)


if __name__ == '__main__':
    # t = [1,1,1,1,2,1,0,1,2,0,0,0]
    # count = Counter()
    # count.update(t)
    # print(count.most_common())
    # print(sorted(list(count.most_common()), key=lambda x: x[0]))
    folder = config.RESULT_PATH / "hardset2021"
    # data_bert = read_json_rows(folder / "bert_ss_0.4_10.jsonl")
    # nli_eval1(data_bert, folder)
    # nli_eval2(data_bert, folder)
    data_nli_sids = read_json_rows(folder / "nli_sids.jsonl")
    # nli_eval_top_rank(data_nli_sids, folder)
    # eval_samples(data_nli_sids)
    nli_pred_evi_set(data_nli_sids, folder)
    # data_nli = read_json_rows(folder / "sids_nli_pred.jsonl")
    # vote_and_filter(data_nli)
