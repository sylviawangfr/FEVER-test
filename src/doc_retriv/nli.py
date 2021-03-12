from tqdm import tqdm
import config
from BERT_test.nli_eval import nli_pred_evi_score_only, eval_nli_examples
import utils.common_types as bert_para
from collections import Counter
from utils.file_loader import read_json_rows
from functools import reduce


def nli_vote(data_nli_with_score):
    id2label = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT ENOUGH INFO"
    }
    hits = 0
    with tqdm(total=len(data_nli_with_score), desc=f"searching triple sentences") as pbar:
        for idx, example in enumerate(data_nli_with_score):
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


def nli_eval1(upstream_data, output_folder):
    paras = bert_para.PipelineParas()
    paras.mode = 'eval'
    paras.data_from_pred = False
    paras.upstream_data = upstream_data
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_train_86.7"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_train_86.7"
    paras.output_folder = output_folder
    paras.sampler = 'nli_evis'
    nli_pred_evi_score_only(paras)
    data_nli = read_json_rows(folder / "sids_nli.jsonl")
    nli_vote(data_nli)


def nli_eval2(upstream_data, output_folder):
    paras = bert_para.PipelineParas()
    paras.mode = 'eval'
    paras.data_from_pred = False
    paras.upstream_data = upstream_data
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_train_86.7"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_train_86.7"
    paras.output_folder = output_folder
    paras.sampler = 'nli_nn'
    eval_nli_examples(paras)


if __name__ == '__main__':
    # t = [1,1,1,1,2,1,0,1,2,0,0,0]
    # count = Counter()
    # count.update(t)
    # print(count.most_common())
    # print(sorted(list(count.most_common()), key=lambda x: x[0]))
    folder = config.RESULT_PATH / "hardset2021"
    data_bert = read_json_rows(folder / "bert_ss_0.4_10.jsonl")
    # nli_eval1(data_bert, folder)
    nli_eval2(data_bert, folder)

    # data_nli = read_json_rows(folder / "sids_nli.jsonl")
    # nli_vote(data_nli)
