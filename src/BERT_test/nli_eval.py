# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

from typing import Dict, List
import torch
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from sklearn.utils.extmath import softmax
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
import numpy as np
from BERT_test.bert_data_processor import *
from BERT_test.eval_util import compute_metrics
from data_util.toChart import *
from utils import c_scorer
from utils.file_loader import read_json_rows, save_file, save_intermidiate_results

logger = logging.getLogger(__name__)


def nli_eval_fever_score(paras: bert_para.PipelineParas, predicted_list):
    if paras.mode == 'eval':
        eval_mode = {'check_sent_id_correct': True, 'standard': True}
        strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(predicted_list,
                                                                paras.original_data,
                                                                mode=eval_mode,
                                                                error_analysis_file=paras.get_f1_log_file('nli'),
                                                                verbose=False)
        tracking_score = strict_score
        print(f"Dev(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}/")
        print("Strict score:", strict_score)
        print(f"Eval Tracking score:", f"{tracking_score}")
        save_intermidiate_results(predicted_list, paras.get_eval_result_file('nli'))
    else:
        delete_unused_evidence(predicted_list)
        clean_result = []
        for i in predicted_list:
            clean_item = {'id': i['id'],
                          'claim': i['claim'],
                          'predicted_evidence': i['predicted_evidence'],
                          'predicted_label': i['predicted_label']}
            clean_result.append(clean_item)
        save_intermidiate_results(clean_result, paras.get_eval_result_file('nli'))


def eval_nli_examples(paras : bert_para.PipelineParas):
    model = BertForSequenceClassification.from_pretrained(paras.BERT_model, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(paras.BERT_tokenizer, do_lower_case=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    eval_batch_size = 8
    sequence_length = 300
    processor = FeverNliProcessor()
    # sampler = 'nli_nn'

    if paras.mode == 'eval':
        eval_examples, eval_list = processor.get_dev_examples(paras.upstream_data,
                                                              data_from_pred=paras.data_from_pred,
                                                              sampler=paras.sampler)
    else:
        eval_examples, eval_list = processor.get_test_examples(paras.upstream_data, paras.sampler)

    eval_features = convert_examples_to_features(
        eval_examples, processor.get_labels(), sequence_length, tokenizer)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    num_labels = len(processor.get_labels())
    loss_for_chart = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        # create eval loss and other metric required by the task
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        loss_for_chart.append(tmp_eval_loss.mean().item())
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    if paras.mode == 'eval':
        draw_loss_epoch_detailed(np.array(loss_for_chart).reshape(1, len(loss_for_chart)), f"nli_eval_{paras.output_folder}")
        eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    probs = softmax(preds)
    probs = probs[:, 0].tolist()
    scores = preds[:, 0].tolist()
    preds = np.argmax(preds, axis=1)

    for i in range(len(eval_list)):
        # Matching id
        eval_list[i]['score'] = scores[i]
        eval_list[i]['prob'] = probs[i]
        eval_list[i]['predicted_label'] = preds[i]
    # fever score and saving
    result = compute_metrics(preds, all_label_ids.numpy(), average='macro')
    result['eval_loss'] = eval_loss

    logger.info("***** Eval results *****")
    if paras.mode == 'eval':
        pred_log = ''
        for key in sorted(result.keys()):
            pred_log = pred_log + key + ":" + str(result[key]) + "\n"
            logger.info("  %s = %s", key, str(result[key]))
            print(key, str(result[key]))
        save_file(pred_log, paras.get_eval_log_file(paras.sampler))

    # not for training, but for test set predict
    # if the item has multiple evidence set
    save_intermidiate_results(eval_list, paras.get_eval_item_file('nli_items'))
    print("Done with nli item eval")
    return eval_list


def eval_nli_and_save(paras : bert_para.PipelineParas):
    eval_list = eval_nli_examples(paras)
    post_step_nli_eval(eval_list, paras)


def post_step_nli_eval(eval_list, paras: bert_para.PipelineParas):
    id2label = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT ENOUGH INFO"
    }
    augmented_dict: Dict[int, Dict[str, str]] = dict()
    for evids_item in tqdm(eval_list):
        evids_id = evids_item['id']  # The id for the current one selection.
        org_id = int(evids_id.split('#')[0])
        # remain_index = evids_id.split('#')[1]
        evids_item['id'] = org_id
        # assert remain_index == 0
        if not org_id in augmented_dict:
            aug_i = {'predicted_label': str(evids_item["predicted_label"]),
                     'predicted_evidence': evids_item["predicted_evidence"]}
            augmented_dict[org_id] = aug_i

    for item in paras.upstream_data:
        if int(item['id']) not in augmented_dict:
            print("not found this example:\n", item)
        else:
            item["predicted_label"] = id2label[int(augmented_dict[int(item['id'])]['predicted_label'])]
            item["predicted_evidence"] = augmented_dict[int(item['id'])]["predicted_evidence"]
    nli_eval_fever_score(paras, paras.upstream_data)
    print("Done with nli evaluation")


def nli_pred_evi_score_only(paras : bert_para.PipelineParas):
    eval_list = eval_nli_examples(paras)
    nli_evi_set_post_step(eval_list, paras)


def nli_evi_set_post_step(eval_examples, eval_list, paras: bert_para.PipelineParas):
    augmented_dict: Dict[int, List[Dict]] = dict()
    with tqdm(total=len(eval_list), desc=f"nli evi post step...") as pbar:
        for idx, evids_item in enumerate(eval_list):
            example = eval_examples[idx]
            evids_id = evids_item['id']  # The id for the current one example.
            org_id = int(evids_id.split('#')[0])
            # remain_index = evids_id.split('#')[1]
            evids_item['id'] = org_id
            aug_i = {'example_idx': evids_id, 'predicted_label': str(evids_item["predicted_label"]),
                     'score': evids_item["score"], 'sids': example['sids']}
            if not org_id in augmented_dict:
                augmented_dict.update({org_id: [aug_i]})
            else:
                augmented_dict[org_id].append(aug_i)
            pbar.update(1)

    for item in paras.upstream_data:
        if int(item['id']) not in augmented_dict:
            print("not found this example:\n", item)
        else:
            item["evi_nli"] = augmented_dict[int(item['id'])]
    save_intermidiate_results(paras.upstream_data, paras.output_folder / "sids_nli.jsonl")
    print("Done with nli evaluation")


def delete_unused_evidence(d_list):
    for item in d_list:
        if item['predicted_label'] == 'NOT ENOUGH INFO':
            item['predicted_evidence'] = []


if __name__ == "__main__":
    # paras = bert_para.BERT_para()
    # paras.original_data = read_json_rows(config.FEVER_DEV_JSONL)[0:3]
    # paras.upstream_data = read_json_rows(config.RESULT_PATH / "dev_s_tfidf_retrieve.jsonl")[0:10]
    # paras.pred = True
    # paras.mode = 'dev'
    # paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_test_refactor"
    # paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_test_refactor"
    # paras.output_folder = "nli_eval"
    #
    # eval_nli_and_save(paras)

    # paras = bert_para.PipelineParas()
    # paras.pred = True
    # paras.mode = 'eval'
    # paras.original_data = read_json_rows(config.FEVER_DEV_JSONL)
    # paras.upstream_data = read_json_rows(config.RESULT_PATH / "test_ss_full/eval_data_ss_test_0.5_top5.jsonl")[0:50]
    #paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_test_refactor"
    #paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_test_refactor"
    # paras.upstream_data = read_json_rows(config.RESULT_PATH / "dev_pred_ss_2019_07_31_05:56:24/eval_data_ss_dev_0.5_top5.jsonl")[0:50]
    # paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_nli_train2019_07_15_16:51:03"
    # paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_nli_train2019_07_15_16:51:03"
    # paras.output_folder = 'before_refactor'
    # eval_nli_and_save(paras)

    paras = bert_para.PipelineParas()
    # paras.pred = True
    paras.mode = 'eval'
    paras.original_data = read_json_rows(config.FEVER_DEV_JSONL)
    paras.upstream_data = read_json_rows(config.RESULT_PATH / "bert_gat_merged_ss_dev_5/eval_data_bert_gat_ss_5_dev_0.1_top[5].jsonl")
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_nli_train2019_07_15_16:51:03"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_nli_train2019_07_15_16:51:03"
    paras.output_folder = 'dev_before_refactor'
    eval_nli_and_save(paras)


    # paras = bert_para.PipelineParas()
    # paras.pred = True
    # paras.mode = 'pred'
    # paras.original_data = read_json_rows(config.FEVER_TEST_JSONL)
    # paras.upstream_data = read_json_rows(config.RESULT_PATH / "bert_gat_merged_ss_test_5/eval_data_bert_gat_ss_5_test_0.1_top[5].jsonl")
    # paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_nli_train2019_07_15_16:51:03"
    # paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_nli_train2019_07_15_16:51:03"
    # paras.output_folder = 'nli_test_bert_gat'
    # eval_nli_and_save(paras)



