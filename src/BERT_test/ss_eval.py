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

import numpy as np
import torch
from sklearn.utils.extmath import softmax
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm

from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score

from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

from utils.file_loader import save_jsonl, read_json_rows, get_current_time_str, save_file, save_intermidiate_results
from typing import Dict
from utils import c_scorer
from BERT_test.bert_data_processor import *
from sample_for_nli.tf_idf_sample_v1_0 import convert_evidence2scoring_format


logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return {"acc": acc_and_f1(preds, labels)}


def eval_ss_and_save(saved_model, saved_tokenizer_model, upstream_data, pred=False, mode='dev'):
    model = BertForSequenceClassification.from_pretrained(saved_model, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(saved_tokenizer_model, do_lower_case=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    processor = FeverSSProcessor()
    eval_batch_size = 8

    if mode == 'dev':
        eval_examples, dev_list = processor.get_dev_examples(upstream_data, pred=pred)
    else:
        eval_examples, dev_list = processor.get_train_examples(upstream_data, pred=pred)

    eval_features = convert_examples_to_features(
        eval_examples, processor.get_labels(), 128, tokenizer)

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


        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]

    # probs_test = softmax_test(preds)
    probs = softmax(preds)
    probs = probs[:, 0].tolist()
    scores = preds[:, 0].tolist()
    preds = np.argmax(preds, axis=1)

    for i in range(len(dev_list)):
        assert str(eval_examples[i].guid) == str(dev_list[i]['selection_id'])
        # Matching id
        dev_list[i]['score'] = scores[i]
        dev_list[i]['prob'] = probs[i]

    # fever score and saving
    result = compute_metrics("ss", preds, all_label_ids.numpy())
    loss = None

    result['eval_loss'] = eval_loss
    result['loss'] = loss

    logger.info("***** Eval results *****")
    pred_log = ''
    for key in sorted(result.keys()):
        pred_log = pred_log + key + ":" + str(result[key]) + "\n"
        logger.info("  %s = %s", key, str(result[key]))

    save_file(pred_log, config.LOG_PATH / f"{get_current_time_str()}_ss_pred.log")

    orginal_file = config.FEVER_DEV_JSONL if mode == 'dev' else config.FEVER_TRAIN_JSONL
    original_list = read_json_rows(orginal_file)
    ss_f1_score_and_save(original_list, dev_list)


def ss_score_converter(original_list, upsteam_eval_list, prob_threshold, top_n=5):
    d_list = original_list
    augmented_dict: Dict[int, Dict[str, Dict]] = dict()
    print("Build selected sentences file:", len(upsteam_eval_list))
    for sent_item in tqdm(upsteam_eval_list):
        selection_id = sent_item['selection_id']  # The id for the current one selection.
        org_id = int(selection_id.split('<##>')[0])
        remain_str = selection_id.split('<##>')[1]
        if org_id in augmented_dict:
            if remain_str not in augmented_dict[org_id]:
                augmented_dict[org_id][remain_str] = sent_item
            else:
                print("Exist")
        else:
            augmented_dict[org_id] = {remain_str: sent_item}

    for item in d_list:
        if int(item['id']) not in augmented_dict:
            # print("Potential error?")
            cur_predicted_sentids = []
        else:
            cur_predicted_sentids = []  # formating doc_id + c_score.SENTLINT + line_number
            sents = augmented_dict[int(item['id'])].values()
            # Modify some mechaism here to selection sentence whether by some score or label
            for sent_i in sents:
                if sent_i['prob'] >= prob_threshold:
                    cur_predicted_sentids.append((sent_i['sid'], sent_i['score'],
                                                  sent_i['prob']))  # Important sentences for scaling training. Jul 21.
                # del sent_i['prob']

            cur_predicted_sentids = sorted(cur_predicted_sentids, key=lambda x: -x[1])

        item['scored_sentids'] = cur_predicted_sentids[:top_n]  # Important sentences for scaling training. Jul 21.
        item['predicted_sentids'] = [sid for sid, _, _ in item['scored_sentids']][:top_n]
        item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])
        # item['predicted_label'] = item['label']  # give ground truth label

    return d_list


def ss_f1_score_and_save(actual_list, upstream_eval_list, prob_thresholds=0.5, top_n = 5, save_data=True, mode = 'dev'):
    if not isinstance(prob_thresholds, list):
        prob_thresholds = [prob_thresholds]

    for scal_prob in prob_thresholds:
        print("Eval Dev Data prob_threshold:", scal_prob)

        results_list = ss_score_converter(actual_list, upstream_eval_list,
                                              prob_threshold=scal_prob, top_n = top_n)

        eval_mode = {'check_sent_id_correct': True, 'standard': False}
        # for a, b in zip(actual_list, results_list):
        #     b['predicted_label'] = a['label']
        strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(results_list,
                                                                    actual_list,
                                                                    mode=eval_mode, verbose=False)
        tracking_score = strict_score
        print(f"Dev(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}/")
        print("Strict score:", strict_score)
        print(f"Eval Tracking score:", f"{tracking_score}")

    if save_data:
        time = get_current_time_str()
        output_eval_file = config.RESULT_PATH / "bert_finetuning" / time / f"ss_eval_{mode}.txt"
        output_items_file = config.RESULT_PATH / "bert_finetuning" / time / f"ss_items_{mode}.jsonl"
        output_ss_file = config.RESULT_PATH / "bert_finetuning" / time / f"ss_scores_{mode}.txt"
        save_intermidiate_results(upstream_eval_list, output_ss_file)
        save_jsonl(actual_list, output_items_file)
        save_file(f"{mode}:(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}/ \nStrict score:{strict_score}", output_eval_file)


def softmax_test(z):
    """Compute softmax values for each sets of scores in x."""
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


if __name__ == "__main__":
    # pass
    # eval_ss_and_save(config.PRO_ROOT / "saved_models/bert/bert-large-uncased.tar.gz", "bert-large-uncased")
    eval_ss_and_save(config.PRO_ROOT / "saved_models/bert_finetuning/2019_06_18_11:10:41",
                     config.PRO_ROOT / "saved_models/bert_finetuning/2019_06_18_11:10:41",
                     config.RESULT_PATH / "dev_s_tfidf_retrieve.jsonl")
