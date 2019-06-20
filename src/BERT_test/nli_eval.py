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
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
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


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {"acc": acc_and_f1(preds, labels)}


def nli_eval_score(actual_list, upstream_eval_list, save_data=True, mode='dev'):
    acc, f1 , acc_and_f1 = compute_metrics(actual_list, upstream_eval_list)
    print(f"{mode}:(acc/f1/acc_and_f1):{acc}/{f1}/{acc_and_f1}")

    if save_data:
        time = get_current_time_str()
        output_nli_file = config.RESULT_PATH / "bert_finetuning" / time / f"nli_eval_{mode}.txt"
        save_file(f"{mode}:(acc/f1/acc_and_f1):{acc}/{f1}/{acc_and_f1}", output_nli_file)


def nli_eval_fever_score(predicted_list, mode='dev'):
    if mode == 'dev':
        actual_list = read_json_rows(config.FEVER_DEV_JSONL)
        eval_mode = {'check_sent_id_correct': False, 'standard': True}
    else:
        actual_list = read_json_rows(config.FEVER_TEST_JSONL)
        eval_mode = {'check_sent_id_correct': False, 'standard': True}

    strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(predicted_list,
                                                                actual_list,
                                                                mode=eval_mode, verbose=False)
    tracking_score = strict_score
    print(f"Dev(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}/")
    print("Strict score:", strict_score)
    print(f"Eval Tracking score:", f"{tracking_score}")

    time = get_current_time_str()
    output_eval_file = config.RESULT_PATH / "bert_finetuning" / time / f"ss_eval_{mode}.txt"
    output_items_file = config.RESULT_PATH / "bert_finetuning" / time / f"ss_items_{mode}.jsonl"
    output_ss_file = config.RESULT_PATH / "bert_finetuning" / time / f"ss_scores_{mode}.txt"
    save_intermidiate_results(predicted_list, output_ss_file)
    save_jsonl(actual_list, output_items_file)
    save_file(f"{mode}:(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}/ \nStrict score:{strict_score}",
              output_eval_file)


def eval_nli_and_save(saved_model, saved_tokenizer_model, upstream_data, pred=False, mode='dev'):
    model = BertForSequenceClassification.from_pretrained(saved_model, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(saved_tokenizer_model, do_lower_case=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    eval_batch_size = 8
    sequence_length = 300
    processor = FeverNliProcessor()

    if mode == 'dev':
        eval_examples, eval_list = processor.get_dev_examples(upstream_data, pred=pred)
    else:
        eval_examples, eval_list = processor.get_train_examples(upstream_data, pred=pred)

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

    probs = softmax(preds)
    probs = probs[:, 0].tolist()
    scores = preds[:, 0].tolist()
    preds = np.argmax(preds, axis=1)

    for i in range(len()):
        # Matching id
        eval_list[i]['score'] = scores[i]
        eval_list[i]['prob'] = probs[i]
        eval_list[i]['predicted_label'] = preds
    # fever score and saving
    result = compute_metrics("nli", preds, all_label_ids.numpy())
    loss = None

    result['eval_loss'] = eval_loss
    result['loss'] = loss

    logger.info("***** Eval results *****")
    pred_log = ''
    for key in sorted(result.keys()):
        pred_log = pred_log + key + ":" + str(result[key]) + "\n"
        logger.info("  %s = %s", key, str(result[key]))

    save_file(pred_log, config.LOG_PATH / f"{get_current_time_str()}_nli_pred.log")

    nli_eval_score(eval_list, eval_list)

    # not for training, but for test set predict
    if pred:
        augmented_dict: Dict[int, str] = dict()
        for evids_item in tqdm(eval_list):
            evids_id = evids_item['id']  # The id for the current one selection.
            org_id = int(evids_id.split('<#>')[0])
            remain_index = evids_id.split('<#>')[1]
            #assert remain_index == 0
            if org_id in augmented_dict:
                augmented_dict[org_id] = evids_item["predicted_label"]
            else:
                print("Exist:", evids_item)


         #todo:verify Dict len
        for item in upstream_data:
            if int(item['id']) not in augmented_dict:
                print("not found this example:\n", item)
            else:
                item["predicted_label"] = augmented_dict[int(item['id'])]
        nli_eval_fever_score(upstream_data)


if __name__ == "__main__":
    pass
    # eval_ss_and_save(config.PRO_ROOT / "saved_models/bert/bert-large-uncased.tar.gz", "bert-large-uncased")
    # eval_ss_and_save(config.PRO_ROOT / "saved_models/bert_finetuning/2019_06_13_17:07:55",
    #                  config.PRO_ROOT / "saved_models/bert_finetuning/2019_06_13_17:07:55")
