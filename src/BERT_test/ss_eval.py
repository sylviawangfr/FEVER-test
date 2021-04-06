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

from typing import Dict

import torch
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from sklearn.utils.extmath import softmax
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm

import utils.common_types as bert_para
from BERT_test.bert_data_processor import *
from BERT_test.eval_util import compute_metrics
from data_util.toChart import *
from sample_for_nli_esim.tf_idf_sample_v1_0 import convert_evidence2scoring_format
from utils import c_scorer
from utils.file_loader import read_json_rows, save_file, save_intermidiate_results
import numpy as np

logger = logging.getLogger(__name__)


def eval_ss_and_save(paras : bert_para.PipelineParas):
    model = BertForSequenceClassification.from_pretrained(paras.BERT_model, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(paras.BERT_tokenizer, do_lower_case=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    processor = FeverSSProcessor()
    eval_batch_size = 8
    eval_examples, eval_list = processor.get_dev_examples(paras)
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

    draw_loss_epoch_detailed(np.array(loss_for_chart).reshape(1, len(loss_for_chart)), 'ss_eval_loss' + get_current_time_str())
    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]

    # probs_test = softmax_test(preds)
    probs = softmax(preds)
    probs = probs[:, 0].tolist()
    # scores = preds[:, 0].tolist()
    preds = np.argmax(preds, axis=1)

    for i in range(len(eval_list)):
        assert str(eval_examples[i].guid) == str(eval_list[i]['selection_id'])
        # Matching id
        # eval_list[i]['score'] = scores[i]
        eval_list[i]['prob'] = probs[i]

    # fever score and saving
    result = compute_metrics(preds, all_label_ids.numpy())

    result['eval_loss'] = eval_loss

    logger.info("***** Eval results *****")
    if paras.mode == 'eval':
        pred_log = ''
        for key in sorted(result.keys()):
            pred_log = pred_log + key + ":" + str(result[key]) + "\n"
            logger.info("  %s = %s", key, str(result[key]))
            print(f"{key}:{result[key]}")
        save_file(pred_log, paras.get_eval_log_file('ss'))

    ss_f1_score_and_save(paras, eval_list)


def pred_ss_and_save(paras : bert_para.PipelineParas):
    model = BertForSequenceClassification.from_pretrained(paras.BERT_model, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(paras.BERT_tokenizer, do_lower_case=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    processor = FeverSSProcessor()
    eval_batch_size = 32
    if paras.mode == 'test':
        eval_examples, eval_list = processor.get_test_examples(paras, sampler='ss_full')
    elif paras.mode == 'eval':
        eval_examples, eval_list = processor.get_dev_examples(paras, sampler='ss_full')
    else:
        eval_examples, eval_list = processor.get_train_examples(paras, sampler='ss_full')

    eval_features = convert_examples_to_features(
        eval_examples, processor.get_labels(), 128, tokenizer)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    if not paras.data_from_pred:
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    else:
        all_label_ids = torch.tensor([-1] * len(eval_features), dtype=torch.long)

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
        # label_ids = label_ids.to(device)

        with torch.no_grad():
            # if `labels` is `None`:
            #    Outputs the classification logits of shape [batch_size, num_labels].
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        loss_fct = CrossEntropyLoss()
        # tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        # eval_loss += tmp_eval_loss.mean().item()
        # nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    # eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    probs = softmax(preds)
    probs = probs[:, 0].tolist()
    scores = preds[:, 0].tolist()
    # preds = np.argmax(preds, axis=1)
    # if paras.mode == 'eval':
    #     logger.info("***** Eval results *****")
    #     result = compute_metrics(preds, all_label_ids.numpy())
    #     result['eval_loss'] = eval_loss
    #     pred_log = ''
    #     for key in sorted(result.keys()):
    #         pred_log = pred_log + key + ":" + str(result[key]) + "\n"
    #         logger.info("  %s = %s", key, str(result[key]))
    #         print(f"{key}:{result[key]}")
    #     save_file(pred_log, paras.get_eval_log_file('pred_ss'))

    for i in range(len(eval_list)):
        assert str(eval_examples[i].guid) == str(eval_list[i]['selection_id'])
        # Matching id
        eval_list[i]['score'] = float(scores[i])
        eval_list[i]['prob'] = float(probs[i])

    # results_list = ss_score_converter(paras.original_data, eval_list, paras.prob_thresholds, paras.top_n)
    return ss_f1_score_and_save(paras, eval_list)


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
                if 'prob' in sent_i and sent_i['prob'] >= prob_threshold:
                    cur_predicted_sentids.append((sent_i['sid'],  # sent_i['score'],
                                                  sent_i['prob']))  # Important sentences for scaling training. Jul 21.
                # del sent_i['prob']

            cur_predicted_sentids = sorted(cur_predicted_sentids, key=lambda x: -x[2])

        item['scored_sentids'] = cur_predicted_sentids[:top_n]  # Important sentences for scaling training. Jul 21.
        item['predicted_sentids'] = [sid for sid, _ in item['scored_sentids']][:top_n]
        item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])
        # item['predicted_label'] = item['label']  # give ground truth label
    return d_list


def ss_f1_score_and_save(paras: bert_para.PipelineParas, upstream_eval_list, save_data=True):

    if not isinstance(paras.prob_thresholds, list):
        prob_thresholds = [paras.prob_thresholds]
    else:
        prob_thresholds = paras.prob_thresholds

    if not isinstance(paras.top_n, list):
        top_n = [paras.top_n]
    else:
        top_n = paras.top_n

    for scal_prob in prob_thresholds:
        print("Eval Data prob_threshold:", scal_prob)

        for n in top_n:
            print(f"max evidence number:", n)
            results_list = ss_score_converter(paras.original_data, upstream_eval_list,
                                          prob_threshold=scal_prob,
                                          top_n=n)
            if paras.mode == 'eval':
                eval_mode = {'check_sent_id_correct': True, 'standard': False}
                strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(results_list,
                                                                    paras.original_data,
                                                                    max_evidence=n,
                                                                    mode=eval_mode,
                                                                    error_analysis_file=paras.get_f1_log_file(f'{scal_prob}_{n}_ss'),
                                                                    verbose=False)
                tracking_score = strict_score
                print(f"Dev(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}/")
                print("Strict score:", strict_score)
                print(f"Eval Tracking score:", f"{tracking_score}")

            if save_data:
                save_intermidiate_results(results_list, paras.get_eval_result_file(f'bert_ss_{scal_prob}_{n}'))
                save_intermidiate_results(upstream_eval_list, paras.get_eval_item_file(f'bert_ss_{scal_prob}_{n}'))
                print(f"results saved at: {paras.output_folder}")
    return results_list

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
    # paras = bert_para.PipelineParas()
    # paras.original_data = read_json_rows(config.FEVER_DEV_JSONL)[2:5]
    # paras.upstream_data = read_json_rows(config.RESULT_PATH / "dev_s_tfidf_retrieve.jsonl")[2:5]
    # paras.pred = False
    # paras.mode = 'dev'
    # paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_3s_full2019_07_17_04:00:55"
    # paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_3s_full2019_07_17_04:00:55"
    # paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_202101_92.9"
    # paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_202101_92.9"
    # paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_train_2021_4"
    # paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_train_2021_4"
    # paras.output_folder = config.LOG_PATH / "bert_ss_" + get_current_time_str()
    # paras.original_data = read_json_rows(config.FEVER_DEV_JSONL)
    # paras.upstream_data = read_json_rows(config.RESULT_PATH / "dev_s_tfidf_retrieve.jsonl")
    # paras.mode = 'eval'
    # paras.data_from_pred = True
    # paras.sample_n = 10
    # paras.post_filter_prob = 0.5
    # paras.top_n = [10, 5]
    # paras.prob_thresholds = [0.4, 0.5]
    # pred_ss_and_save(paras)
    # eval_ss_and_save(paras)

    # paras.original_data = read_json_rows(config.FEVER_TEST_JSONL)
    # paras.upstream_data = read_json_rows(config.RESULT_PATH / 'doc_test_2020_09_21_10:05:13.jsonl')
    # paras.mode = 'test'
    # paras.pred = True
    # paras.top_n = [10]
    # paras.sample_n = 10
    # paras.prob_thresholds = 0.1
    # pred_ss_and_save(paras)

    # paras.original_data = read_json_rows(config.FEVER_DEV_JSONL)
    # paras.upstream_data = read_json_rows(config.RESULT_PATH / 'doc_dev.jsonl')
    # paras.mode = 'eval'
    # paras.pred = True
    # paras.top_n = [10]
    # paras.mode = 'dev'
    # paras.sample_n = 10
    # paras.prob_thresholds = 0.1
    # pred_ss_and_save(paras)
    paras = bert_para.PipelineParas()
    paras.data_from_pred = True
    paras.mode = 'train'
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_202103_94.9"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_202103_94.9"
    paras.output_folder = config.RESULT_PATH / 'train_2021_t'
    paras.original_data = read_json_rows(config.FEVER_TRAIN_JSONL)[80000:80002]
    paras.upstream_data = read_json_rows(config.RESULT_PATH / 'train_2021/es_doc_10.jsonl')[80000:80002]
    paras.sample_n = 10
    paras.top_n = [10]
    paras.prob_thresholds = [0.01]
    pred_ss_and_save(paras)
    # eval_lis = read_json_rows(config.RESULT_PATH / 'train_2021/item_bert_ss_0.01_10.jsonl')
    # ss_f1_score_and_save(paras, eval_lis)

