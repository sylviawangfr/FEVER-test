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

import os
import random

import numpy as np
import torch
from sklearn.utils.extmath import softmax
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from utils.file_loader import save_jsonl, read_json_rows, get_current_time_str, save_file, save_intermidiate_results
from typing import Dict
from utils import c_scorer
from BERT_test.bert_data_processor import *
from sample_for_nli.tf_idf_sample_v1_0 import convert_evidence2scoring_format


logger = logging.getLogger(__name__)




def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]


        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


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
    if task_name == "ss":
        return {"acc": acc_and_f1(preds, labels)}
    elif task_name == "nli":
        return {"acc": acc_and_f1(preds, labels)}
    else:
        raise KeyError(task_name)


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

    save_file(pred_log, config.LOG_PATH / f"{get_current_time_str()}__ss_pred.log")

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

        eval_mode = {'check_sent_id_correct': True, 'standard': True}
        for a, b in zip(actual_list, results_list):
            b['predicted_label'] = a['label']
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


def eval_nli_and_save(pred=False):
    pass


def fever_finetuning(taskname, upstream_train_data, upstream_dev_data):
    bert_model = "bert-large-uncased"
    pretrained_model_name_or_path = config.PRO_ROOT / "saved_models/bert/bert-large-uncased.tar.gz"
    cache_dir = config.PRO_ROOT / "saved_models" / "bert_finetuning"
    output_dir = config.PRO_ROOT / "saved_models" / "bert_finetuning" / get_current_time_str()
    max_seq_length = 128
    do_lower_case = True
    train_batch_size = 32
    learning_rate = 5e-5
    num_train_epochs = 3.0
    # Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.
    warmup_proportion = 0.1
    # local_rank for distributed training on gpus
    local_rank = -1
    seed = 42
    gradient_accumulation_steps = 8
    fp16 = True
    loss_scale = 0
    server_ip = None
    server_port = None
    do_eval = True

    if server_ip and server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(server_ip, server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "ss": FeverSSProcessor,
        "nli": FeverNliProcessor,
    }

    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(local_rank != -1), fp16))

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            gradient_accumulation_steps))

    train_batch_size = train_batch_size // gradient_accumulation_steps

    random.seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    task_name = taskname.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    # tokenizer = BertTokenizer.from_pretrained(bert_model)

    # Prepare model
    cache_dir = cache_dir if cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(local_rank))
    model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                          cache_dir=cache_dir,
                                                          num_labels=num_labels)
    if fp16:
        model.half()
    model.to(device)
    if local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # get train data
    train_examples = None
    num_train_optimization_steps = None
    train_examples = processor.get_train_examples(upstream_train_data, pred=False)
    num_train_optimization_steps = int(
        len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
    if local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    train_features = convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    model.train()
    for _ in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # define a new function to compute loss values for both output_modes
            try:
                logits = model(input_ids, segment_ids, input_mask, labels=None)
            except:
                print(torch.cuda.current_device())
                print(torch.cuda.cudaStatus)

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    # modify learning rate with special warm up BERT uses
                    # if fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = learning_rate * warmup_linear.get_lr(global_step, warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

    if local_rank == -1 or torch.distributed.get_rank() == 0:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        # model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=num_labels)
        # tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=do_lower_case)
    # else:
        # model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=num_labels)
    # model.to(device)

    if do_eval and (local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_ss_and_save(output_dir, output_dir, upstream_dev_data, pred=False, mode='dev')


if __name__ == "__main__":
    fever_finetuning('ss', config.RESULT_PATH / "tfidf/train_2019_06_15_15:48:58.jsonl", config.RESULT_PATH / "tfidf/dev_2019_06_15_15:48:58.jsonl")
    # eval_ss_and_save(config.PRO_ROOT / "saved_models/bert/bert-large-uncased.tar.gz", "bert-large-uncased")
    # eval_ss_and_save(config.PRO_ROOT / "saved_models/bert_finetuning/2019_06_13_17:07:55",
    #                  config.PRO_ROOT / "saved_models/bert_finetuning/2019_06_13_17:07:55")
