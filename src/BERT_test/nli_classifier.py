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

import torch
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import BertTokenizer
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import utils.common_types as bert_para
from BERT_test.bert_data_processor import *
from BERT_test.nli_eval import eval_nli_examples
from data_util.toChart import *
from utils.file_loader import get_current_time_str
from utils.file_loader import read_json_rows
import numpy as np

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
    if task_name == "ss":
        return {"acc": acc_and_f1(preds, labels)}
    elif task_name == "nli":
        return {"acc": acc_and_f1(preds, labels)}
    else:
        raise KeyError(task_name)


def nli_finetuning(upstream_train_data, output_folder='fine_tunning', sampler=None):
    bert_model = "bert-large-uncased"
    pretrained_model_name_or_path = config.PRO_ROOT / "saved_models/bert/bert-large-uncased.tar.gz"
    cache_dir = config.PRO_ROOT / "saved_models" / "bert_finetuning"
    output_dir = config.PRO_ROOT / "saved_models" / "bert_finetuning" / f"nli_{output_folder}"
    max_seq_length = 300
    do_lower_case = True
    train_batch_size = 16
    learning_rate = 5e-6
    num_train_epochs = 3.0
    # Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.
    warmup_proportion = 0.1
    # local_rank for distributed training on gpus
    local_rank = -1

    seed = 42
    gradient_accumulation_steps = 8

    loss_scale = 0
    server_ip = None
    server_port = None
    do_eval = True
    processor = FeverNliProcessor()
    # fp16 = True if torch.cuda.is_available() else False
    fp16 = False

    if server_ip and server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(server_ip, server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(local_rank != -1), fp16))

    if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            gradient_accumulation_steps))

    train_batch_size = int(train_batch_size / gradient_accumulation_steps)

    random.seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    task_name = 'nli'

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

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
    train_examples = processor.get_train_examples(upstream_train_data, sampler)
    print(f"length of train examples: {len(train_examples)}")
    logger.info("  Batch size = %d", train_batch_size)
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
    train_dataloader = DataLoader(train_data, pin_memory=True, sampler=train_sampler, batch_size=train_batch_size)

    loss_for_chart = []
    model.train()
    for epoch in range(int(num_train_epochs)):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        epoch_loss = []
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # print('input_ids: ', len(input_ids))
                # define a new function to compute loss values for both output_modes
                try:
                    logits = model(input_ids, segment_ids, input_mask, labels=None)

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
                    pbar.update(1)
                    mean_loss = tr_loss * gradient_accumulation_steps / nb_tr_steps
                    pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                    epoch_loss.append(mean_loss)
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

                except Exception as e:
                    print("exception happened: ")
                    # e = sys.exc_info()[0]
                    print("Error: %s" % e)
                    print(torch.cuda.current_device())
                    raise e
        loss_for_chart.append(epoch_loss)
    draw_loss_epoch_detailed(loss_for_chart, f"loss_{output_folder}_{learning_rate}")
    # print(f"get rank: {torch.distributed.get_rank()}")
    if local_rank == -1 or torch.distributed.get_rank() == 0:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        print("save model to: ", output_dir)
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
        paras = bert_para.PipelineParas()
        paras.original_data = read_json_rows(config.FEVER_DEV_JSONL)
        paras.upstream_data = read_json_rows(config.RESULT_PATH / "tfidf/dev_2019_06_15_15:48:58.jsonl")
        paras.data_from_pred = False
        paras.mode = 'eval'
        paras.BERT_model = output_dir
        paras.BERT_tokenizer = output_dir
        paras.output_folder = config.LOG_PATH
        paras.sample_n = 3
        paras.sampler = "nli_nn"
        eval_nli_examples(paras)


if __name__ == "__main__":
    # train_data = read_json_rows(config.RESULT_PATH / "tfidf/train_2019_06_15_15:48:58.jsonl")
    # train_data = read_json_rows(config.RESULT_PATH / 'train_s_tfidf_retrieve.jsonl')
    train_data = read_json_rows(config.RESULT_PATH / 'train_2021/train_ss.jsonl')
    nli_finetuning(train_data, output_folder="nli_train_extended" + get_current_time_str(), sampler='nli_nn')
    # train_data = read_json_rows(config.RESULT_PATH / "bert_ss_dev_10/eval_data_ss_10_dev_0.1_top[10].jsonl")[0:1000]
    # nli_finetuning(train_data, output_folder="nli_train_extended" + get_current_time_str(), sampler='nli_tfidf')



