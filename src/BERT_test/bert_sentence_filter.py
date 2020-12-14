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
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm

from BERT_test.bert_data_processor import *
from BERT_sampler.ss_sampler import get_claim_sample_list
import numpy as np
from utils.resource_manager import BERTSSModel

logger = logging.getLogger(__name__)


def filter_sentences(claim, doc_l):
    bertModel = BERTSSModel()
    model = bertModel.get_model()
    eval_batch_size = 8
    eval_examples, eval_list = get_claim_sample_list(claim, doc_l)

    eval_features = convert_examples_to_features(
        eval_examples, ["true", "false"], 128, bertModel.get_tokenizer())

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([-1] * len(eval_features), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    model.eval()
    preds = []
    num_labels = 2
    device = bertModel.get_device()
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="perdicting"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        # label_ids = label_ids.to(device)
        with torch.no_grad():
            # if `labels` is `None`:
            #    Outputs the classification logits of shape [batch_size, num_labels].
            logits = model(input_ids, segment_ids, input_mask, labels=None)

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = preds[0]
    probs = softmax(preds)
    probs = probs[:, 0].tolist()
    scores = preds[:, 0].tolist()
    for i in range(len(eval_list)):
        assert str(eval_examples[i].guid) == str(eval_list[i]['selection_id'])
        # Matching id
        eval_list[i]['score'] = scores[i]
        eval_list[i]['prob'] = probs[i]

    eval_list.sort(key=lambda k: int(k['prob']), reverse=True)
    return eval_list


if __name__ == "__main__":
   pass

