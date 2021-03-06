import logging
import random
from typing import Dict, List, Tuple

import numpy as np
from allennlp.data import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from overrides import overrides
from wn_featurizer import wn_persistent_api
from wn_featurizer.additional_feature import encode_num_in_ltokens

from data_util.customized_field import IdField

# from pathlib import Path

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class WNDocSIMIReader(DatasetReader):
    """
    WordNet augmented Data Reader.
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 example_filter=None,
                 wn_p_dict=None, wn_feature_list=wn_persistent_api.default_fn_list,
                 max_l=None, num_encoding=True, shuffle_sentences=False, ablation=None) -> None:

        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='tokens')}
        self._example_filter = example_filter
        self.wn_p_dict = wn_p_dict
        if wn_p_dict is None:
            raise ValueError("Need to specify WN feature dict for FEVER Reader.")
        self.wn_feature_list = wn_feature_list
        num_encoding_dim = 5 if num_encoding else 0
        self.wn_feature_size = len(self.wn_feature_list) * 3 + num_encoding_dim + 2
        self.max_l = max_l
        self.shuffle_sentences = shuffle_sentences
        self.ablation = ablation

        if self.ablation is not None and self.ablation['rm_wn']:
            self.wn_feature_size -= (len(self.wn_feature_list) * 3 + num_encoding_dim)
        elif self.ablation is not None and self.ablation['rm_simi']:
            self.wn_feature_size -= 2

    @overrides
    def _read(self, data_list):
        logger.info("Reading Fever instances from upstream sampler")
        for example in data_list:
            label = example["label"]

            if self._example_filter is None:
                pass
            elif self._example_filter(example):
                continue

            # We use binary parse here
            premise: List[Tuple[str, float, float]] = example["evid"]
            hypothesis = example["claim"]

            if len(premise) == 0:
                premise = [("@@@EMPTY@@@", 0.0, 0.0)]

            pid = str(example['id'])

            yield self.text_to_instance(premise, hypothesis, pid, label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: List[Tuple[str, float, float]],  # Important type information
                         hypothesis: str,
                         pid: str = None,
                         label: str = None) -> Instance:

        fields: Dict[str, Field] = {}

        premise_tokens_list = []
        premise_prob_values = []
        premise_doc_prob_values = []

        # sentence_count = len(premise)

        if self.shuffle_sentences:
            # Potential improvement. Shuffle the input sentences. Maybe close this at last several epoch.
            random.shuffle(premise)

        for premise_sent, prob, doc_prob in premise:
            cur_premise_tokens = [Token(t) for t in premise_sent.split(' ')]  # Removing code for parentheses in NLI

            prob_value = np.ones((len(cur_premise_tokens), 1), dtype=np.float32) * prob
            prob_doc_value = np.ones((len(cur_premise_tokens), 1), dtype=np.float32) * doc_prob

            premise_tokens_list.extend(cur_premise_tokens)

            premise_prob_values.append(prob_value)
            premise_doc_prob_values.append(prob_doc_value)

        premise_prob = np.concatenate(premise_prob_values, axis=0)
        premise_doc_prob = np.concatenate(premise_doc_prob_values, axis=0)

        hypothesis_tokens = [Token(t) for t in hypothesis.split(' ')]
        hypothesis_prob = np.ones((len(hypothesis_tokens), 1), dtype=np.float32)
        hypothesis_doc_prob = np.ones((len(hypothesis_tokens), 1), dtype=np.float32)

        premise_prob = np.concatenate([premise_prob, premise_doc_prob], axis=1)
        hypothesis_prob = np.concatenate([hypothesis_prob, hypothesis_doc_prob], axis=1)

        if self.max_l is not None:
            premise_tokens_list = premise_tokens_list[:self.max_l]
            hypothesis_tokens = hypothesis_tokens[:self.max_l]
            premise_prob = premise_prob[:self.max_l, :]
            hypothesis_prob = hypothesis_prob[:self.max_l, :]

        fields['premise'] = TextField(premise_tokens_list, self._token_indexers)    # (t_len, 1)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)

        # WN feature dict:
        premise_s = ' '.join([t for t, p, _ in premise]).split(' ')
        hypothesis_s = hypothesis.split(' ')

        if self.max_l is not None:
            premise_s = premise_s[:self.max_l]
            hypothesis_s = hypothesis_s[:self.max_l]

        if self.ablation is not None and self.ablation['rm_wn']:
            p_feature_array = np.concatenate([premise_prob], axis=1)
            h_feature_array = np.concatenate([hypothesis_prob], axis=1)

        elif self.ablation is not None and self.ablation['rm_simi']:
            example_feature = wn_persistent_api.compute_wn_features_p_accerate(premise_s,
                                                                               hypothesis_s,
                                                                               self.wn_p_dict)

            p_wn_nparray, h_wn_nparray = wn_persistent_api.wn_raw_feature_to_nparray(
                example_feature,
                self.wn_feature_list)
            # Appending more features
            p_num_feature = encode_num_in_ltokens(premise_s)  # (t_len, 5)
            h_num_feature = encode_num_in_ltokens(hypothesis_s)  # (t_len, 5)
            p_feature_array = np.concatenate([p_wn_nparray, p_num_feature], axis=1)
            h_feature_array = np.concatenate([h_wn_nparray, h_num_feature], axis=1)

        else:
            # Whole Model no ablation.
            example_feature = wn_persistent_api.compute_wn_features_p_accerate(premise_s,
                                                                               hypothesis_s,
                                                                               self.wn_p_dict)

            p_wn_nparray, h_wn_nparray = wn_persistent_api.wn_raw_feature_to_nparray(
                example_feature,
                self.wn_feature_list)

            # Appending more features
            p_num_feature = encode_num_in_ltokens(premise_s) # (t_len, 5)
            h_num_feature = encode_num_in_ltokens(hypothesis_s) # (t_len, 5)
            p_feature_array = np.concatenate([p_wn_nparray, p_num_feature, premise_prob], axis=1)
            h_feature_array = np.concatenate([h_wn_nparray, h_num_feature, hypothesis_prob], axis=1)

        assert len(premise_tokens_list) == p_feature_array.shape[0]
        assert len(hypothesis_tokens) == h_feature_array.shape[0]

        fields['p_wn_feature'] = ArrayField(p_feature_array)
        fields['h_wn_feature'] = ArrayField(h_feature_array)

        if label:
            fields['label'] = LabelField(label, label_namespace='labels')

        if pid:
            fields['pid'] = IdField(pid)

        return Instance(fields)
