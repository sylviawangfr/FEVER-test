from __future__ import absolute_import, division, print_function

import logging
from sentence_retrieval.sampler_for_nmodel import get_full_list
import config

from utils.text_clean import *
import sample_for_nli.adv_sampler_v01 as nn_sampler
import sample_for_nli.tf_idf_sample_v1_0 as tfidf_sampler
from sentence_retrieval.sampler_for_nmodel import get_tfidf_sample_list_for_nn

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class FeverSSProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir, pred = False):
        train_list = get_tfidf_sample_list_for_nn(data_dir, pred=pred)
        return self._create_examples(train_list)

    def get_dev_examples(self, data_dir, pred = False):
        """See base class."""
        dev_list = get_tfidf_sample_list_for_nn(data_dir, pred=pred)
        return self._create_examples(dev_list), dev_list

    def get_labels(self):
        """See base class."""
        return ["true", "false"]

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            guid = line['selection_id']
            text_a = convert_brc(line['text'])
            text_b = convert_brc(line['query'])
            label = line['selection_label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class FeverNliProcessor(DataProcessor):

    def get_train_examples(self, data_dir, sampler='tfidf'):
        sampler_fun = self.get_sampler(sampler)
        train_list = sampler_fun(config.FEVER_TRAIN_JSONL, data_dir, tokenized=True)
        return self._create_examples(train_list)

    def get_dev_examples(self, data_dir, pred = False, sampler='tfidf'):
        """See base class."""
        dev_list = FeverNliProcessor.get_sampler(config.FEVER_DEV_JSONL, data_dir, tokenized=True)
        return self._create_examples(dev_list)

    def get_labels(self):
        """See base class."""
        return ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line['id']
            text_a = convert_brc(line['evid'])
            text_b = convert_brc(line['claim'])
            label = line['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _get_sampler(self, s):
        processors = {
            "tfidf": tfidf_sampler.sample_v1_0,
            "nn": nn_sampler.get_adv_sampled_data
        }
        return processors[s]