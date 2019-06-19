from __future__ import absolute_import, division, print_function

import logging
from sentence_retrieval.sampler_for_nmodel import get_full_list
import config

from utils.text_clean import *
import sample_for_nli.adv_sampler_v01 as nn_sampler
import sample_for_nli.tf_idf_sample_v1_0 as tfidf_sampler
import sentence_retrieval.sampler_for_nmodel as ss_sampler

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
        sampler = self._get_sampler('tfidf')
        train_list = sampler(data_dir, pred=pred)
        return self._create_examples(train_list)

    def get_dev_examples(self, data_dir, pred = False):
        """See base class."""
        sampler = self._get_sampler('tfidf')
        dev_list = sampler(data_dir, pred=pred)
        return self._create_examples(dev_list), dev_list

    def get_test_examples(self, data_dir, pred=True):
        """See base class."""
        sampler = self._get_sampler('doc')
        test_list = sampler(config.FEVER_TEST_JSONL, data_dir, pred=pred, top_k=10)
        return self._create_examples(test_list), test_list

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

    def _get_sampler(self, s):
        processors = {
            "tfidf": ss_sampler.get_tfidf_sample_list_for_nn,
            "doc": ss_sampler.get_full_list
        }
        return processors[s]



class FeverNliProcessor(DataProcessor):

    def get_train_examples(self, data_dir, sampler='nn'):
        sampler_fun = self.get_sampler(sampler)
        train_list = sampler_fun(config.FEVER_TRAIN_JSONL, data_dir, tokenized=True)
        return self._create_examples(train_list)

    def get_dev_examples(self, data_dir, pred = False, sampler='nn'):
        """See base class."""
        dev_list = self.get_sampler(config.FEVER_DEV_JSONL, data_dir, tokenized=True)
        return self._create_examples(dev_list), dev_list

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
