#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# DEFAULTS = {
#     'corenlp_classpath': os.getenv('CLASSPATH')
# }


from data_util.tokenizers.nltk_tokenizer import NLTKTokenizer
# def set_default(key, value):
#     # global DEFAULTS
#     DEFAULTS[key] = value
from data_util.tokenizers.regexp_tokenizer import RegexpTokenizer
from data_util.tokenizers.simple_tokenizer import SimpleTokenizer

# Spacy is optional
try:
    from .spacy_tokenizer import SpacyTokenizer
except ImportError:
    pass


def get_class(name):
    if name == 'spacy':
        return SpacyTokenizer
    # if name == 'corenlp':
    #     return CoreNLPTokenizer
    if name == 'regexp':
        return RegexpTokenizer
    if name == 'simple':
        return SimpleTokenizer

    raise RuntimeError('Invalid tokenizer: %s' % name)


def get_annotators_for_args(args):
    annotators = set()
    if args.use_pos:
        annotators.add('pos')
    if args.use_lemma:
        annotators.add('lemma')
    if args.use_ner:
        annotators.add('ner')
    return annotators


def get_annotators_for_model(model):
    return get_annotators_for_args(model.args)

if __name__ == "__main__":
    t = NLTKTokenizer()
    t.parse_dependency('Roman Atwood is a content creator.')
