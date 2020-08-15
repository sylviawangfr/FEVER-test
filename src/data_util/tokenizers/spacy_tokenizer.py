#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Tokenizer that is backed by spaCy (spacy.io).

Requires spaCy package and the spaCy english model.
"""

import copy

import spacy

from .tokenizer import Tokens, Tokenizer


class SpacyTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = 'en_core_web_sm'
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        nlp_kwargs = {'parser': False}
        if not {'lemma', 'pos', 'ner'} & self.annotators:
            nlp_kwargs['tagger'] = False
        if not {'ner'} & self.annotators:
            nlp_kwargs['entity'] = False
        self.nlp = spacy.load(model, **nlp_kwargs)


    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')
        tokens = self.nlp.tokenizer(clean_text)
        if {'lemma', 'pos', 'ner'} & self.annotators:
            self.nlp.tagger(tokens)
        if {'ner'} & self.annotators:
            self.nlp.entity(tokens)

        data = []
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i].idx
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1].idx
            else:
                end_ws = tokens[i].idx + len(tokens[i].text)

            data.append((
                tokens[i].text,
                text[start_ws: end_ws],
                (tokens[i].idx, tokens[i].idx + len(tokens[i].text)),
                tokens[i].tag_,
                tokens[i].lemma_,
                tokens[i].ent_type_,
            ))

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={'non_ent': ''})

    def parseDependency(self, text):
        clean_text = text.replace('\n', ' ')
        doc = self.nlp(text)
        for word in doc:
            if word.dep_ in ("xcomp", "ccomp"):
                print("".join(w.text_with_ws for w in word.subtree))


    def parse_noun_chunks(self, text):
        clean_text = text.replace('\n', ' ')
        doc = self.nlp("text")
        chunks = []
        for chunk in doc.noun_chunks:
            chunks.append(chunk.text)
        return chunks
