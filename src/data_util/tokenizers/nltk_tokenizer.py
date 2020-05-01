#!/usr/bin/env python3

"""Tokenizer that is backed by NLTK

Requires punkt,
nltk.download('punkt')
"""

import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import copy
from .tokenizer import Tokens, Tokenizer
from nltk.chunk import tree2conlltags
from nltk.parse import CoreNLPParser
from nltk.parse import CoreNLPDependencyParser

from nltk.parse.dependencygraph import *
from nltk.tree import *



class NLTKTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = kwargs.get('model', 'en')
        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        self.nlp = nltk

    def get_spans(self, txt):
        tokens = nltk.word_tokenize(txt)
        offset = 0
        for i in range(len(tokens)):
            token = tokens[i]
            offset = txt.find(token, offset)
            start_ws = offset
            if i + 1 < len(tokens):
                end_ws = txt.find(tokens[i+1], offset)
            else:
                end_ws = offset + len(tokens[i].text)

            token_ws = txt[start_ws: end_ws],
            offset = txt.find(token, offset)

            yield token, token_ws, (offset, offset + len(token))
            offset += len(token)


    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')
        tokens = self.nlp.word_tokenize(clean_text)
        lemma_func = WordNetLemmatizer()

        if {'lemma', 'pos'} & self.annotators:
            tagged = pos_tag(tokens)

        if {'ner'} & self.annotators:
            ents = nltk.chunk.ne_chunk(tagged)
            ents_tags = tree2conlltags(ents)

        data = []
        sps = self.get_spans(clean_text)
        for i in range(len(tokens)):
            t, t_ws, os = sps.next()
            data.append((
                tokens[i],
                t_ws,
                os,
                tagged[i][1],
                lemma_func.lemmatize(tokens[i]),
                ents_tags[i][2],
            ))

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={'non_ent': ''})


    def parse_dependency(self, text):
        parser = CoreNLPParser(url='http://localhost:9000')
        iter = parser.raw_parse(text)
        t = next(iter)
        t.pretty_print()
        parser = CoreNLPDependencyParser(url='http://localhost:9000')
        dg = next(parser.raw_parse(text))
        t2 = dg.tree()
        t2.pretty_print()





