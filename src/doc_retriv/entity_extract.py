import spacy
import json


class EntityExtractor:

    def __init__(self):
        self.nlp_spacy = spacy.load("en_core_web_sm")

    def load_tokenizer(self, spacy_tokenizer):
        self.nlp_spacy = spacy.load(spacy_tokenizer)
        return self.nlp_spacy

    def extract_entities(self, text):
        doc = self.nlp_spacy(text)
        return doc.ents

    def split_sents(self, text):
        doc = self.nlp_spacy(text)
        return doc.sents
