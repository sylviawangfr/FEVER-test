import json

import config
from data_util.tokenizers import tokenizer, spacy_tokenizer
from utils import fever_db, text_clean
from tqdm import tqdm


def easy_tokenize(text, tok):
    return tok.tokenize(text_clean.normalize(text)).words()


def save_jsonl(d_list, filename):
    print("Save to Jsonl:", filename)
    with open(filename, encoding='utf-8', mode='w') as out_f:
        for item in d_list:
            out_f.write(json.dumps(item) + '\n')


def load_jsonl(filename):
    d_list = []
    with open(filename, encoding='utf-8', mode='r') as in_f:
        print("Load Jsonl:", filename)
        for line in tqdm(in_f):
            item = json.loads(line.strip())
            d_list.append(item)

    return d_list


def tokenized_claim(in_file, out_file):
    tok = spacy_tokenizer.SpacyTokenizer(annotators={'pos', 'lemma'}, model='en_core_web_sm')
    d_list = load_jsonl(in_file)
    for item in tqdm(d_list):
        item['claim'] = ' '.join(easy_tokenize(item['claim'], tok))

    save_jsonl(d_list, out_file)


def tokenized_claim_list(in_list):
    tok = spacy_tokenizer.SpacyTokenizer(annotators=['pos', 'lemma'])
    for item in tqdm(in_list):
        item['claim'] = ' '.join(easy_tokenize(item['claim'], tok))

    return in_list

def tokenize_docids(in_file, out_file):
    tok = spacy_tokenizer.SpacyTokenizer(annotators={'pos', 'lemma'}, model='en_core_web_sm')
    d_list = load_jsonl(in_file)
    for item in tqdm(d_list):
        item['word'] = ' '.join(easy_tokenize(item['claim'], tok))

    save_jsonl(d_list, out_file)

def test_spacy(text):
    tok = spacy_tokenizer.SpacyTokenizer(annotators={'pos', 'lemma'}, model='en_core_web_sm')
    print(tok.tokenize(text_clean.normalize(text)).words())


if __name__ == '__main__':
    test_spacy("Hourglass is performed by an Australian singer-songwriter.")
    # tokenized_claim(config.FEVER_DEV_JSONL, config.DATA_ROOT / "tokenized_fever/dev.jsonl")
    # tokenized_claim(config.FEVER_TRAIN_JSONL, config.DATA_ROOT / "tokenized_fever/train.jsonl")