from utils import common
import config
from utils.fever_db import create_db, save_wiki_pages, create_sent_db, build_sentences_table, check_document_id
from data_util.data_preperation.tokenize_fever import *
from data_util.data_preperation.build_tfidf import *
from data_util.data_readers.fever_reader import *
import fire
import argparse

def tokenization():
    print("Start tokenizing dev and training set.")
    tokenized_claim(config.FEVER_DEV_JSONL, config.T_FEVER_DEV_JSONL)
    # tokenized_claim(config.FEVER_TRAIN_JSONL, config.T_FEVER_TRAIN_JSONL)
    print("Tokenization finished.")


def build_database():
    print("Start building wiki document database. This might take a while.")
    create_db(str(config.FEVER_DB))
    save_wiki_pages(str(config.FEVER_DB))
    create_sent_db(str(config.FEVER_DB))
    build_sentences_table(str(config.FEVER_DB))
    check_document_id(str(config.FEVER_DB))
    print("Wiki document database is ready.")


def build_tfidf():
    args = argparse.Namespace(db_path=str(config.FEVER_DB),
                              num_workers=2,
                              ngram=2,
                              hash_size=int(math.pow(2, 24)),
                              tokenizer='simple',
                              out_dir=str(config.DATA_ROOT / 'saved_models'))
    build_and_save_tfidf(args)


def build_fever_vocab():
    input_file = config.T_FEVER_TRAIN_JSONL
    d_list = load_jsonl(input_file)

    # Dev set
    input_file = config.T_FEVER_DEV_JSONL

    d_list.extend(load_jsonl(input_file))

    vocab = fever_build_vocab(d_list)
    print(vocab)

    build_vocab_embeddings(vocab, config.DATA_ROOT / "embeddings/glove.840B.300d.txt",
                           embd_dim=300, saved_path=config.DATA_ROOT / "vocab_cache" / "nli_basic")


if __name__ == '__main__':
    fire.Fire()
    # tokenization()
    # build_database()
    # build_tfidf()
    build_fever_vocab()
