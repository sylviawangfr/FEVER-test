from utils.fever_db import create_db, save_wiki_pages, create_sent_db, build_sentences_table, check_document_id
from data_util.data_readers.fever_reader import *
import fire
import argparse
from tfidf_model.tokenize_fever import *
from tfidf_model.build_tfidf import *

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


if __name__ == '__main__':
    fire.Fire()
    build_database()
    build_tfidf()
