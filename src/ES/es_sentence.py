# elastic search
from utils.file_loader import *
import config
from elasticsearch import Elasticsearch
import elasticsearch.helpers as ESH

es = Elasticsearch([{'host': config.ELASTIC_HOST, 'port': config.ELASTIC_PORT}])
def init_index():
    # delete index if exists
    if es.indices.exists(config.WIKIPAGE_INDEX):
        print("index wikipages exists, jump over creating elasticsearch index for wiki pages.")
    else:
        print("Creating elasticsearch index for wiki pages...")
        wiki_mapping = read_json(config.WIKIPAGE_MAPPING)
        es.indices.create(index=config.WIKIPAGE_INDEX, ignore=400, body=wiki_mapping)
        print("ES index for wiki pages has been created successfully")


def init_fever_sentence_index():
    if es.indices.exists(config.FEVER_DOC_INDEX):
        print("index fever_sens exists, jump over creating elasticsearch index for fever sents.")
    else:
        print("Creating elasticsearch index for fever sents..")
        mapping = read_json(config.FEVER_SEN_MAPPING)
        es.indices.create(index=config.FEVER_SEN_INDEX, ignore=400, body=mapping)
        print("ES index for fever sentences has been created successfully")


def init_wikipages():
    # process wikipedia dump file to ES
    thread_number = 5
    thread_exe(add_wiki_bunch, config.WIKI_PAGE_PATH.iterdir(), thread_number, "indexing wiki pages")
    print("indexed all wiki docs in ES")


def add_wiki_bunch(file):
    bunch_size = 2000
    json_rows = read_json_rows(file)
    clean_rows = [json.dumps(parse_pages_checks(row)) for row in json_rows]
    iters = iter_baskets_contiguous(clean_rows, bunch_size)
    for piece in iters:
        ESH.bulk(es, piece, index=config.WIKIPAGE_INDEX)


def test_indexing():
    f = config.WIKI_PAGE_PATH / "wiki-001.jsonl"
    add_wiki_bunch(f)


if __name__ == '__main__':
    init_index()
    init_wikipages()
    # test_indexing()
    #  pass