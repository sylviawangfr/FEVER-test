# elastic search
from utils.file_loader import *
from utils.thread_executor import thread_exe
import config
from tqdm import tqdm
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


def init_wikipages():
    # process wikipedia dump file to ES
    thread_number = 5
    thread_exe(add_wiki_bunch, config.WIKI_PAGE_PATH.iterdir(), thread_number, "indexing wiki pages")


def add_wiki_bunch(file):
    bunch_size = 200
    json_rows = read_json_rows(file)
    iters = iter_baskets_contiguous(json_rows, bunch_size)
    for piece in tqdm(iters, total=100, desc="bulk indexing", position=1):
        ESH.bulk(es, piece, index=config.WIKIPAGE_INDEX)


def add_evids():
    pass


if __name__ == '__main__':
    init_index()
    init_wikipages()
    #  pass