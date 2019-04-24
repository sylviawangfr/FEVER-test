# elastic search
from utils.file_loader import *
from utils.thread_executor import thread_exe
import config
import tqdm
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Document, Date, Integer, Keyword, Text
from elasticsearch_dsl.connections import connections

es = Elasticsearch([{'host': config.ELASTIC_HOST, 'port': config.ELASTIC_PORT}])
def init_index():
    # delete index if exists
    if es.indices.exists(config.WIKIPAGE_INDEX):
        print("index wikipages exists, jump over creating elasticsearch index for wiki pages.")
    else:
        print("Creating elasticsearch index for wiki pages...")
        wiki_mapping = read_data(config.WIKIPAGE_MAPPING)
        es.indices.create(index=config.WIKIPAGE_INDEX, ignore=400, body=wiki_mapping)
        print("ES index for wiki pages has been created successfully")


def init_wikipages():
    # process wikipedia dump file to ES
    thread_number = 5
    page_folder = config.DATA_ROOT / "wiki-pages"
    thread_exe(add_wiki_bunch, tqdm(page_folder), thread_number)


def add_wiki_bunch(file):
    bunch_size = 50
    json_rows = read_data(file)


def add_evids():
    pass


if __name__ == '__main__':
    init_es()
    init_wikipedia()