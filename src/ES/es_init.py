# elastic search
import sys

import elasticsearch.helpers as ESH
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

import config
from utils.file_loader import *

client = Elasticsearch([{'host': config.ELASTIC_HOST, 'port': config.ELASTIC_PORT, 'timeout': 300, 'max_retries': 10, 'retry_on_timeout': True}])


def init_index():
    # delete index if exists
    if client.indices.exists(config.WIKIPAGE_INDEX):
        print("index wikipages exists, jump over creating elasticsearch index for wiki pages.")
    else:
        print("Creating elasticsearch index for wiki pages...")
        wiki_mapping = read_json(config.WIKIPAGE_MAPPING)
        try:
            client.indices.create(index=config.WIKIPAGE_INDEX, body=wiki_mapping)
            print("ES index for wiki pages has been created successfully")
        except Exception as err:
            print("exception happened: ")
            print(err)
            e = sys.exc_info()[0]


def init_fever_sentence_index():
    if client.indices.exists(config.FEVER_SEN_INDEX):
        print("index fever_sens exists, jump over creating elasticsearch index for fever sents.")
    else:
        print("Creating elasticsearch index for fever sents..")
        mapping = read_json(config.FEVER_SEN_MAPPING)
        client.indices.create(index=config.FEVER_SEN_INDEX, ignore=400, body=mapping)
        print("ES index for fever sentences has been created successfully")


# def build_sentences_records():
    # count = 0
    # search = Search(using=client, index=config.WIKIPAGE_INDEX)
    # must = []
    # must.append({'regexp': {'text': '.+'}})
    # must.append({'regexp': {'lines': '.+'}})
    #
    # search = search.query(Q('bool', must=must))
    # try:
    #     search.execute()
    #     print("got all wiki doc records.")
    #     thread_exe(bulk_save_sentence_hit, search.scan(), 8, "building sentence index")
    # except Exception as e:
    #     print(e)
    # print("done with building sentence index")



# def bulk_save_sentence_hit(hit):
#     doc_id = hit.id
#     lines = hit.lines
#     lines_items = json.loads(lines)
#     sen_json_l = []
#     for line in lines_items:
#         if line['sentences']:
#             sent_pid = doc_id + '(-.-)' + str(line['line_num'])
#             sent = line['sentences']
#             h_links = json.dumps(line['h_links'])
#             # to json
#             s_dict = {'sid': sent_pid, 'text': sent, 'h_links': h_links, 'doc_id': doc_id}
#             sen_json_l.append(json.dumps(s_dict))
#
#     # bunk insert es
#     ESH.bulk(client, sen_json_l, index=config.FEVER_SEN_INDEX)


def index_wikipages():
    # process wikipedia dump file to ES
    thread_number = 5
    thread_exe(add_wiki_bunch, config.WIKI_PAGE_PATH.iterdir(), thread_number, "indexing wiki pages")
    print("indexed all wiki docs in ES")


def index_sentences():
    thread_number = 5
    thread_exe(add_sentence_bunch, config.WIKI_PAGE_PATH.iterdir(), thread_number, "indexing sentence pages")


def add_wiki_bunch(file):
    bunch_size = 2000
    json_rows = read_json_rows(file)
    clean_rows = [json.dumps(parse_pages_checks(row), ensure_ascii=False) for row in json_rows]
    iters = iter_baskets_contiguous(clean_rows, bunch_size)
    for piece in iters:
        ESH.bulk(client, piece, index=config.WIKIPAGE_INDEX)


def add_sentence_bunch(file):
    json_rows = read_json_rows(file)
    clean_rows = [parse_pages_checks(row) for row in json_rows]
    sen_json_l = []
    for r in clean_rows:
        doc_id = r['id']
        lines = json.loads(r['lines'])
        for line in lines:
            if line['sentences']:
                sid = doc_id + '(-.-)' + str(line['line_num'])
                sent = line['sentences']
                h_links = json.dumps(line['h_links'], ensure_ascii=False)
                # to json
                s_dict = {'sid': sid, 'text': sent, 'h_links': h_links, 'doc_id': doc_id, 'doc_id_keyword': doc_id}
                sen_json_l.append(json.dumps(s_dict, ensure_ascii=False))
    ESH.bulk(client, sen_json_l, index=config.FEVER_SEN_INDEX)


def test_indexing():
    f = config.WIKI_PAGE_PATH / "wiki-001.jsonl"
    add_sentence_bunch(f)


if __name__ == '__main__':
    # test_indexing()
    # init_index()
    # index_wikipages()

    init_fever_sentence_index()
    index_sentences()
