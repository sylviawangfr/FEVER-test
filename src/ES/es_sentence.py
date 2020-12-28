# elastic search
# alternative for sqlit , not in use due to slow query speed
import elasticsearch.helpers as ESH
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q

# import config
from utils.file_loader import *

es = Elasticsearch([{'host': config.ELASTIC_HOST, 'port': config.ELASTIC_PORT}])

#
# def init_index():
#     # delete index if exists
#     if es.indices.exists(config.WIKIPAGE_INDEX):
#         print("index wikipages exists, jump over creating elasticsearch index for wiki pages.")
#     else:
#         print("Creating elasticsearch index for wiki pages...")
#         wiki_mapping = read_json(config.WIKIPAGE_MAPPING)
#         es.indices.create(index=config.WIKIPAGE_INDEX, ignore=400, body=wiki_mapping)
#         print("ES index for wiki pages has been created successfully")
#
#
# def init_fever_sentence_index():
#     if es.indices.exists(config.FEVER_DOC_INDEX):
#         print("index fever_sens exists, jump over creating elasticsearch index for fever sents.")
#     else:
#         print("Creating elasticsearch index for fever sents..")
#         mapping = read_json(config.FEVER_SEN_MAPPING)
#         es.indices.create(index=config.FEVER_SEN_INDEX, ignore=400, body=mapping)
#         print("ES index for fever sentences has been created successfully")
#
#
# def init_wikipages():
#     # process wikipedia dump file to ES
#     thread_number = 5
#     thread_exe(add_wiki_bunch, config.WIKI_PAGE_PATH.iterdir(), thread_number, "indexing wiki pages")
#     print("indexed all wiki docs in ES")


def add_wiki_bunch(file):
    bunch_size = 2000
    json_rows = read_json_rows(file)
    clean_rows = [json.dumps(parse_pages_checks(row), ensure_ascii=False) for row in json_rows]
    iters = iter_baskets_contiguous(clean_rows, bunch_size)
    for piece in iters:
        ESH.bulk(es, piece, index=config.WIKIPAGE_INDEX)


def test_indexing():
    f = config.WIKI_PAGE_PATH / "wiki-001.jsonl"
    add_wiki_bunch(f)
    

def get_all_doc_ids(max_ind=None):
    id_list = []
    search = Search(using=es, index=config.WIKIPAGE_INDEX)
    must = []
    must.append({'regexp': {'text': '.+'}})
    must.append({'regexp': {'lines': '.+'}})

    search = search.query(Q('bool', must=must)). \
                 source(include=['id'])
    try:
        search.execute()
        thread_exe(lambda hit: id_list.append(hit.id), search.scan(), 100, "get doc ids")
    except Exception as e:
        print(e)
    finally:
        return id_list


def get_evidence_es(doc_id, line_num):
    key = f'{doc_id}(-.-){line_num}'
    # cursor.execute("SELECT * FROM sentences WHERE id=?", (normalize(key),))
    search = Search(using=es, index=config.FEVER_SEN_INDEX)
    key = normalize(key)
    search = search.query("match", id=key)
    _id, text, h_links, doc_id = None, None, None, None
    try:
        search.execute()
        hit = next(search.scan())
        _id = hit.id
        text = hit.text
        h_links = hit.h_links
    except Exception as e:
        print(e)
    finally:
        return _id, text, h_links



def get_all_sent_by_doc_id_es(doc_id, with_h_links=False):
    r_list = []
    id_list = []
    h_links_list = []
    search = Search(using=es, index=config.FEVER_SEN_INDEX)
    doc_id = normalize(doc_id)
    search = search.query("match", doc_id=doc_id)
    for hit in search.scan():
        r_list.append(hit.text)
        id_list.append(hit.id)
        h_links_list.append(json.loads(hit.h_links))

    if with_h_links:
        return r_list, id_list, h_links_list
    else:
        return r_list, id_list



if __name__ == '__main__':
    # init_index()
    # init_wikipages()
    # test_indexing()
     pass