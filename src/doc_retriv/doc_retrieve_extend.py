from ES.es_search import search_and_merge, search_doc_id, search_and_merge2, search_and_merge3, merge_result
from utils.c_scorer import *
from utils.common import thread_exe
from utils.fever_db import *
from utils.file_loader import read_json_rows, get_current_time_str, read_all_files
from dbpedia_sampler.dbpedia_triple_linker import link_sentence
from dbpedia_sampler.dbpedia_virtuoso import get_resource_wiki_page
from dbpedia_sampler.sentence_util import get_phrases, get_phrases_and_nouns_merged
import difflib
from utils.text_clean import convert_brc
from dbpedia_sampler.dbpedia_subgraph import construct_subgraph_for_claim, get_isolated_nodes
from dbpedia_sampler.uri_util import isURI
from dbpedia_sampler.dbpedia_virtuoso import get_categories2
from bert_serving.client import BertClient


def retrieve_docs(example, context_dict=None):
    claim = example['claim']
    id = example['id']
    result_context = []
    result_uris = []
    if context_dict is not None and len(context_dict) > 0:
        context, object_urls = get_context_dbpedia(context_dict, id)
        result_context = search_and_merge3(context)
        result_uris = search_extended_URIs(object_urls)
    nouns = get_phrases_and_nouns_merged(claim)
    result_es = search_and_merge2(nouns)
    result_dbpedia = search_entity_dbpedia(claim)
    result_dbpedia.extend(result_uris)
    result_dbpedia = merge_result(result_dbpedia)
    result = merge_es_and_dbpedia(result_es, result_dbpedia, result_context)
    if len(result) > 10:
        result = result[:10]
    return result


def strategy_over_all(claim):
    doc_result = []
    claim_dict = construct_subgraph_for_claim(claim)
    claim_graph = claim_dict['graph']
    if len(claim_graph) > 0:
        isolated_nodes = get_isolated_nodes(claim_dict)
        # 1. ES search linked entity page
        # 2. BERT filter: linked triples VS entity pages -> candidate sentences
        # 3. BERT filter: claim VS entity page sentences
        if len(isolated_nodes) > 0:
            # 1. candidate sentence to sent_context_graph
            # 2. ES search sent_context_graph new entity pages
            # 3. BERT filter:  extended context triples VS new entity page sentences  -> aggregated envidence set
            pass
        else:
            # search triple doc
            pass
    else:
        # cannot extract context graph
        nouns = get_phrases_and_nouns_merged(claim)
        doc_result = search_and_merge2(nouns)
    pass


def strategy_search_phrases(claim):
    nouns = get_phrases_and_nouns_merged(claim)
    result_es = search_and_merge2(nouns)
    return result_es


def strategy_search_claim_context(claim_graph):
    if len(claim_graph) > 0:
        isolated_nodes = get_isolated_nodes()
    pass


def strategy_search_candidate_extended_context(sents, claim_graph, claim_isolated_nodes):
    pass


def search_entities_and_nouns(claim):
    ents, phrases = get_phrases(claim)
    result_es = search_and_merge2(list(set(ents) | set(phrases)))


def search_entities_extended(ents):
    # 1. link entities to dbpedia
    # 2. find connections between entities
    # 3. search entity and extended phrases

    pass


def filter_entities_in_media_subset(resources):
    pass


def merge_es_and_dbpedia(r_es, r_db, r_context=[]):
    r_es_ids = [i['id'] for i in r_es]
    r_db_ids = [i['id'] for i in r_db]
    r_context_ids = [i['id'] for i in r_context]
    for idx_i, i in enumerate(r_es_ids):
        for idx_j, j in enumerate(r_db_ids):
            if i == j:
                if len(r_es[idx_i]['phrases']) > 1:
                    r_es[idx_i]['score'] += r_db[idx_j]['score']
                else:
                    p = r_db[idx_j]['phrases'][0].lower()
                    doc_id = convert_brc(r_db[idx_j]['id']).replace('_', ' ').lower()
                    ratio = difflib.SequenceMatcher(None, p, doc_id).ratio()
                    if ratio > 0.8:
                        r_es[idx_i]['score'] += r_db[idx_j]['score'] * 0.5
    merged = r_es
    for idx, i in enumerate(r_db_ids):
        if i not in r_es_ids:
            p = r_db[idx]['phrases'][0].lower()
            doc_id = convert_brc(r_db[idx]['id']).replace('_', ' ').lower()
            ratio = difflib.SequenceMatcher(None, p, doc_id).ratio()
            if ratio > 0.8:
                r_db[idx]['score'] *= 2
            merged.append(r_db[idx])

    merged_ids = [i['id'] for i in merged]
    for idx_j, j in enumerate(r_context_ids):
        for idx_i, i in enumerate(merged_ids):
            if i == j:
                merged[idx_i]['score'] += r_context[idx_j]['score']

    for idx_j, j in enumerate(r_context_ids):
        if j not in merged_ids:
            merged.append(r_context[idx_j])

    merged.sort(key=lambda x: x.get('score'), reverse=True)
    return merged


def search_entity_dbpedia(claim):
    not_linked_phrases_l, linked_phrases_l = link_sentence(claim)
    docs = []
    for resource in linked_phrases_l:
        resource_uri = resource['URI']
        wiki_links = get_resource_wiki_page(resource_uri)
        if wiki_links is None or len(wiki_links) < 1:
            continue
        for item in wiki_links:
            possible_doc_id = item.split('/')[-1]
            verified_id_es = search_doc_id(possible_doc_id)
            for r_es in verified_id_es:
                if len(list(filter(lambda x: (x['id'] == r_es['id']), docs))) < 1:
                    docs.append({'id': r_es['id'], 'score': r_es['score'], 'phrases': [resource['text']]})
    return docs


def link_entities(claim):
    not_linked_phrases_l, linked_phrases_l = link_sentence(claim)
    triples = []
    phrase_links = dict()
    threshold = 0.5
    for resource in linked_phrases_l:
        resource_uri = resource['URI']

        phrase = resource['text']
        if phrase in phrase_links:
            phrase_links[phrase].append(resource_uri)
        else:
            phrase_links[phrase] = [resource_uri]
    for k,v in dict.iteritems():
        if len(v) > 1:

            # sorted_matching_index = sorted(range(len(keyword_matching_score)), key=lambda k: keyword_matching_score[k],
            #                                reverse=True)
            # top_score = keyword_matching_score[sorted_matching_index[0]]
            media_subset = []
            for i in v:
                media_subset.append(is_media_subset(i))



def is_media_subset(resource):
    media_subset = ['work', 'person', 'art', 'creative work', 'television show', 'MusicGroup', 'band', 'film', 'book']
    categories = get_categories2(resource)
    categories = [i['keywords'] for i in categories]
    for c in categories:
        if c in media_subset:
            return 1
    return 0




def distill_docs(es_docs):
    # validate wiki page title and dbpedia entities
    pass


def search_medias(phrase):
    # filter duplicated entity linking, TV, book, film, series, band, song, album
    pass

def search_extended_URIs(sub_obj_l):
    docs = []
    for sub_obj in sub_obj_l:
        obj = sub_obj[1]
        wiki_links_obj = get_resource_wiki_page(obj)
        if wiki_links_obj is None or len(wiki_links_obj) < 1:
            continue
        for item in wiki_links_obj:
            possible_doc_id = item.split('/')[-1]
            verified_id_es = search_doc_id(possible_doc_id)
            for r_es in verified_id_es:
                if len(list(filter(lambda x: (x['id'] == r_es['id']), docs))) < 1:
                    docs.append({'id': r_es['id'], 'score': r_es['score'], 'phrases': [possible_doc_id.replace('_', ' ')]})
    return docs


def read_claim_context_graphs(dir):
    # config.RESULT_PATH / "sample_ss_graph_dev_test"
    data_dev = read_all_files(dir)
    # data_dev = read_json_rows(dir)
    cached_graph_d = dict()
    for i in data_dev:
        if 'claim_links' in i and len(i['claim_links']) > 0:
            cached_graph_d[i['id']] = i['claim_links']
        else:
            c_d = construct_subgraph_for_claim(i['claim'])
            if 'graph' in c_d:
                cached_graph_d[i['id']] = c_d['graph']
    return cached_graph_d


def get_context_dbpedia(cached_graphs, id):
    context = []
    object_urls = []
    if id in cached_graphs:
        tris = cached_graphs[id]
        for t in tris:
            if 'keywords' in t and 'subject' in t:
                entity = t['subject'].split('/')[-1]
                keywords = t['keywords']
                if isinstance(keywords, list) \
                        and len(list(filter(lambda x: (x['entity'] == entity
                                                       and sorted(x['keywords']) == sorted(keywords)), context))) < 1:
                    context.append({'entity': entity, 'keywords': keywords})
            if 'object' in t and "http://dbpedia.org/resource/" in t['object'] \
                    and len(list(filter(lambda x: (x[0] == t['subject'] and x[1] == t['object']), object_urls))) < 1:
                object_urls.append([t['subject'], t['object']])
    return context, object_urls


def retri_doc_and_update_item(item, context_dict=None):
    docs = retrieve_docs(item, context_dict)
    if len(docs) < 1:
        print("failed claim:", item.get('id'))
        item['predicted_docids'] = []
    else:
        item['predicted_docids'] = [j.get('id') for j in docs][:10]
    return item


def get_doc_ids_and_fever_score(in_file, out_file, top_k=10, eval=True, log_file=None, context_dict=None):
    if isinstance(in_file, list):
        d_list = in_file
    else:
        d_list = read_json_rows(in_file)

    print("total items: ", len(d_list))
    for i in tqdm(d_list):
        retri_doc_and_update_item(i, context_dict)
    # def retri_doc_and_update_item_with_context(data_l):
    #     retri_doc_and_update_item(data_l, context_dict)
    # thread_number = 2
    # thread_exe(retri_doc_and_update_item_with_context, iter(d_list), thread_number, "query wiki pages")
    save_intermidiate_results(d_list, out_file)
    if eval:
        eval_doc_preds(d_list, top_k, log_file)
    return d_list


def eval_doc_preds(doc_list, top_k, log_file):
    dt = get_current_time_str()
    print(fever_doc_only(doc_list, doc_list, max_evidence=top_k,
                         analysis_log=config.LOG_PATH / f"{dt}_doc_retri_no_hits.jsonl"))
    eval_mode = {'check_doc_id_correct': True, 'standard': False}
    if log_file is None:
        log_file = config.LOG_PATH / f"{dt}_analyze_doc_retri.log"
    print(fever_score(doc_list, doc_list, mode=eval_mode, max_evidence=top_k, error_analysis_file=log_file))


def rerun_failed_items(full_retri_doc, failed_list, updated_file_name):
    r_list = read_json_rows(full_retri_doc)
    for i in r_list:
        if i['id'] in failed_list:
            retri_doc_and_update_item(i)
    save_intermidiate_results(r_list, updated_file_name)


def run_claim_context_graph(data):
    bert_client = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=60000)
    for i in data:
        claim = i['claim']
        claim_gragh_dict = construct_subgraph_for_claim(claim, bert_client)
        claim_g = claim_gragh_dict['graph']
        print(claim)
        print(json.dumps(claim_g, indent=2))
        print("----------------------------")


if __name__ == '__main__':
    # context_graph_dict = read_claim_context_graphs(config.RESULT_PATH / "sample_ss_graph_test_pred")
    # context_graph_dict = read_claim_context_graphs(config.RESULT_PATH / "sample_ss_graph.jsonl")
    # r = search_entity_dbpedia("Giada at Home was only available on DVD.")
    # docs = read_json_rows(config.RESULT_PATH / 'doc_retri_no_hits.jsonl')
    # docs = read_json_rows(config.FEVER_TEST_JSONL)
    # for i in docs:
    #     if 'predicted_docids' in i:
    #         i.pop('predicted_docids')
    # get_doc_ids_and_fever_score(docs, config.RESULT_PATH / 'doc_redo_test.jsonl', context_dict=context_graph_dict)
    # pass
    # i = retrieve_docs("L.A. Reid has served as the president of a record label.")
    # print(i)
    # j = retrieve_docs("Trouble with the Curve")
    # print(j)
    # get_doc_ids_and_fever_score(config.LOG_PATH / "test.jsonl", config.RESULT_PATH / f"{get_current_time_str()}_train_doc_retrive.jsonl")
    # get_doc_ids_and_fever_score(config.FEVER_DEV_JSONL,
    #                             config.RESULT_PATH / f"{get_current_time_str()}_train_doc_retrive.jsonl", top_k=10)
    #
    # get_doc_ids_and_fever_score(config.FEVER_TRAIN_JSONL, config.DOC_RETRV_TRAIN)
    # get_doc_ids_and_fever_score(config.FEVER_DEV_JSONL, config.RESULT_PATH / f"doc_dev_{get_current_time_str()}.jsonl")
    # get_doc_ids_and_fever_score(config.FEVER_TEST_JSONL, config.DOC_RETRV_TEST, eval=False)
    # print(retrieve_docs("Brian Wilson was part of the Beach Boys."))
    # get_doc_ids_and_fever_score(config.FEVER_TEST_JSONL, config.DOC_RETRV_TEST / get_current_time_str())
    # a_list = read_json_rows(config.DOC_RETRV_DEV)
    # fever_doc_only(a_list, a_list, analysis_log=config.LOG_PATH / f"{get_current_time_str()}_doc_retri_no_hits_.jsonl")
    # rerun_failed_items(config.DOC_RETRV_TEST, [49649, 24225, 149500,202840,64863], config.RESULT_PATH / 'test_update.jsonl')
    data = read_json_rows(config.RESULT_PATH / 'doc_retri_no_hits.jsonl')
    run_claim_context_graph(data)
