from ES.es_search import search_doc_id, search_and_merge2, search_and_merge4, merge_result, search_doc_id_and_keywords, search_doc_id_and_keywords_in_sentences
from utils.c_scorer import *
from utils.common import thread_exe
from utils.fever_db import *
from utils.file_loader import read_json_rows, get_current_time_str, read_all_files, save_and_append_results
from dbpedia_sampler.dbpedia_triple_linker import lookup_doc_id
from dbpedia_sampler.dbpedia_virtuoso import get_resource_wiki_page
from dbpedia_sampler.sentence_util import get_ents_and_phrases, get_phrases_and_nouns_merged
import difflib
from utils.text_clean import convert_brc
from dbpedia_sampler.dbpedia_subgraph import construct_subgraph_for_claim
from dbpedia_sampler.dbpedia_virtuoso import get_categories2
from bert_serving.client import BertClient
from doc_retriv.SentenceEvidence import *
from utils.check_sentences import Evidences
import copy
from typing import List
from utils.tokenizer_simple import is_capitalized
import itertools


def prepare_candidate_doc1(data_l, out_filename: Path, log_filename: Path):
    for example in tqdm(data_l):
        candidate_docs_1 = prepare_candidate_es_for_example2(example)
        if len(candidate_docs_1) < 1:
            print("failed claim:", example.get('id'))
            example['predicted_docids'] = []
            example['doc_and_line'] = []
        else:
            example['predicted_docids'] = [j.get('id') for j in candidate_docs_1][:10]
            example['doc_and_line'] = candidate_docs_1
    save_intermidiate_results(data_l, out_filename)
    eval_doc_preds(data_l, 10, log_filename)
    return data_l


def prepare_candidate_es_for_example2(example):
    claim = convert_brc(normalize(example['claim']))
    entities, nouns = get_ents_and_phrases(claim)
    candidate_docs_1 = search_and_merge4(entities, nouns)
    return candidate_docs_1


def get_es_entity_links(doc_and_line):
    doc_and_line.sort(key=lambda x: x.get('score'), reverse=True)
    doc_and_line = doc_and_line[:15]
    all_phrases = list(set([p for d in doc_and_line for p in d['phrases']]))
    all_docids = [d['id'] for d in doc_and_line]
    phrase_to_doc_dict = dict()
    docid_to_phrases = {d['id']: d['phrases'] for d in doc_and_line}
    for doc in all_docids:
        for p in all_phrases:
            doc_id_clean = convert_brc(doc).replace("_", " ")
            if is_capitalized(p) and not is_date_or_number(p) and p.lower() in doc_id_clean.lower():
                if p in phrase_to_doc_dict:
                    phrase_to_doc_dict[p].append(doc)
                else:
                    phrase_to_doc_dict.update({p: [doc]})
    phrase_to_doc_links = dict()
    hit_phrases = []
    for p in phrase_to_doc_dict:
        doc_ids = phrase_to_doc_dict[p]
        filtered_doc_ids = []
        for doc in doc_ids:
            doc_id_clean = convert_brc(doc).replace("_", " ")
            if doc_id_clean.lower() == p.lower() or is_media(doc_id_clean) \
                    or (len(docid_to_phrases[doc]) > 1 and docid_to_phrases[doc] not in hit_phrases):
                filtered_doc_ids.append(doc)
                hit_phrases.append(docid_to_phrases[doc])
        linked_phrase = lookup_doc_id(p, filtered_doc_ids)
        if len(linked_phrase) > 0:
            phrase_to_doc_links.update({p: linked_phrase})
    return phrase_to_doc_links


def prepare_es_entity_links(es_data_l, output_file):
    es_enttiy_docs = []
    with tqdm(total=len(es_data_l), desc=f"preparing es entity docs") as pbar:
        for idx, example in enumerate(es_data_l):
            doc_and_line = example['doc_and_line']
            doc_dict = get_es_entity_links(doc_and_line)
            es_enttiy_docs.append({'id': example['id'], 'es_entity_docs': doc_dict})
            pbar.update(1)
    save_intermidiate_results(es_enttiy_docs, output_file)
    return es_enttiy_docs


def prepare_claim_graph(data_l, data_with_es, out_filename: Path, log_filename: Path):
    bc = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=60000)
    flush_save = []
    batch = 20
    flush_num = batch
    with tqdm(total=len(data_l), desc=f"constructing claim graph") as pbar:
        for idx, example in enumerate(data_l):
            example_with_es = data_with_es[idx]
            if data_with_es is not None:
                extend_entity_docs = example_with_es['es_entity_docs']
            else:
                extend_entity_docs = None
            example = prepare_claim_graph_for_example(example, extend_entity_docs=extend_entity_docs, bc=bc)
            flush_save.append(example)
            flush_num -= 1
            pbar.update(1)
            if flush_num == 0 or idx == (len(data_l) - 1):
                save_and_append_results(flush_save, idx + 1, out_filename, log_filename)
                flush_num = batch
                flush_save = []
    bc.close()
    print("done with claim graph.")


def prepare_claim_graph_for_example(example, extend_entity_docs=None, bc: BertClient=None):
     claim = convert_brc(normalize(example['claim']))
     claim_dict = construct_subgraph_for_claim(claim, extend_entity_docs=extend_entity_docs, bc=bc)
     claim_dict.pop('embedding')
     example['claim_dict'] = claim_dict
     return example


def prepare_candidate_doc2(data_original, data_with_claim_dict_l, out_filename: Path, log_filename: Path):
    flush_save = []
    batch = 10
    flush_num = batch
    with tqdm(total=len(data_with_claim_dict_l), desc=f"searching entity docs") as pbar:
        for idx, example in enumerate(data_with_claim_dict_l):
            candidate_docs_2 = prepare_candidate2_example(example)
            if len(candidate_docs_2) < 1:
                print("failed claim:", example.get('id'))
                data_original[idx]['resource_docs'] = {}
            else:
                data_original[idx]['resource_docs'] = candidate_docs_2
            flush_save.append(data_original[idx])
            flush_num -= 1
            pbar.update(1)
            if flush_num == 0 or idx == (len(data_with_claim_dict_l) - 1):
                save_and_append_results(flush_save, idx + 1, out_filename, log_filename)
                flush_num = batch
                flush_save = []


def prepare_candidate2_example(example):
    claim_dict = example['claim_dict']
    claim_graph = claim_dict['graph']
    claim_triples = []
    for idx_t, t in enumerate(claim_graph):
        t['tri_id'] = idx_t
        try:
            claim_triples.append(Triple(t))
        except Exception as e:
            print(t)
            print(f"id:{example['id']}")
            raise e
    linked_l = claim_dict['linked_phrases_l']
    all_resources = []
    for p in linked_l:
        for link in p['links']:
            if len(list(filter(lambda x: link['URI'] == x, all_resources))) < 1:
                all_resources.append(link)
    entity_candidate_docs = search_entity_docs(all_resources)
    triple_candidate_docs = search_entity_docs_for_triples(claim_triples)
    candidate_docs_2 = merge_entity_and_triple_docs(entity_candidate_docs, triple_candidate_docs)
    return candidate_docs_2


def prepare_candidate_docs(original_data, es_data, entity_data, out_filename: Path, log_filename: Path):
    error_items = []
    with tqdm(total=len(original_data), desc=f"preparing candidate docs") as pbar:
        for idx, example in enumerate(original_data):
            es_r = es_data[idx]['doc_and_line']
            ent_r = entity_data[idx]['resource_docs']
            merged = merge_es_and_entity_docs(es_r, ent_r)
            if len(merged) == 0:
                error_items.append(example)
            example['candidate_docs'] = merged
            example['predicted_docids'] = [j.get('id') for j in merged][:10]
            pbar.update(1)
        save_intermidiate_results(original_data, out_filename)
        save_intermidiate_results(error_items, log_filename)
        eval_doc_preds(original_data, 10, log_filename)


MEDIA = ['tv', 'film', 'book', 'novel', 'band', 'album', 'music', 'series', 'poem', 'song', 'advertisement',
             'company',
             'episode', 'season', 'animator', 'actor', 'singer', 'writer', 'drama', 'character']
def is_media(resource_record):
    if resource_record is not None and any([j in resource_record.lower() for j in MEDIA]):
        return True
    else:
        return False


def merge_es_and_entity_docs(r_es, r_ents):
    def merge_doc_l(doc_l):
        merge_docs = []
        for d in doc_l:
            if len(list(filter(lambda x: x['id'] == d['id'], merge_docs))) == 0:
                merge_docs.append(d)
            else:
                for m in merge_docs:
                    if m['id'] == d['id'] and m['score'] < d['score']:
                        merge_docs.remove(m)
                        merge_docs.append(d)
                        break
        return merge_docs

    all_doc_items_with_dup = [d for docs in r_ents.values() for d in docs]
    all_ents_docs = merge_doc_l(all_doc_items_with_dup)
    all_es_docs = merge_doc_l(r_es)
    r_ents_ids = [i['id'] for i in all_ents_docs]
    r_es_ids = [i['id'] for i in all_es_docs]
    for idx_i, i in enumerate(r_es_ids):
        for idx_j, j in enumerate(r_ents_ids):
            if i == j:
                if len(r_es[idx_i]['phrases']) > 1:
                    all_es_docs[idx_i]['score'] += all_ents_docs[idx_j]['score']
                else:
                    p = all_ents_docs[idx_j]['phrases'][0].lower()
                    doc_id = convert_brc(all_ents_docs[idx_j]['id']).replace('_', ' ').lower()
                    ratio = difflib.SequenceMatcher(None, p, doc_id).ratio()
                    if ratio > 0.8:
                        r_es[idx_i]['score'] += all_ents_docs[idx_j]['score'] * 0.5
    merged = all_es_docs
    for idx, i in enumerate(r_ents_ids):
        if i not in r_es_ids:
            phrases = all_ents_docs[idx]['phrases']
            if len(phrases) == 1:
                p = all_ents_docs[idx]['phrases'][0].lower()
                doc_id = convert_brc(i).replace('_', ' ').lower()
                ratio = difflib.SequenceMatcher(None, p, doc_id).ratio()
                if ratio >= 0.8 or is_media(doc_id):
                    all_ents_docs[idx]['score'] *= 1.5
            else:  # from triple
                all_ents_docs[idx]['score'] *= 2
            merged.append(all_ents_docs[idx])
    merged.sort(key=lambda x: x.get('score'), reverse=True)
    return merged


def merge_entity_and_triple_docs(entity_docs, triple_docs):
    merged_dict = entity_docs
    all_tri_docs = [d for docs in triple_docs.values() for d in docs]
    all_tri_doc_ids = dict()
    for d in all_tri_docs:
        if d['id'] not in all_tri_doc_ids:
            all_tri_doc_ids.update({d['id']: d['score']})
    for docs in entity_docs.values():
        for d in docs:
            if d['id'] in all_tri_doc_ids:
                d['score'] += 0.5 * all_tri_doc_ids[d['id']]

    for i in triple_docs:
        if i not in entity_docs:
            merged_dict.update({i: triple_docs[i]})
    return merged_dict


def search_entity_docs(resources):
    docs_all = dict()
    for resource in resources:
        docs = []
        resource_uri = resource['URI']
        wiki_links = get_resource_wiki_page(resource_uri)
        if wiki_links is None or len(wiki_links) < 1:
            continue
        for item in wiki_links:
            possible_doc_id = item.split('/')[-1]
            verified_id_es = search_doc_id(possible_doc_id)
            for r_es in verified_id_es:
                if convert_brc(r_es['id']) == possible_doc_id and len(list(filter(lambda x: (x['id'] == r_es['id']), docs))) < 1:
                    docs.append({'id': r_es['id'], 'score': r_es['score'], 'phrases': [resource['text']]})
        if len(docs) > 0:
            docs_all.update({resource_uri: docs})
    return docs_all


def search_entity_docs_for_triples(triples: List[Triple]):
    docs = dict()
    all_resources = dict()
    for tri in triples:
        if len(list(filter(lambda x: x == tri.subject, all_resources))) < 1:
            all_resources.update({tri.subject: [tri.text] if len(tri.relatives) < 1 else tri.relatives})
        if "http://dbpedia.org/resource/" in tri.object and len(list(filter(lambda  x: x == tri.object, all_resources))) < 1:
            all_resources.update({tri.object: [tri.text] if len(tri.relatives) < 1 else tri.relatives})
    for resource_uri in all_resources.keys():
        entity_pages = []
        wiki_links = get_resource_wiki_page(normalize(resource_uri))
        if wiki_links is None or len(wiki_links) < 1:
            continue
        for item in wiki_links:
            possible_doc_id = item.split('/')[-1]
            verified_id_es = search_doc_id(possible_doc_id)
            for r_es in verified_id_es:
                if convert_brc(r_es['id']) == possible_doc_id and len(list(filter(lambda x: (x['id'] == r_es['id']), entity_pages))) < 1:
                    entity_pages.append({'id': r_es['id'],
                                 'score': r_es['score'],
                                 'phrases': all_resources[resource_uri]})
        if len(entity_pages) > 0:
            docs.update({resource_uri: entity_pages})
    return docs


def is_media_subset(resource):
    media_subset = ['work', 'person', 'art', 'creative work', 'television show', 'MusicGroup', 'band', 'film', 'book']
    categories = get_categories2(resource)
    categories = [i['keywords'] for i in categories]
    for c in categories:
        if c in media_subset:
            return 1
    return 0


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


def eval_doc_preds(doc_list, top_k, log_file):
    if 'evidence' not in doc_list[0]:
        return

    dt = get_current_time_str()
    print(fever_doc_only(doc_list, doc_list, max_evidence=top_k,
                         analysis_log=config.LOG_PATH / f"{dt}_doc_retri_no_hits.jsonl"))
    eval_mode = {'check_doc_id_correct': True, 'standard': False}
    if log_file is None:
        log_file = config.LOG_PATH / f"{dt}_analyze_doc_retri.log"
    print(fever_score(doc_list, doc_list, mode=eval_mode, max_evidence=top_k, error_analysis_file=log_file))


def run_claim_context_graph(data):
    bert_client = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=60000)
    for i in data:
        claim = i['claim']
        claim_gragh_dict = construct_subgraph_for_claim(claim, bert_client)
        claim_g = claim_gragh_dict['graph']
        print(claim)
        print(json.dumps(claim_g, indent=2))
        print("----------------------------")


def rerun_failed_graph(folder):
    # failed_items = [20986, 217205, 149990, 84858, 25545, 4705, 217187,
    #                 182050,88781, 10688, 206031, 182033,
    #                 96740,182032, 134670, 88589,182051, 23588, 10324, 206024, 156889]
    failed_items = [84858, 25545, 10688, 206031, 96740,
                    10324, 156889]
    # data_original = read_json_rows(config.FEVER_DEV_JSONL)[0:10000]
    # data_context = read_json_rows(folder / "claim_graph_19998.jsonl")
    # data_entity = read_json_rows(folder / "entity_doc_19998.jsonl")
    data_context = read_json_rows(folder / "claim_graph_10000.jsonl")
    data_context.extend(read_json_rows(folder / "claim_graph_19998.jsonl"))
    data_entity = read_json_rows(folder / "entity_doc.jsonl")

    bc = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=60000)
    for idx, i in enumerate(data_context):
        if i['id'] in failed_items:
            claim = convert_brc(normalize(i['claim']))
            claim_dict = construct_subgraph_for_claim(claim, bc)
            claim_dict.pop('embedding')
            i['claim_dict'] = claim_dict
            # print(json.dumps(i['claim_dict'].get('linked_phrases_l'), indent=2))
            candidate_docs_2 = prepare_candidate2_example(i)
            if len(candidate_docs_2) < 1:
                print("failed claim:", i.get('id'))

                data_entity[idx]['resource_docs'] = {}
            else:
                data_entity[idx]['resource_docs'] = candidate_docs_2
    # save_intermidiate_results(data_context, folder / "rerun_claim_graph_19998.jsonl")
    # save_intermidiate_results(data_entity, folder / "rerun_entity_doc_19998.jsonl")
    save_intermidiate_results(data_context, folder / "rerun_claim_graph.jsonl")
    save_intermidiate_results(data_entity, folder / "rerun_entity_doc.jsonl")


def redo_example_docs(error_data, log_filename):
    bc = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=60000)
    for example in tqdm(error_data):
        es_doc_and_lines = prepare_candidate_es_for_example2(example)
        entity_docs = get_es_entity_links(es_doc_and_lines)
        graph_data_example = prepare_claim_graph_for_example(example, extend_entity_docs=entity_docs, bc=bc)
        ent_resource_docs = prepare_candidate2_example(graph_data_example)
        merged = merge_es_and_entity_docs(es_doc_and_lines, ent_resource_docs)
        example['candidate_docs'] = merged
        example['predicted_docids'] = [j.get('id') for j in merged][:10]
    eval_doc_preds(error_data, 10, log_filename)


def do_testset_es(folder):
    data = read_json_rows(config.FEVER_TEST_JSONL)
    prepare_candidate_doc1(data, folder / "test_es_doc_10.jsonl", folder / "test_es_doc_10.log")


def do_testset_graph1(folder):
    data = read_json_rows(config.FEVER_TEST_JSONL)[0:10000]
    prepare_claim_graph(data, folder / "test_claim_graph_10000.jsonl", folder / "test_claim_graph_10000.log")
    # data = read_json_rows(config.FEVER_TEST_JSONL)[10000:19998]
    # prepare_claim_graph(data, folder / "test_claim_graph_19998.jsonl", folder / "test_claim_graph_19998.log")


def do_testset_graph2(folder):
    data = read_json_rows(config.FEVER_TEST_JSONL)[10000:19998]
    prepare_claim_graph(data, folder / "test_claim_graph_19998.jsonl", folder / "test_claim_graph_19998.log")


def do_dev_set_with_es_entity():
    folder = config.RESULT_PATH / "hardset2021"
    original_data = read_json_rows(folder / "dev_has_multi_doc_evidence.jsonl")
    # data_with_es = prepare_candidate_doc1(original_data, folder / "es_doc_10.jsonl", folder / "es_doc_10.log")

    # data_with_es = read_json_rows(folder / "es_doc_10.log")
    # data_with_es_entities = prepare_es_entity_links(data_with_es, folder / "es_entity_docs.jsonl")

    data_with_es_entities = read_json_rows(folder / "es_entity_docs.jsonl")
    # assert(len(data_with_es_entities) == len(data_with_es))
    # data = read_json_rows(config.FEVER_DEV_JSONL)
    prepare_claim_graph(original_data, data_with_es_entities, folder / "claim_graph.jsonl", folder / "claim_graph.log")
    #
    # original_data = read_json_rows(config.FEVER_DEV_JSONL)
    data_context = read_json_rows(folder / "claim_graph.jsonl")
    # # data_context.extend(read_json_rows(folder / "claim_graph_19998.jsonl"))
    # assert (len(data_original) == len(data_context))
    prepare_candidate_doc2(original_data, data_context, folder / "entity_doc.jsonl", folder / "entity_doc.log")
    #
    # data_original = read_json_rows(config.FEVER_DEV_JSONL)
    es_data = read_json_rows(folder / "es_doc_10.jsonl")
    ent_data = read_json_rows(folder / "entity_doc.jsonl")
    assert (len(es_data) == len(original_data) and (len(ent_data) == len(original_data)))
    prepare_candidate_docs(original_data, es_data, ent_data, folder / "candidate_docs.jsonl",
                           folder / "candidate_docs.log")


def do_dev_set():
    folder = config.RESULT_PATH / "extend_20210115"

    original_data = read_json_rows(config.FEVER_DEV_JSONL)
    prepare_candidate_doc1(original_data, folder / "es_doc_10.jsonl", folder / "es_doc_10.log")

    # data = read_json_rows(config.FEVER_DEV_JSONL)
    # prepare_claim_graph(data, None, folder / "claim_graph.jsonl", folder / "claim_graph.log")
    #
    # data_original = read_json_rows(config.FEVER_DEV_JSONL)
    # data_context = read_json_rows(folder / "claim_graph.jsonl")
    # assert(len(data_original) == len(data_context))
    # prepare_candidate_doc2(data_original, data_context, folder / "entity_doc.jsonl", folder / "entity_doc.log")

    # data_original = read_json_rows(config.FEVER_DEV_JSONL)
    # es_data = read_json_rows(folder / "es_doc_10.jsonl")
    # ent_data = read_json_rows(folder / "entity_doc.jsonl")
    # assert(len(es_data) == len(data_original) and (len(ent_data) == len(data_original)))
    # prepare_candidate_docs(data_original, es_data, ent_data, folder / "candidate_docs.jsonl", folder / "candidate_docs.log")
#

def run_dev_failed_docs():
    folder = config.RESULT_PATH / "extend_20210107"

    # rerun_failed_graph(folder)
    error_data = read_json_rows(folder / 'candidate_docs.log')
    redo_example_docs(error_data, folder / "redo09.log")

    # data = read_json_rows(config.FEVER_DEV_JSONL)[2:3]
    # prepare_claim_graph(data, folder / "claim_graph_2.jsonl", folder / "claim_graph_2.log")
    # es_data = read_json_rows(folder / "es_doc_10.jsonl")
    # eval_doc_preds(es_data, 5, config.LOG_PATH / 'doc_eval_1231')


if __name__ == '__main__':
    # data = read_json_rows(config.DATA_ROOT /"dev_has_multi_evidence.jsonl")
    # redo_example_docs(data, config.LOG_PATH / "test.log")
    # folder = config.RESULT_PATH / "extend_test_20210102"
    # do_testset_es(folder)
    do_dev_set_with_es_entity()
    # do_dev_set()
    # original_data = read_json_rows(config.RESULT_PATH / 'errors/es_doc_10.log')
    # prepare_candidate_doc1(original_data, config.RESULT_PATH / 'errors/es_doc_16.jsonl', config.RESULT_PATH / 'errors/es_doc_16.log')
