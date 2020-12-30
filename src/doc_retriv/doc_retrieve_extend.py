from ES.es_search import search_and_merge, search_doc_id, search_and_merge2, search_and_merge3, merge_result, search_doc_id_and_keywords, search_doc_id_and_keywords_in_sentences
from utils.c_scorer import *
from utils.common import thread_exe
from utils.fever_db import *
from utils.file_loader import read_json_rows, get_current_time_str, read_all_files, save_and_append_results
from dbpedia_sampler.dbpedia_triple_linker import link_sentence
from dbpedia_sampler.dbpedia_virtuoso import get_resource_wiki_page
from dbpedia_sampler.sentence_util import get_phrases, get_phrases_and_nouns_merged
import difflib
from utils.text_clean import convert_brc
from dbpedia_sampler.dbpedia_subgraph import construct_subgraph_for_claim, construct_subgraph_for_candidate
from dbpedia_sampler.uri_util import isURI
from dbpedia_sampler.dbpedia_virtuoso import get_categories2
from bert_serving.client import BertClient
from doc_retriv.SentenceEvidence import *
from utils.check_sentences import Evidences
import copy
from typing import List
import itertools


# def retrieve_docs(example, context_dict=None):
#     claim = example['claim']
#     id = example['id']
#     result_context = []
#     result_uris = []
#     if context_dict is not None and len(context_dict) > 0:
#         context, object_urls = get_context_dbpedia(context_dict, id)
#         result_context = search_and_merge3(context)
#         result_uris = search_extended_URIs(object_urls)
#     nouns = get_phrases_and_nouns_merged(claim)
#     result_es = search_and_merge2(nouns)
#     result_dbpedia = search_entity_dbpedia(claim)
#     result_dbpedia.extend(result_uris)
#     result_dbpedia = merge_result(result_dbpedia)
#     result = merge_es_and_dbpedia(result_es, result_dbpedia, result_context)
#     if len(result) > 10:
#         result = result[:10]
#     return result


def prepare_candidate_doc1(data_l, out_filename: Path, log_filename: Path):
    for example in tqdm(data_l):
        claim = convert_brc(normalize(example['claim']))
        nouns = get_phrases_and_nouns_merged(claim)
        # 2. get ES page candidates -> candidate docs 1
        #  . BERT filter: claim VS candidate docs 1 sents -> candidate sentences 1
        candidate_docs_1 = search_and_merge2(nouns)
        if len(candidate_docs_1) < 1:
            print("failed claim:", example.get('id'))
            example['predicted_docids'] = []
            example['doc_and_line'] = []
        else:
            example['predicted_docids'] = [j.get('id') for j in candidate_docs_1][:10]
            example['doc_and_line'] = candidate_docs_1
    save_intermidiate_results(data_l, out_filename)
    eval_doc_preds(data_l, 10, log_filename)


def prepare_claim_graph(data_l, out_filename: Path, log_filename: Path):
    bc = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=60000)
    flush_save = []
    batch = 2
    flush_num = batch
    with tqdm(total=len(data_l), desc=f"constructing claim graph") as pbar:
        for idx, example in enumerate(data_l):
            claim = convert_brc(normalize(example['claim']))
            claim_dict = construct_subgraph_for_claim(claim, bc)
            claim_dict.pop('embedding')
            example['claim_dict'] = claim_dict
            flush_save.append(example)
            flush_num -= 1
            pbar.update(1)
            if flush_num == 0 or idx == (len(data_l) - 1):
                save_and_append_results(flush_save, idx + 1, out_filename, log_filename)
                flush_num = batch
                flush_save = []
    bc.close()


def prepare_candidate_doc2(data_original, data_with_claim_dict_l, out_filename: Path, log_filename: Path):
    flush_save = []
    batch = 10
    flush_num = batch
    with tqdm(total=len(data_with_claim_dict_l), desc=f"searching entity docs") as pbar:
        for idx, example in enumerate(data_with_claim_dict_l):
            # claim = convert_brc(normalize(example['claim']))
            claim_dict = example['claim_dict']
            claim_graph = claim_dict['graph']
            claim_triples = []
            for idx_t, t in enumerate(claim_graph):
                t['tri_id'] = idx_t
                try:
                    claim_triples.append(Triple(t))
                except Exception as e:
                    print(t)
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
                if len(list(filter(lambda x: (x['id'] == r_es['id']), docs))) < 1:
                    docs.append({'id': r_es['id'], 'score': r_es['score'], 'phrases': [resource['text']]})
        if len(docs) > 0:
            docs_all.update({resource_uri: docs})
    return docs_all


def merge_es_and_dbpedia(r_ents, r_tris, r_context):
    # all_ents_docs = [d for docs in r_ents.values for d in docs]
    # all_tri_docs = [d for docs in r_tris.values for d in docs]
    # all_context_docs = [d for docs in r_context.values for d in docs]
    # r_ents_ids = [i['id'] for i in r_ents]
    # r_tri_ids = [i['id'] for i in r_tris]
    # r_context_ids = [i['id'] for i in r_context]
    # for idx_i, i in enumerate(r_ents_ids):
    #     for idx_j, j in enumerate(r_tri_ids):
    #         if i == j:
    #             if len(r_ents[idx_i]['phrases']) > 1:
    #                 r_ents[idx_i]['score'] += r_tris[idx_j]['score']
    #             else:
    #                 p = r_tris[idx_j]['phrases'][0].lower()
    #                 doc_id = convert_brc(r_tris[idx_j]['id']).replace('_', ' ').lower()
    #                 ratio = difflib.SequenceMatcher(None, p, doc_id).ratio()
    #                 if ratio > 0.8:
    #                     r_ents[idx_i]['score'] += r_tris[idx_j]['score'] * 0.5
    merged = r_ents
    # for idx, i in enumerate(r_db_ids):
    #     if i not in r_es_ids:
    #         p = r_tris[idx]['phrases'][0].lower()
    #         doc_id = convert_brc(r_tris[idx]['id']).replace('_', ' ').lower()
    #         ratio = difflib.SequenceMatcher(None, p, doc_id).ratio()
    #         if ratio > 0.8:
    #             r_tris[idx]['score'] *= 2
    #         merged.append(r_tris[idx])
    #
    # merged_ids = [i['id'] for i in merged]
    # for idx_j, j in enumerate(r_context_ids):
    #     for idx_i, i in enumerate(merged_ids):
    #         if i == j:
    #             merged[idx_i]['score'] += r_context[idx_j]['score']
    #
    # for idx_j, j in enumerate(r_context_ids):
    #     if j not in merged_ids:
    #         merged.append(r_context[idx_j])
    #
    # merged.sort(key=lambda x: x.get('score'), reverse=True)
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


def prepare_triple_sentences(data_l, out_filename: Path, log_filename: Path):
    pass




def strategy_over_all(claim):
    # 1. ES search phrases
    nouns = get_phrases_and_nouns_merged(claim)
    # 2. get ES page candidates -> candidate docs 1
    #  . BERT filter: claim VS candidate docs 1 sents -> candidate sentences 1
    candidate_docs_1 = search_and_merge2(nouns)
    # candidate_sentences_1 = filter_bert_claim_vs_sents(claim, candidate_docs_1)
    claim_dict = construct_subgraph_for_claim(claim)
    claim_graph = claim_dict['graph']
    if len(claim_graph) > 0:
        # 1. ES search all linked entity page -> candidate docs 2
        candidate_docs_2 = search_entity_docs_for_triples(claim_graph)
        candidate_docs = candidate_docs_1 + candidate_docs_2
        # 2. BERT filter: claim VS candidate docs 2 sents -> candidate sentences 2
        candidate_sents_2 = filter_bert_claim_vs_sents(claim, candidate_docs)
        linked_triples = get_linked_triples(claim_graph)
        candidate_sentences_3 = []
        if len(linked_triples) > 0:
            # 3. ES filter: linked triples VS candidate docs 2 -> candidate sentences 3
            candidate_sentences_3 = search_triples_in_docs(linked_triples, candidate_docs)
        # 4. sort candidate sentences 2 + 3 -> candidate sentence 4
        candidate_sentences_4 = candidate_sents_2 + candidate_sentences_3
        isolated_phrases = claim_dict['no_relatives']
        if len(isolated_phrases) > 0:
            # 5. isolated_nodes candidate docs 2 sentences to sent_context_graph -> new entities
            # 6. ES search sent_context_graph new entity pages -> candidate docs 3
            # 7. BERT filter:  extended context triples VS candidate docs 3 sentences  -> candidate sentence 3
            # 8. aggregate envidence set -> candidate sentences 4
            candidate_sentence_set = strategy_one_hop(claim_dict, linked_triples, candidate_sentences_4)

    else:
        # *. BERT filter: claim VS candidate docs 1 sents -> candidate sentences 1
        # cannot extract context graph, return candidate sentences 1
        candidate_docs = candidate_docs_1
        candidate_sentences_1 = filter_bert_claim_vs_sents(claim, candidate_docs_1)
    pass


def merge_sentences_and_generate_evidence_set(linked_triples_with_sentences, candidate_sentences):
    evidence_set_from_triple = generate_triple_sentence_combination(linked_triples_with_sentences, [])
    evidence_set_from_sentences = generate_sentence_combination(candidate_sentences)
    new_evidence_set = copy.deepcopy(evidence_set_from_triple)
    for evid_s in evidence_set_from_sentences:
        for evid_t in evidence_set_from_triple:
            new_evidence = copy.deepcopy(evid_t)
            new_evidence.add_sent(evid_s)
            new_evidence_set.append(new_evidence)
    new_evidence_set.extend(evid_s)
    new_evidence_set = list(set(new_evidence_set))
    for e_s in new_evidence_set:
        for s in e_s.evidences_list:
            extend_sentences = candidate_sentences[s].extend_sentences
            for extend_s in extend_sentences:
                new_evidence = copy.deepcopy(e_s)
                new_evidence.add_sent(extend_s)
                new_evidence_set.append(new_evidence)
    new_evidence_set = list(set(new_evidence_set))
    return new_evidence_set


def generate_sentence_combination(list_of_sentences):
    new_evidence_set = set()
    for i in len(list_of_sentences):
        combination_set = itertools.combinations(list_of_sentences, i)
        evid_l = [Evidences(s.sid) for s in combination_set]
        new_evidence_set.update(set(evid_l))
    return list(new_evidence_set)


def generate_expand_evidence_from_hlinks(possible_evidence, hlink_docs):
    return []


def generate_triple_sentence_combination(list_of_triples, list_of_evidence: List[Evidences]):
    if len(list_of_triples) == 0:
        return list_of_evidence
    else:
        triple = list_of_triples.pop()
        if len(triple.sentences) == 0:
            new_evidence_l = list_of_evidence
        else:
            new_evidence_l = []
            for tri_sid in triple.sentences:
                if len(list_of_evidence) == 0:
                    new_evidence_l = [Evidences(tri_s) for tri_s in triple.sentences]
                    break
                else:
                    tmp_evidence_l = copy.deepcopy(list_of_evidence)
                    for e in tmp_evidence_l:
                        e.add_sent(tri_sid)
                    new_evidence_l.extend(tmp_evidence_l)
        return generate_triple_sentence_combination(list_of_triples, new_evidence_l)


def get_linked_triples(context_graph):
    tris = []
    for t in context_graph:
        if t['relation'] != '' and t['object'] != '':
            tris.append(Triple(t))
    return tris


def update_relative_hash(relative_hash:dict, connected_phrases):
    for i in connected_phrases:
        if i in relative_hash:
            relative_hash[i].extend(connected_phrases)
            relative_hash[i].remove(i)
            relative_hash[i] = list(set(relative_hash[i]))


def strategy_one_hop(claim_dict, linked_triples, candidate_sentences: List[SentenceEvidence]):
    # 5. isolated_nodes candidate docs 2 sentences to sent_context_graph -> new entities
    # 6. ES search sent_context_graph new entity pages -> candidate docs 3
    # 7. BERT filter:  extended context triples VS candidate docs 3 sentences  -> candidate sentence 3
    # 8. aggregate envidence set -> candidate sentences 4
    # extend_evidence_l = []
    # isolated_nodes_copy = copy.deepcopy(isolated_nodes)
    relative_hash = claim_dict['relative_hash']
    for c_s in candidate_sentences:
        sent_context_graph, extend_triples = construct_subgraph_for_candidate(claim_dict, c_s['lines'], c_s['doc_id'])
        if len(extend_triples) < 1:
            continue
        extended_docs = search_entity_docs_for_triples(extend_triples)
        extended_sentences = search_triples_in_docs(extend_triples, extended_docs)
        for e_s in extended_sentences:
            extend_phrases = e_s['phrases']
            ts_phrases = c_s['phrases']
            bridged_phrases = list(set(extend_phrases + ts_phrases))
            update_relative_hash(relative_hash, bridged_phrases)
            c_s.extend_sentences.append(e_s.sid)

    # sort out minimal evidence set, then BERT filter h_links
    possible_evidence_set = merge_sentences_and_generate_evidence_set(linked_triples, candidate_sentences)
    # BERT filter: (minimal set + h_link_sentence) VS claim
    no_relatives_found = []
    for i in relative_hash:
        if len(relative_hash[i]) == 0:
            no_relatives_found.append(i)
    if len(no_relatives_found) > 0:
        for e_s in possible_evidence_set:
            h_link_docs = get_hlink_docs(e_s)
            extend_hlink_e = generate_expand_evidence_from_hlinks(e_s, h_link_docs)
            possible_evidence_set.extend(extend_hlink_e)
    possible_evidence_set = list(set(possible_evidence_set))
    # bert NLI model
    pass


def get_hlink_docs(evidence):
    return []

def filter_bert_claim_vs_triplesents_and_hlinks(claim, sentence):
    pass
    # h_links = sentence['h_links']
    # h_links_docs = search_entity_docs(h_links)
    # for doc_id in h_links_docs:
    #     cur_r_list, cur_id_list = fever_db.get_all_sent_by_doc_id(cursor, doc_id, with_h_links=False)
    #     # Merging to data list and removing duplicate
    #     for i in range(len(cur_r_list)):
    #         if cur_id_list[i] in id_list:
    #             continue
    #         else:
    #             r_list.append(cur_r_list[i])
    #             id_list.append(cur_id_list[i])
    #
    # # assert len(id_list) == len(set(id_list))  # check duplicate
    # # assert len(r_list) == len(id_list)
    # if not (len(id_list) == len(set(id_list)) or len(r_list) == len(id_list)):
    #     utils.get_adv_print_func(err_log_f)
    #
    # zipped_s_id_list = list(zip(r_list, id_list))
    # # Sort using id
    # # sorted(evidences_set, key=lambda x: (x[0], x[1]))
    # zipped_s_id_list = sorted(zipped_s_id_list, key=lambda x: (x[1][0], x[1][1]))
    #
    # all_sent_list = convert_to_formatted_sent(zipped_s_id_list, all_evidence_set, contain_head=True,
    #                                           id_tokenized=True)


def filter_bert_claim_vs_sents(claim, docs):
    pass


def search_triples_in_docs(triples, docs:dict): #  list[Triple]
    # phrase match via ES
    possible_sentences = []
    for tri in triples:
        sentences = []
        subject_text = tri.text
        if len(docs) > 0 and subject_text in docs:
            resource_docs = docs[subject_text]
            for doc in resource_docs:
                doc_id = doc['id']
                tmp_sentences = search_doc_id_and_keywords_in_sentences(doc_id, tri['keywords'])
                if len(tmp_sentences) > 0:
                    for i in tmp_sentences:
                        i['tri_id'] = tri.tri_id
                        tri.sentences.add(i['sid'])
                    sentences.extend([SentenceEvidence(ts) for ts in tmp_sentences])
        if len(sentences) > 0:
            possible_sentences.extend(sentences)
    return possible_sentences


def search_triple_in_sentence(tri, doc_id):
    # phrase match via ES
    possible_sentences = []
    sentences = []
    tmp_sentences = search_doc_id_and_keywords_in_sentences(doc_id, tri['keywords'])
    if len(tmp_sentences) > 0:
        for i in tmp_sentences:
            i['tri_id'] = tri.tri_id
            tri.sentences.add(i['sid'])
        sentences.extend([SentenceEvidence(ts) for ts in tmp_sentences])
    if len(sentences) > 0:
        possible_sentences.extend(sentences)
    return possible_sentences


def search_entity_docs_for_triples(triples: List[Triple]):
    docs = dict()
    all_resources = []
    for tri in triples:
        if len(list(filter(lambda  x: x == tri.subject, all_resources))) < 1:
            all_resources.append(tri.subject)
        if "http://dbpedia.org/resource/" in tri.object and len(list(filter(lambda  x: x == tri.object, all_resources))) < 1:
            all_resources.append(tri.object)
    for idx, resource_uri in enumerate(all_resources):
        entity_pages = []
        wiki_links = get_resource_wiki_page(resource_uri)
        if wiki_links is None or len(wiki_links) < 1:
            continue
        for item in wiki_links:
            possible_doc_id = item.split('/')[-1]
            verified_id_es = search_doc_id(possible_doc_id)
            for r_es in verified_id_es:
                if len(list(filter(lambda x: (x['id'] == r_es['id']), entity_pages))) < 1:
                    entity_pages.append({'id': r_es['id'],
                                 'score': r_es['score'],
                                 'phrases': [tri.text] if len(tri.relatives) < 1 else tri.relatives})
        if len(entity_pages) > 0:
            docs.update({resource_uri: entity_pages})
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


def retri_doc_and_update_item(item, context_dict=None):
    docs = retrieve_docs(item, context_dict)
    if len(docs) < 1:
        print("failed claim:", item.get('id'))
        item['predicted_docids'] = []
    else:
        item['predicted_docids'] = [j.get('id') for j in docs][:10]
        item['doc_and_line'] = docs
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
    # print(generate_triple_sentence_combination([[1,2], [3,4], [5,6]], []))
    folder = config.RESULT_PATH / "extend_20201229"

    # prepare_candidate_doc1(data, folder / "es_doc_10.jsonl", folder / "es_doc_10.log")

    # data = read_json_rows(config.FEVER_DEV_JSONL)[0:10000]
    # prepare_claim_graph(data, folder / "claim_graph_10000.jsonl", folder / "claim_graph_10000.log")
    # data = read_json_rows(config.FEVER_DEV_JSONL)[10000:19998]
    # prepare_claim_graph(data, folder / "claim_graph_19998.jsonl", folder / "claim_graph_19998.log")

    data_original = read_json_rows(config.FEVER_DEV_JSONL)[0:100]
    data_context = read_json_rows(folder / "claim_graph_10000.jsonl")[0:100]
    prepare_candidate_doc2(data_original, data_context, folder / "entity_doc_100.jsonl", folder / "entity_doc_100.log")
