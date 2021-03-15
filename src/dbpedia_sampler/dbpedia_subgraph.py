import numpy as np
import sklearn.metrics.pairwise as pw

import log_util
from dbpedia_sampler import bert_similarity
from dbpedia_sampler import dbpedia_triple_linker
from utils.tokenizer_simple import get_dependent_verb, is_capitalized
from memory_profiler import profile
from bert_serving.client import BertClient
import gc
import json
# CANDIDATE_UP_TO = 150
# SCORE_CONFIDENCE = 0.6

log = log_util.get_logger("subgraph")


# def fill_relative_hash(relative_hash, graph):
#     for i in graph:
#         if 'relatives' in i and len(i['relatives']) > 1:
#             relatives = i['relatives']
#             if relatives[0] in relative_hash:
#                 relative_hash[relatives[0]].append(relatives[1])
#                 relative_hash[relatives[0]] = list(set(relative_hash[relatives[0]]))
#             if relatives[1] in relative_hash:
#                 relative_hash[relatives[1]].append(relatives[0])
#                 relative_hash[relatives[1]] = list(set(relative_hash[relatives[1]]))


# @profile
def construct_subgraph_for_sentence(sentence_text, extend_entity_docs=None,
                                    doc_title='',
                                    lookup_hash=None,
                                    embedding_hash=None,
                                    entities=[],
                                    nouns=[], hlinks=[]):
    not_linked_phrases_l, linked_phrases_l = dbpedia_triple_linker.link_sentence(sentence_text,
                                                                                 extend_entity_docs=extend_entity_docs,
                                                                                 doc_title=doc_title,
                                                                                 lookup_hash=lookup_hash,
                                                                                 entities=entities,
                                                                                 nouns=nouns
                                                                                 )

    linked_phrases = [i['text'] for i in linked_phrases_l]
    all_phrases = not_linked_phrases_l + linked_phrases
    all_uris = dict()
    for i in linked_phrases_l:
        uris = i['links']
        for u in uris:
            if u['URI'] not in all_uris:
                all_uris.update({u['URI']: u})
    if embedding_hash is None:
        embeddings_hash = {key: {'one_hop': [], 'keyword1': [], 'keyword2': []} for key in all_uris}
        embeddings_hash.update({p: [] for p in all_phrases})
    else:
        for key in all_uris:
            if key not in embedding_hash:
                embedding_hash.update({key: {'one_hop': [], 'keyword1': [], 'keyword2': []}})
        for p in all_phrases:
            if p not in embedding_hash:
                embedding_hash.update({p: []})
    lookup_hash = dict()
    for i in linked_phrases_l:
        lookup_hash[i['text']] = i

    verb_d = get_dependent_verb(sentence_text, all_phrases)
    r0 = dbpedia_triple_linker.filter_text_vs_one_hop(all_phrases, linked_phrases_l, embeddings_hash)
    r1 = dbpedia_triple_linker.filter_date_vs_property(not_linked_phrases_l, linked_phrases_l, verb_d)
    r2 = dbpedia_triple_linker.filter_resource_vs_keyword(linked_phrases_l)
    r3 = dbpedia_triple_linker.filter_verb_vs_one_hop(verb_d, linked_phrases_l, embeddings_hash)
    tmp_result = r0 + r1 + r2 + r3
    tmp_result = dbpedia_triple_linker.remove_duplicate_triples(tmp_result)
    sent_graph = dbpedia_triple_linker.filter_triples(tmp_result)
    sent_graph = dbpedia_triple_linker.remove_duplicate_triples(sent_graph)
    # only keyword-match on those no exact match triples
    # fill_relative_hash(relative_hash, sent_graph)
    # isolated_node = []
    # for i in relative_hash:
    #     if len(relative_hash[i]) == 0 and i in lookup_hash:
    #         isolated_node.append(lookup_hash[i])
    # r4 = dbpedia_triple_linker.filter_keyword_vs_keyword(isolated_node, linked_phrases_l, embeddings_hash, fuzzy_match=False, bc=bc)
    # for i in r4:
    #     if not dbpedia_triple_linker.does_tri_exit_in_list(i, merged_result):
    #         merged_result.append(i)
    # fill_relative_hash(relative_hash, r3)
    # tmp_no_relatives_found = []
    #     # has_relatives = []
    #     # no_relatives_found = []
    #     # for i in relative_hash:
    #     #     if len(relative_hash[i]) == 0:
    #     #         tmp_no_relatives_found.append(i)
    #     #     else:
    #     #         has_relatives.append(i)
    #     # for i in tmp_no_relatives_found:
    #     #     if len(list(filter(lambda x: i in x or x in i, has_relatives))) == 0:
    #     #         no_relatives_found.append(i)

    # isolated_nodes = []
    # for i in no_relatives_found:
    #     if i in lookup_hash:
    #         isolated_nodes.append(lookup_hash[i])
            # possible_links = lookup_hash[i]['links']
            # for t in possible_links:
            #     if not dbpedia_triple_linker.does_node_exit_in_list(t['URI'], merged_result):
            #         single_node = dict()
            #         single_node['subject'] = t['URI']
            #         single_node['object'] = ''
            #         single_node['relation'] = ''
            #         single_node['keywords'] = t['text']
            #         single_node['relatives'] = []
            #         single_node['text'] = t['text']
            #         single_node['exact_match'] = t['exact_match']
            # continue

        # for i in t['categories']:
        #     if not dbpedia_triple_linker.does_tri_exit_in_list(i, merged_result):
        #         merged_result.append(i)

    # print(json.dumps(merged_result, indent=4))
    claim_d = dict()
    claim_d['linked_phrases_l'] = linked_phrases_l
    claim_d['not_linked_phrases_l'] = not_linked_phrases_l
    claim_d['graph'] = sent_graph
    claim_d['embedding'] = embeddings_hash
    # claim_d['lookup_hash'] = lookup_hash
    # claim_d['no_relatives'] = no_relatives_found
    # claim_d['isolated_nodes'] = isolated_nodes
    # claim_d['relative_hash'] = relative_hash
    return claim_d


def get_isolated_nodes(no_relative_phrases, linked_phrases_l):
    isolated_nodes = []
    phrase2nodes = {i['text']: i for i in linked_phrases_l}
    for i in no_relative_phrases:
        if i in phrase2nodes:
            isolated_nodes.append(phrase2nodes[i])
    return isolated_nodes


@profile
def construct_subgraph_for_candidate(claim_dict_with_embedding, candidate_sent, doc_title=''):
    claim_linked_phrases_l = claim_dict_with_embedding['linked_phrases_l']
    embedding_hash = claim_dict_with_embedding['embedding']
    lookup_hash = dict()
    for i in claim_linked_phrases_l:
        lookup_hash[i['text']] = i

    graph_dict = construct_subgraph_for_sentence(candidate_sent, doc_title=doc_title, lookup_hash=lookup_hash, embedding_hash=embedding_hash)
    claim_linked_resources = claim_dict["linked_phrases_l"]
    claim_linked_phs = [i['text'] for i in claim_linked_resources]
    triples_extended_one_hop = extend_graph(graph_dict, claim_linked_resources, claim_linked_phs)
    graph = graph_dict['graph']
    return graph, triples_extended_one_hop


def construct_subgraph_for_candidate2(candidate_sent, doc_title='', additional_phrase=[], additional_resources=[], hlinks=[]):
    # sentence graph
    additional_linked_phrases_l = additional_resources
    lookup_hash = dict()
    for i in additional_linked_phrases_l:
        lookup_hash[i['text']] = i
    graph_dict = construct_subgraph_for_sentence(candidate_sent, doc_title=doc_title, lookup_hash=lookup_hash, hlinks=hlinks)
    triples_extended_one_hop = extend_graph(graph_dict, additional_resources, additional_phrase)
    graph = graph_dict['graph']
    return graph, triples_extended_one_hop


def extend_graph(graph_dict, additional_resources=[], additional_phrases=[]):
    embedding_hash = graph_dict['embedding']
    linked_phrase_l = graph_dict['linked_phrases_l']
    # 1. additional resource VS graph resource one_hop
    all_tmp_result = []
    all_linked_phrases = [i['text'] for i in linked_phrase_l]
    filtered_additional_phrases = [i for i in additional_phrases if i not in all_linked_phrases]
    filtered_additional_resources = [i for i in additional_resources if i['text'] not in all_linked_phrases]
    for a_res in filtered_additional_resources:
        tmp_result = dbpedia_triple_linker.filter_resource_vs_keyword2(a_res, linked_phrase_l)
        if len(tmp_result) > 0:
            all_tmp_result.extend(tmp_result)
    to_compare_linked_phrases_l = [i for i in linked_phrase_l if i['text'] not in additional_phrases]
    tmp_result = dbpedia_triple_linker.filter_text_vs_one_hop(filtered_additional_phrases, to_compare_linked_phrases_l, embedding_hash, threshold=0.6)
    if len(tmp_result) > 0:
        tmp_result = dbpedia_triple_linker.filter_triples(tmp_result, top_k=4)
        all_tmp_result.extend(tmp_result)
    # # 2. additional resource one_hop VS graph resource
    # graph_linked_phs = [i['text'] for i in linked_phrase_l if i['text'] not in additional_phrases]
    # filtered_additional_resources = [i for i in additional_resources if i['text'] not in all_linked_phrases]
    # tmp_result = dbpedia_triple_linker.filter_text_vs_one_hop(graph_linked_phs, filtered_additional_resources, embedding_hash, threshold=0.645)
    # if len(tmp_result) > 0:
    #     all_tmp_result.extend(tmp_result)
    # 3. additional resource one_hop VS graph resouce one_hop
    tmp_result = dbpedia_triple_linker.filter_keyword_vs_keyword(to_compare_linked_phrases_l, filtered_additional_resources, embedding_hash, fuzzy_match=False)
    if len(tmp_result) > 0:
        tmp_result = dbpedia_triple_linker.filter_triples(tmp_result, top_k=4)
        all_tmp_result.extend(tmp_result)
    extend_result = dbpedia_triple_linker.remove_duplicate_triples(all_tmp_result)
    return extend_result


def does_node_exist_in_graph(node_uri, graph):
    for tri in graph:
        if node_uri['URI'] == tri['subject'] or node_uri['URI'] == tri['object']:
            return True
    return False


def extend_connection_between_claim_and_sent(claim_isolated_nodes_origin, sent_graph, linked_phrases_l, embeddings_hash):
    claim_isolated_nodes = []
    sent_possible_links = linked_phrases_l
    for c_n in claim_isolated_nodes_origin:
        if not does_node_exist_in_graph(c_n, sent_graph):
            claim_isolated_nodes.append(c_n)
    if len(claim_isolated_nodes) < 1:
        return []

    # claim_isolated resource VS sent_resource_one_hop
    sent_filtered_one_hop = []
    for claim_node in claim_isolated_nodes:
        tmp_filtered = []
        c_phrase = claim_node['text']
        c_phrase_embedding = embeddings_hash[c_phrase]
        if len(c_phrase_embedding) < 1:
            continue
        for i in sent_possible_links:
            sent_i_links = i['links']
            if i['text'] in claim_node['text'] or claim_node['text'] in i['text']:
                continue
            same_link = False
            for l in sent_i_links:
                if claim_node['URI'] == l['URI']:
                    same_link = True
                    break
            if same_link:
                continue

            try:
                out = dbpedia_triple_linker.get_topk_similar_triples(c_phrase, i, embeddings_hash, top_k=3, threshold=0.6)
            except Exception as err:
                log.error(err)
                continue
            tmp_filtered.extend(out)
        tmp_filtered.sort(key=lambda k: k['score'], reverse=True)
        sent_filtered_one_hop.extend(tmp_filtered[:3])
    return sent_filtered_one_hop


def test_claim():
    cc1 = "Michelin Guides are published by George Lucas."
    s9 = "The term normally refers to the annually published Michelin Red Guide , the oldest " \
         "European hotel and restaurant reference guide , which awards Michelin stars for excellence " \
         "to a select few establishments ."

    for i in range(5):
        t = construct_subgraph_for_sentence(cc1)
        s = construct_subgraph_for_candidate(t, s9, "")
        del s
        del t
        gc.collect()
    return



if __name__ == '__main__':
    # ss1 = "Michelin Guides are published by George Lucas."
    # ss2 = "The term normally refers to the annually published Michelin Red Guide , the oldest European hotel and restaurant reference guide , which awards Michelin stars for excellence to a select few establishments ."
    # s6 = "Mozilla Firefox ( or simply Firefox ) is a free and open-source web browser developed by the Mozilla Foundation and its subsidiary the Mozilla Corporation ."
    # s7 = "Howard Eugene Johnson -LRB- 30 January 1915 -- 28 May 2000 -RRB- , better known as `` Stretch \'\' Johnson , was a tap dancer and social activist ."
    # s8 = "Magic Johnson was a tap dancer"
    # s8 = "Where the Heart Is ( 2000 film ) - The filmstars Natalie Portman , Stockard Channing , Ashley Judd , and Joan Cusack with supporting roles done by James Frain , Dylan Bruno , Keith David , and Sally Field ."
    # ss1 = "New Orleans Pelicans compete in the National Football Association"
    # ss2 = "The Pelicans compete in the National Basketball Association -LRB- NBA -RRB- as a member club of the league 's Western Conference Southwest Division ."
    # s9 = 'Tap Tap was a series of rhythm games by Tapulous available for the iOS of which several versions , both purchasable and free , have been produced .'
    # s8 = "T - Pain, His debut album , Rappa Ternt Sanga , was released in 2005 ."
    # s9 = "Chanhassen High School - Chanhassen had an enrollment of 1,576 students during the 2014-15 school year , with an 18:1 student teacher ratio ."
    # ss1 = "Giada at Home was only available on DVD ."
    ss1= 'Sheryl Lee has yet to appear in a film as of 2016.'
    # ss2 = "Giada at Home - It first aired on October 18 , 2008 on the Food Network ."
    # ss1 = "Cheese in the Trap (TV series) only stars animals."
    # ss1 = "Michelle Obama's husband was born in Kenya"
    # text = "Home for the Holidays stars the fourth stepchild of Charlie Chaplin"
    claim_dict = construct_subgraph_for_sentence(ss1)
    print(json.dumps(claim_dict['graph'], indent=4))
    # print(construct_subgraph_for_candidate(claim_dict, ss2, doc_title=''))
    # test_claim()
