import numpy as np
import sklearn.metrics.pairwise as pw

import log_util
from dbpedia_sampler import bert_similarity
from dbpedia_sampler import dbpedia_triple_linker
from utils.tokenizer_simple import get_dependent_verb, is_capitalized
from memory_profiler import profile
from bert_serving.client import BertClient
import gc

# CANDIDATE_UP_TO = 150
# SCORE_CONFIDENCE = 0.6

log = log_util.get_logger("subgraph")


def fill_relative_hash(relative_hash, graph):
    for i in graph:
        if 'relatives' in i and len(i['relatives']) > 1:
            relatives = i['relatives']
            if relatives[0] in relative_hash:
                relative_hash[relatives[0]].append(relatives[1])
                relative_hash[relatives[0]] = list(set(relative_hash[relatives[0]]))
            if relatives[1] in relative_hash:
                relative_hash[relatives[1]].append(relatives[0])
                relative_hash[relatives[1]] = list(set(relative_hash[relatives[1]]))


# @profile
def construct_subgraph_for_claim(claim_text, extend_entity_docs=None, bc:BertClient=None):
    not_linked_phrases_l, linked_phrases_l = dbpedia_triple_linker.link_sentence(claim_text, extend_entity_docs=extend_entity_docs, doc_title='')
    linked_phrases = [i['text'] for i in linked_phrases_l]
    all_phrases = not_linked_phrases_l + linked_phrases
    relative_hash = {key: [] for key in all_phrases}
    all_uris = dict()
    for i in linked_phrases_l:
        uris = i['links']
        for u in uris:
            if u['URI'] not in all_uris:
                all_uris.update({u['URI']: u})
    embeddings_hash = {key: {'one_hop': [], 'keyword1': [], 'keyword2': []} for key in all_uris}
    embeddings_hash.update({p: [] for p in all_phrases})
    embeddings_hash.update({'all_phrases': []})
    lookup_hash = dict()
    for i in linked_phrases_l:
        lookup_hash[i['text']] = i

    verb_d = get_dependent_verb(claim_text, all_phrases)
    r0 = dbpedia_triple_linker.filter_text_vs_one_hop(all_phrases, linked_phrases_l, embeddings_hash, bc=bc)
    r1 = dbpedia_triple_linker.filter_date_vs_property(not_linked_phrases_l, linked_phrases_l, verb_d)
    r2 = dbpedia_triple_linker.filter_resource_vs_keyword(linked_phrases_l)
    r3 = dbpedia_triple_linker.filter_verb_vs_one_hop(verb_d, linked_phrases_l, embeddings_hash, bc=bc)
    merged_result = []
    for i in r0 + r1 + r2 + r3:
        if not dbpedia_triple_linker.does_tri_exit_in_list(i, merged_result):
            merged_result.append(i)
    # only keyword-match on those no exact match triples
    fill_relative_hash(relative_hash, merged_result)
    # isolated_node = []
    # for i in relative_hash:
    #     if len(relative_hash[i]) == 0 and i in lookup_hash:
    #         isolated_node.append(lookup_hash[i])
    # r4 = dbpedia_triple_linker.filter_keyword_vs_keyword(isolated_node, linked_phrases_l, embeddings_hash, fuzzy_match=False, bc=bc)
    # for i in r4:
    #     if not dbpedia_triple_linker.does_tri_exit_in_list(i, merged_result):
    #         merged_result.append(i)
    # fill_relative_hash(relative_hash, r3)
    no_relatives_found = []
    for i in relative_hash:
        if len(relative_hash[i]) == 0:
            no_relatives_found.append(i)

    for i in no_relatives_found:
    #     if len(t['categories']) < 1 or len(t['categories']) > 20:
        if i in lookup_hash:
            possible_links = lookup_hash[i]['links']
            for t in possible_links:
                if not dbpedia_triple_linker.does_node_exit_in_list(t['URI'], merged_result):
                    single_node = dict()
                    single_node['subject'] = t['URI']
                    single_node['object'] = ''
                    single_node['relation'] = ''
                    single_node['keywords'] = t['text']
                    single_node['relatives'] = []
                    single_node['text'] = t['text']
                    single_node['exact_match'] = t['exact_match']
                    merged_result.append(single_node)
            # continue

        # for i in t['categories']:
        #     if not dbpedia_triple_linker.does_tri_exit_in_list(i, merged_result):
        #         merged_result.append(i)

    # print(json.dumps(merged_result, indent=4))
    claim_d = dict()
    claim_d['linked_phrases_l'] = linked_phrases_l
    claim_d['not_linked_phrases_l'] = not_linked_phrases_l
    claim_d['graph'] = merged_result
    claim_d['embedding'] = embeddings_hash
    claim_d['lookup_hash'] = lookup_hash
    claim_d['no_relatives'] = no_relatives_found
    claim_d['relative_hash'] = relative_hash
    return claim_d


# @profile
def construct_subgraph_for_candidate(claim_dict, candidate_sent, doc_title='', bc:BertClient=None):
    claim_linked_phrases_l = claim_dict['linked_phrases_l']
    claim_graph = claim_dict['graph']
    embeddings_hash = claim_dict['embedding']

    # sentence graph
    not_linked_phrases_l, linked_phrases_l = dbpedia_triple_linker.link_sentence(candidate_sent,
                                                                                 doc_title=doc_title,
                                                                                 lookup_hash=claim_dict['lookup_hash'])

    sent_linked_phrases = [i['text'] for i in linked_phrases_l]
    all_phrases = not_linked_phrases_l + sent_linked_phrases
    relative_hash = {key: set() for key in sent_linked_phrases}
    all_uris_dict = dict()
    for i in linked_phrases_l:
        uris = i['links']
        for u in uris:
            if u['URI'] not in all_uris_dict:
                all_uris_dict.update({u['URI']: u})

    claim_all_uris = dict()
    for i in claim_linked_phrases_l:
        claim_uris = i['links']
        for u in claim_uris:
            if u['URI'] not in claim_all_uris:
                claim_all_uris.update({u['URI']: u})

    for i in all_uris_dict.keys():
        if i not in embeddings_hash:
            embeddings_hash.update({i: {'one_hop': []}})
    for p in all_phrases:
        if p not in embeddings_hash:
            embeddings_hash.update({p: []})
    embeddings_hash.update({'not_linked_phrases_l': [], 'linked_phrases_l': []})
    for i in linked_phrases_l:
        if not i['text'] in claim_dict['lookup_hash']:
            claim_dict['lookup_hash'][i['text']] = i

    verb_d = get_dependent_verb(candidate_sent, all_phrases)
    sent_graph = []
    r0 = dbpedia_triple_linker.filter_text_vs_one_hop(all_phrases, linked_phrases_l, embeddings_hash, bc=bc)
    r1 = dbpedia_triple_linker.filter_date_vs_property(not_linked_phrases_l, linked_phrases_l, verb_d)
    r2 = dbpedia_triple_linker.filter_resource_vs_keyword(linked_phrases_l)
    r3 = dbpedia_triple_linker.filter_verb_vs_one_hop(verb_d, linked_phrases_l, embeddings_hash, bc=bc)
    # only keyword-match on those no exact match triples

    for i in r0 + r1 + r2 + r3:
        if not dbpedia_triple_linker.does_tri_exit_in_list(i, sent_graph):
            sent_graph.append(i)
    fill_relative_hash(relative_hash, sent_graph)
    no_relatives_found = []
    for i in relative_hash:
        if len(relative_hash[i]) == 0 and i in claim_dict['lookup_hash']:
            no_relatives_found.append(claim_dict['lookup_hash'][i])

    # r3 = dbpedia_triple_linker.filter_keyword_vs_keyword(no_relatives_found, embeddings_hash, fuzzy_match=False, bc=bc)
    # for i in r3:
    #     if not dbpedia_triple_linker.does_tri_exit_in_list(i, sent_graph):
    #         sent_graph.append(i)
    # fill_relative_hash(relative_hash, r3)
    claim_isolated_nodes = []
    for tri in claim_graph:
        if tri['relation'] == "" and tri['object'] == "":
            claim_isolated_nodes.append(claim_all_uris[tri['subject']])
    sent_extended_one_hop = extend_connection_between_claim_and_sent(claim_isolated_nodes, sent_graph, linked_phrases_l, embeddings_hash, bc=bc)
    extend_triples = []
    for i in sent_extended_one_hop:
        if not dbpedia_triple_linker.does_tri_exit_in_list(i, sent_graph):
            sent_graph.append(i)
            extend_triples.append(i)

    return sent_graph, extend_triples


def get_isolated_nodes(claim_dict):
    claim_graph = claim_dict['graph']
    claim_linked_phrases_l = claim_dict['linked_phrases_l']
    claim_all_uris = dict()
    for i in claim_linked_phrases_l:
        claim_uris = i['links']
        for u in claim_uris:
            if u['URI'] not in claim_all_uris:
                claim_all_uris.update({u['URI']: u})
    claim_isolated_nodes = []
    for tri in claim_graph:
        if tri['relation'] == "" and tri['object'] == "":
            claim_isolated_nodes.append(claim_all_uris[tri['subject']])
    return claim_isolated_nodes


def does_node_exist_in_graph(node_uri, graph):
    for tri in graph:
        if node_uri['URI'] == tri['subject'] or node_uri['URI'] == tri['object']:
            return True
    return False


def extend_connection_between_claim_and_sent(claim_isolated_nodes_origin, sent_graph, linked_phrases_l, embeddings_hash, bc:BertClient=None):
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
                out = dbpedia_triple_linker.get_topk_similar_triples(c_phrase, i, embeddings_hash, top_k=3, threshold=0.6, bc=bc)
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
        t = construct_subgraph_for_claim(cc1)
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
    # ss2 = "Giada at Home - It first aired on October 18 , 2008 on the Food Network ."
    # ss1 = "Cheese in the Trap (TV series) only stars animals."
    # ss1 = "Michelle Obama's husband was born in Kenya"
    text = "Home for the Holidays stars the fourth stepchild of Charlie Chaplin"
    claim_dict = construct_subgraph_for_claim(text)
    # print(construct_subgraph_for_candidate(claim_dict, ss2, doc_title=''))
    # test_claim()
