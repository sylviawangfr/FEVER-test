import numpy as np
import sklearn.metrics.pairwise as pw

import log_util
from dbpedia_sampler import bert_similarity
from dbpedia_sampler import dbpedia_triple_linker
from utils.tokenizer_simple import get_dependent_verb
from memory_profiler import profile

CANDIDATE_UP_TO = 150
SCORE_CONFIDENCE = 0.85

log = log_util.get_logger("subgraph")


def construct_subgraph(sentence, doc_title=''):
    not_linked_phrases_l, linked_phrases_l = dbpedia_triple_linker.link_sentence(sentence, doc_title)
    phrases = not_linked_phrases_l + [i['text'] for i in linked_phrases_l]

    relative_hash = {key: set() for key in phrases}
    embeddings_hash = {key: [] for key in phrases}

    merged_result = dbpedia_triple_linker.filter_text_vs_one_hop(not_linked_phrases_l, linked_phrases_l, embeddings_hash, relative_hash)
    r2 = dbpedia_triple_linker.filter_resource_vs_keyword(linked_phrases_l, embeddings_hash, relative_hash)
    no_exact_found = []
    # only keyword-match on those no exact match triples
    for i in linked_phrases_l:
        relatives = relative_hash[i['text']]
        if len(relatives) == 0:
            no_exact_found.append(i)
    r3 = dbpedia_triple_linker.filter_keyword_vs_keyword(no_exact_found, embeddings_hash, relative_hash)
    for i in r2 + r3:
        if not dbpedia_triple_linker.does_tri_exit_in_list(i, merged_result):
            merged_result.append(i)

    for t in linked_phrases_l:
        relatives = relative_hash[t['text']]
        if len(relatives) == 0:
            categories_triples = t['categories']
            merged_result.extend(categories_triples)

    # print(json.dumps(merged_result, indent=4))
    return merged_result

@profile
def construct_subgraph_for_claim(claim_text):
    not_linked_phrases_l, linked_phrases_l = dbpedia_triple_linker.link_sentence(claim_text, '')
    phrases = not_linked_phrases_l + [i['text'] for i in linked_phrases_l]

    relative_hash = {key: set() for key in phrases}
    embeddings_hash = {i['text']: {'phrase': [], 'one_hop': []} for i in linked_phrases_l}
    embeddings_hash.update({'not_linked_phrases_l': [], 'linked_phrases_l': []})
    lookup_hash = {i['text']: {'URI': i['URI'], 'text': i['text'], 'outbounds': i['outbounds'],
                               'categories': i['categories']} for i in linked_phrases_l}

    all_phrases = not_linked_phrases_l + [i['text'] for i in linked_phrases_l]
    verb_d = get_dependent_verb(claim_text, all_phrases)
    merged_result = dbpedia_triple_linker.filter_text_vs_one_hop(not_linked_phrases_l, linked_phrases_l, embeddings_hash, verb_d)
    r1 = dbpedia_triple_linker.filter_date_vs_property(claim_text, not_linked_phrases_l, linked_phrases_l, verb_d)
    r2 = dbpedia_triple_linker.filter_resource_vs_keyword(linked_phrases_l, embeddings_hash, relative_hash, fuzzy_match=True)
    no_exact_found = []
    # only keyword-match on those no exact match triples
    for i in linked_phrases_l:
        relatives = relative_hash[i['text']]
        if len(relatives) == 0:
            no_exact_found.append(i)
    r3 = dbpedia_triple_linker.filter_keyword_vs_keyword(no_exact_found, embeddings_hash, relative_hash, fuzzy_match=True)
    for i in r1 + r2 + r3:
        if not dbpedia_triple_linker.does_tri_exit_in_list(i, merged_result):
            merged_result.append(i)

    for t in no_exact_found:
        if len(t['categories']) < 1 or len(t['categories']) > 20:
            single_node = dict()
            single_node['subject'] = t['URI']
            single_node['object'] = ''
            single_node['relation'] = ''
            single_node['keywords'] = t['text']
            single_node['text'] = t['text']
            merged_result.append(single_node)
            continue

        for i in t['categories']:
            if not dbpedia_triple_linker.does_tri_exit_in_list(i, merged_result):
                merged_result.append(i)

    # print(json.dumps(merged_result, indent=4))
    claim_d = dict()
    claim_d['linked_phrases_l'] = linked_phrases_l
    claim_d['not_linked_phrases_l'] = not_linked_phrases_l
    claim_d['graph'] = merged_result
    claim_d['embedding'] = embeddings_hash
    claim_d['lookup_hash'] = lookup_hash
    return claim_d

@profile
def construct_subgraph_for_candidate(claim_dict, candidate_sent, doc_title=''):
    claim_linked_phrases_l = claim_dict['linked_phrases_l']
    claim_graph = claim_dict['graph']

    claim_linked_text = [i['text'] for i in claim_linked_phrases_l]
    filtered_links = []

    # sentence graph
    not_linked_phrases_l, linked_phrases_l = dbpedia_triple_linker.link_sentence(candidate_sent,
                                                                                 doc_title=doc_title,
                                                                                 lookup_hash=claim_dict['lookup_hash'])
    sent_linked_text = [i['text'] for i in linked_phrases_l]
    relative_hash = {key: set() for key in sent_linked_text}
    embeddings_hash = {i['text']: {'phrase': [], 'one_hop': []} for i in linked_phrases_l}
    embeddings_hash.update({'not_linked_phrases_l': [], 'linked_phrases_l': []})
    for i in linked_phrases_l:
        if not i['text'] in claim_dict['lookup_hash']:
            claim_dict['lookup_hash'].update({i['text']: {'URI': i['URI'], 'text': i['text'], 'outbounds': i['outbounds'],
                               'categories': i['categories']}})
    all_phrases = not_linked_phrases_l + [i['text'] for i in linked_phrases_l]
    verb_d = get_dependent_verb(candidate_sent, all_phrases)
    sent_graph = dbpedia_triple_linker.filter_text_vs_one_hop(not_linked_phrases_l, linked_phrases_l, embeddings_hash, verb_d)
    r1 = dbpedia_triple_linker.filter_date_vs_property(candidate_sent, not_linked_phrases_l, linked_phrases_l, verb_d)
    r2 = dbpedia_triple_linker.filter_resource_vs_keyword(linked_phrases_l, embeddings_hash, relative_hash, fuzzy_match=True)
    no_exact_found = []
    # only keyword-match on those no exact match triples
    for i in linked_phrases_l:
        relatives = relative_hash[i['text']]
        if len(relatives) == 0:
            no_exact_found.append(i)
    r3 = dbpedia_triple_linker.filter_keyword_vs_keyword(no_exact_found, embeddings_hash, relative_hash, fuzzy_match=False)
    for i in r1 + r2 + r3:
        if not dbpedia_triple_linker.does_tri_exit_in_list(i, sent_graph):
            sent_graph.append(i)

    claim_not_in_sent_nodes = []
    for i in claim_linked_phrases_l:
        overlap = False
        for j in linked_phrases_l:
            if i['text'] == j['text'] or i['URI'] == j['URI']:
                overlap = True
                break
        if not overlap:
            claim_not_in_sent_nodes.append(i)

    sent_not_in_claim_nodes = []
    for i in linked_phrases_l:
        overlap = False
        for j in claim_linked_phrases_l:
            if i['text'] == j['text'] or i['URI'] == j['URI']:
                overlap = True
                break
        if not overlap:
            sent_not_in_claim_nodes.append(i)

    claim_embedding = claim_dict['embedding']

    claim_not_in_sent_nodes_embedding = []
    empty_embedding_idx = []
    for idx, i in enumerate(claim_not_in_sent_nodes):
        claim_node_text = i['text']
        claim_node_embedding = claim_embedding[claim_node_text]['phrase']
        if len(claim_node_embedding) < 1:
            empty_embedding_idx.append(idx)
        else:
            claim_not_in_sent_nodes_embedding.append(claim_node_embedding)
    if len(empty_embedding_idx) > 0:
        claim_not_in_sent_nodes_text = [i['text'] for i in claim_not_in_sent_nodes]
        claim_not_in_sent_nodes_embedding = bert_similarity.get_phrase_embedding(claim_not_in_sent_nodes_text)
        if len(claim_not_in_sent_nodes_embedding) > 0:
            for j in empty_embedding_idx:
                node_text = claim_not_in_sent_nodes[j]['text']
                claim_embedding[node_text]['phrase'] = claim_not_in_sent_nodes_embedding[j]

    sent_not_in_claim_nodes_embedding = []
    empty_embedding_idx = []
    for idx, i in enumerate(sent_not_in_claim_nodes):
        sent_node_text = i['text']
        sent_node_embedding = embeddings_hash[sent_node_text]['phrase']
        if len(sent_node_embedding) < 1:
            empty_embedding_idx.append(idx)
        else:
            sent_not_in_claim_nodes_embedding.append(sent_node_embedding)
    if len(empty_embedding_idx) > 0:
        sent_not_in_claim_nodes_text = [i['text'] for i in sent_not_in_claim_nodes]
        sent_not_in_claim_nodes_embedding = bert_similarity.get_phrase_embedding(sent_not_in_claim_nodes_text)
        if len(sent_not_in_claim_nodes_embedding) > 0:
            for j in empty_embedding_idx:
                node_text = sent_not_in_claim_nodes[j]['text']
                embeddings_hash[node_text]['phrase'] = sent_not_in_claim_nodes_embedding[j]

    # claim_resource VS sent_resource_one_hop
    sent_filtered_one_hop = []
    if len(claim_not_in_sent_nodes) > 0 and len(claim_not_in_sent_nodes_embedding) > 0:
        for i in sent_not_in_claim_nodes:
            one_hop = dbpedia_triple_linker.get_one_hop(i)
            one_hop_keywords = [' '.join(tri['keywords']) for tri in one_hop]
            if len(one_hop) > CANDIDATE_UP_TO:
                continue
            one_hop_embedding = embeddings_hash[i['text']]['one_hop']
            if len(one_hop_embedding) < 1:
                one_hop_embedding = bert_similarity.get_phrase_embedding(one_hop_keywords)
                embeddings_hash[i['text']]['one_hop'] = one_hop_embedding
            if len(one_hop_embedding) < 1:
                continue
            top_k = 3
            try:
                out = pw.cosine_similarity(claim_not_in_sent_nodes_embedding, one_hop_embedding).flatten()
            except Exception as err:
                log.error(err)
                continue
            topk_idx = np.argsort(out)[::-1][:top_k]
            len2 = len(one_hop_embedding)
            for item in topk_idx:
                score = float(out[item])
                if score < float(SCORE_CONFIDENCE):
                    break
                else:
                    try:
                        tri2 = one_hop[item % len2]
                        tri2['text'] = i['text']
                        tri2['URI'] = i['URI']
                        tri2['score'] = score
                        sent_filtered_one_hop.append(tri2)
                    except Exception as err:
                        log.error(err)
                        log.error(f"one_hop: {one_hop}")
                        log.error(f"item: {item}, len2: {len2}")

    # claim one hop VS sentence resource
    claim_filtered_one_hop = []
    if len(sent_not_in_claim_nodes) > 0 and len(sent_not_in_claim_nodes_embedding) > 0:
        for i in claim_not_in_sent_nodes:
            c_one_hop = dbpedia_triple_linker.get_one_hop(i)
            c_one_hop_keywords = [' '.join(tri['keywords']) for tri in c_one_hop]
            if len(c_one_hop) > CANDIDATE_UP_TO:
                continue

            c_one_hop_embedding = claim_dict['embedding'][i['text']]['one_hop']
            if len(c_one_hop_embedding) < 1:
                c_one_hop_embedding = bert_similarity.get_phrase_embedding(c_one_hop_keywords)
                claim_dict['embedding'][i['text']]['one_hop'] = c_one_hop_embedding

            top_k = 3
            if len(c_one_hop_embedding) < 1:
                continue

            out = pw.cosine_similarity(sent_not_in_claim_nodes_embedding, c_one_hop_embedding).flatten()
            topk_idx = np.argsort(out)[::-1][:top_k]
            len2 = len(c_one_hop_embedding)
            for item in topk_idx:
                score = float(out[item])
                if score < float(SCORE_CONFIDENCE):
                    break
                else:
                    node1 = linked_phrases_l[item // len2]
                    filtered_links.append(node1)
                    tri2 = c_one_hop[item % len2]
                    tri2['text'] = i['text']
                    tri2['URI'] = i['URI']
                    tri2['score'] = score
                    claim_filtered_one_hop.append(tri2)

    # all together and sort
    for i in claim_filtered_one_hop:
        if not dbpedia_triple_linker.does_tri_exit_in_list(i, claim_graph):
            claim_graph.append(i)

    for i in sent_filtered_one_hop:
        if not dbpedia_triple_linker.does_tri_exit_in_list(i, sent_graph):
            sent_graph.append(i)

    return sent_graph


if __name__ == '__main__':
    cc1 = "Michelin Guides are published by George Lucas."
    s9 = "The term normally refers to the annually published Michelin Red Guide , the oldest European hotel and restaurant reference guide , which awards Michelin stars for excellence to a select few establishments ."
    s6 = "Mozilla Firefox ( or simply Firefox ) is a free and open-source web browser developed by the Mozilla Foundation and its subsidiary the Mozilla Corporation ."
    s7 = "Howard Eugene Johnson -LRB- 30 January 1915 -- 28 May 2000 -RRB- , better known as `` Stretch \'\' Johnson , was a tap dancer and social activist ."
    s8 = "Magic Johnson was a tap dancer"
    s9 = 'Tap Tap was a series of rhythm games by Tapulous available for the iOS of which several versions , both purchasable and free , have been produced .'
    s8 = "T - Pain, His debut album , Rappa Ternt Sanga , was released in 2005 ."
    s9 = "Chanhassen High School - Chanhassen had an enrollment of 1,576 students during the 2014-15 school year , with an 18:1 student teacher ratio ."
    # claim_dict = construct_subgraph_for_claim(s8)
    # construct_subgraph_for_candidate(claim_dict, s9, doc_title='Chanhassen High School')
