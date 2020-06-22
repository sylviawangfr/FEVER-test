from dbpedia_sampler import dbpedia_triple_linker
from dbpedia_sampler import bert_similarity
import numpy as np
import sklearn.metrics.pairwise as pw
import difflib


CANDIDATE_UP_TO = 150
SCORE_CONFIDENCE = 0.85


def construct_subgraph(sentence, doc_title=''):
    not_linked_phrases_l, linked_phrases_l = dbpedia_triple_linker.link_sentence(sentence, doc_title)
    phrases = not_linked_phrases_l + [i['text'] for i in linked_phrases_l]

    relative_hash = {key: set() for key in phrases}
    embeddings_hash = {key: [] for key in phrases}

    merged_result = dbpedia_triple_linker.filter_text_vs_keyword(not_linked_phrases_l, linked_phrases_l, embeddings_hash, relative_hash)
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


def construct_subgraph_for_claim(claim_text):
    not_linked_phrases_l, linked_phrases_l = dbpedia_triple_linker.link_sentence(claim_text, '')
    phrases = not_linked_phrases_l + [i['text'] for i in linked_phrases_l]

    relative_hash = {key: set() for key in phrases}
    embeddings_hash = {i['text']: {'phrase': [], 'one_hop': []} for i in linked_phrases_l}
    embeddings_hash.update({'not_linked_phrases_l': [], 'linked_phrases_l': []})

    merged_result = dbpedia_triple_linker.filter_text_vs_keyword(not_linked_phrases_l, linked_phrases_l, embeddings_hash)
    r2 = dbpedia_triple_linker.filter_resource_vs_keyword(linked_phrases_l, embeddings_hash, relative_hash, fuzzy_match=True)
    no_exact_found = []
    # only keyword-match on those no exact match triples
    for i in linked_phrases_l:
        relatives = relative_hash[i['text']]
        if len(relatives) == 0:
            no_exact_found.append(i)
    r3 = dbpedia_triple_linker.filter_keyword_vs_keyword(no_exact_found, embeddings_hash, relative_hash, fuzzy_match=True)
    for i in r2 + r3:
        if not dbpedia_triple_linker.does_tri_exit_in_list(i, merged_result):
            merged_result.append(i)

    for t in no_exact_found:
        for i in t['categories']:
            if not dbpedia_triple_linker.does_tri_exit_in_list(i, merged_result):
                merged_result.append(i)

    # print(json.dumps(merged_result, indent=4))
    claim_dict = dict()
    claim_dict['linked_phrases_l'] = linked_phrases_l
    claim_dict['not_linked_phrases_l'] = not_linked_phrases_l
    claim_dict['graph'] = merged_result
    claim_dict['embedding'] = embeddings_hash
    return claim_dict


def construct_subgraph_for_candidate(claim_dict, candidate_sent, doc_title=''):
    claim_linked_phrases_l = claim_dict['linked_phrases_l']
    claim_graph = claim_dict['graph']

    claim_linked_text = [i['text'] for i in claim_linked_phrases_l]
    filtered_links = []

    # sentence graph
    not_linked_phrases_l, linked_phrases_l = dbpedia_triple_linker.link_sentence(candidate_sent, doc_title)
    sent_linked_text = [i['text'] for i in linked_phrases_l]
    relative_hash = {key: set() for key in sent_linked_text}
    embeddings_hash = {i['text']: {'phrase': [], 'one_hop': []} for i in linked_phrases_l}
    embeddings_hash.update({'not_linked_phrases_l': [], 'linked_phrases_l': []})

    sent_graph = dbpedia_triple_linker.filter_text_vs_keyword(not_linked_phrases_l, linked_phrases_l, embeddings_hash)
    r2 = dbpedia_triple_linker.filter_resource_vs_keyword(linked_phrases_l, embeddings_hash, relative_hash, fuzzy_match=True)
    no_exact_found = []
    # only keyword-match on those no exact match triples
    for i in linked_phrases_l:
        relatives = relative_hash[i['text']]
        if len(relatives) == 0:
            no_exact_found.append(i)
    r3 = dbpedia_triple_linker.filter_keyword_vs_keyword(no_exact_found, embeddings_hash, relative_hash, fuzzy_match=False)
    for i in r2 + r3:
        if not dbpedia_triple_linker.does_tri_exit_in_list(i, sent_graph):
            sent_graph.append(i)

    for t in no_exact_found:
        for i in t['categories']:
            if not dbpedia_triple_linker.does_tri_exit_in_list(i, sent_graph):
                sent_graph.append(i)

    # claim_resources VS sent_resources
    # for i in claim_linked_phrases_l:
    #     keyword_matching = [difflib.SequenceMatcher(None, i['text'], j['text']).ratio() for j in linked_phrases_l]
    #     sorted_matching_index = sorted(range(len(keyword_matching)), key=lambda k: keyword_matching[k], reverse=True)
    #     top_match = keyword_matching[sorted_matching_index[0]]
    #     if top_match > 0.9:
    #         filtered_links.append(linked_phrases_l[sorted_matching_index[0]])

    # claim_resource VS sent_resource_one_hop
    filtered_text = [i['text'] for i in filtered_links]
    todo_sent_one_hop = []
    for i in linked_phrases_l:
        if not i['text'] in filtered_text:
            todo_sent_one_hop.append(i)
    claim_linked_phrase_embedding = claim_dict['embedding']['linked_phrases_l']
    if len(claim_linked_phrase_embedding) < 1:
        claim_linked_phrase_embedding = bert_similarity.get_phrase_embedding(claim_linked_text)
        claim_dict['embedding']['linked_phrases_l'] = claim_linked_phrase_embedding
    filtered_one_hop = []
    for i in todo_sent_one_hop:
        one_hop = i['categories'] + i['inbounds'] + i['outbounds']
        one_hop_keywords = [' '.join(tri['keywords']) for tri in one_hop]
        if len(one_hop) > CANDIDATE_UP_TO:
            continue
        one_hop_embedding = embeddings_hash[i['text']]['one_hop']
        if len(one_hop_embedding) < 1:
            one_hop_embedding = bert_similarity.get_phrase_embedding(one_hop_keywords)
            embeddings_hash[i['text']]['one_hop'] = one_hop_embedding
        top_k = 3
        out = pw.cosine_similarity(claim_linked_phrase_embedding, one_hop_embedding).flatten()
        topk_idx = np.argsort(out)[::-1][:top_k]
        len2 = len(one_hop_embedding)
        for item in topk_idx:
            score = float(out[item])
            if score < float(SCORE_CONFIDENCE):
                break
            else:
                tri2 = one_hop[item % len2]
                tri2['text'] = i['text']
                tri2['URI'] = i['URI']
                tri2['score'] = score
                filtered_one_hop.append(tri2)

    # claim one hop VS sentence resource
    sent_linked_embedding = embeddings_hash['linked_phrases_l']
    if len(sent_linked_embedding) < 1:
        sent_linked_embedding = bert_similarity.get_phrase_embedding(sent_linked_text)
        embeddings_hash['linked_phrases_l'] = sent_linked_embedding
    for i in claim_linked_phrases_l:
        c_one_hop = i['categories'] + i['inbounds'] + i['outbounds']
        c_one_hop_keywords = [' '.join(tri['keywords']) for tri in c_one_hop]
        if len(c_one_hop) > CANDIDATE_UP_TO:
            continue

        c_one_hop_embedding = claim_dict['embedding'][i['text']]['one_hop']
        if len(one_hop_embedding) < 1:
            one_hop_embedding = bert_similarity.get_phrase_embedding(c_one_hop_keywords)
            claim_dict['embedding'][i['text']]['one_hop'] = one_hop_embedding
            c_one_hop_embedding = one_hop_embedding

        top_k = 3
        out = pw.cosine_similarity(sent_linked_embedding, c_one_hop_embedding).flatten()
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
                claim_graph.append(tri2)

    # all together and sort
    sent_graph = dbpedia_triple_linker.merge_linked_l1_to_l2(filtered_one_hop, sent_graph)
    print(claim_graph)
    print(filtered_links)
    return sent_graph


if __name__ == '__main__':
    # embedding1 = bert_similarity.get_phrase_embedding(['Advertising'])
    # embedding2 = bert_similarity.get_phrase_embedding(['Pranksters'])
    # out = pw.cosine_similarity(embedding1, embedding2) # 0.883763313293457
    # text = "Autonomous cars shift insurance liability toward manufacturers"
    claim = "Roman Atwood is a content creator."
    sentence1 = "Brett Atwood is a website editor , content strategist and former print and online journalist whose " \
               "writings have appeared in Billboard , Rolling Stone , Vibe , " \
               "The Hollywood Reporter and other publications "
    sentence2 = "Roman Bernard Atwood (born May 28, 1983) is an American YouTube personality and prankster."
    # link_sentence(sentence2, doc_title='Roman Atwood')
    claim_dict = construct_subgraph_for_claim(claim)
    construct_subgraph_for_candidate(claim_dict, sentence2, doc_title='Roman Atwood')
