from utils.tokenizer_simple import *
from dbpedia_sampler import dbpedia_lookup
from dbpedia_sampler import dbpedia_virtuoso
from dbpedia_sampler import bert_similarity
import json
import itertools
import numpy as np
import sklearn.metrics.pairwise as pw


def get_phrases(text):
    chunks, ents = split_claim_spacy(text)
    entities = [en[0] for en in ents]
    capitalized_phrased = split_claim_regex(text)
    print(f"chunks: {chunks}")
    print(f"entities: {entities}")
    print(f"capitalized phrases: {capitalized_phrased}")
    phrases = list(set(chunks) | set(entities) | set(capitalized_phrased))
    merged = []
    for p in phrases:
        is_dup = False
        for m in merged:
            if m in p:
                merged.remove(m)
                merged.append(p)
                is_dup = True
                break
            if p in m:
                is_dup = True
                break
        if not is_dup:
            merged.append(p)
    return merged


def construct_subgraph(text):
    phrases = get_phrases(text)
    entity_graphs = []
    not_linked_phrases_l = []
    linked_phrases_l = []
    relative_hash = {key: set() for key in phrases}
    embeddings_hash = {key: [] for key in phrases}

    for p in phrases:
        linked_phrase = dict()
        resource_dict = dbpedia_lookup.lookup_resource(p)
        if not isinstance(resource_dict, dict):
            not_linked_phrases_l.append(p)
        else:
            resource = resource_dict['URI']
            linked_phrase['categories'] = dbpedia_lookup.to_triples(resource_dict)
            linked_phrase['ontology_linked_values'] = dbpedia_virtuoso.get_ontology_linked_values_inbound(resource) + \
                                                      dbpedia_virtuoso.get_ontology_linked_values_outbound(resource)
            linked_phrase['properties'] = dbpedia_virtuoso.get_properties(resource)
            linked_phrase['linked_resources'] = dbpedia_virtuoso.get_one_hop_resource_inbound(resource) + \
                                                dbpedia_virtuoso.get_one_hop_resource_outbound(resource)
            linked_phrase['text'] = p
            linked_phrase['URI'] = resource
            linked_phrases_l.append(linked_phrase)

    merged_result = filter_text_vs_keyword(not_linked_phrases_l, linked_phrases_l, embeddings_hash, relative_hash)
    r2 = filter_resource_vs_keyword(linked_phrases_l, embeddings_hash, relative_hash)
    no_exact_found = []
    # only keyword-match on those no exact match triples
    for i in linked_phrases_l:
        relatives = relative_hash[i['text']]
        if len(relatives) == 0:
            no_exact_found.append(i)
    r3 = filter_keyword_vs_keyword(no_exact_found, embeddings_hash, relative_hash)
    for i in r2 + r3:
        if not does_tri_exit_in_list(i, merged_result):
            merged_result.append(i)

    print(json.dumps(merged_result, indent=4))


def filter_text_vs_keyword(not_linked_phrases_l, linked_phrases_l, keyword_embeddings, relative_hash):
    result = []
    for one_phrase in not_linked_phrases_l:
        for linked_p in linked_phrases_l:
            text = linked_p['text']
            filtered_triples = get_topk_similar_triples(one_phrase, linked_p, keyword_embeddings, top_k=3)
            for tri in filtered_triples:
                if not does_tri_exit_in_list(tri, result):
                    relative_hash[one_phrase].add(linked_p)
                    relative_hash[text].add(one_phrase)
                    result.append(tri)
    return result


def does_tri_exit_in_list(tri, tri_l):
    for item in tri_l:
        if tri['subject'] == item['subject'] \
                and tri['object'] == item['object'] \
                and tri['relation'] == item['relation']:
            return True
    return False


def filter_resource_vs_keyword(linked_phrases_l, keyword_embeddings, relative_hash):
    result = []
    for i in itertools.permutations(linked_phrases_l, 2):
        uri_matched = False
        resource1 = i[0]         # key
        resource2 = i[1]        # candidates
        candidates = resource2['categories'] + resource2['ontology_linked_values'] + \
                     resource2['linked_resources'] + resource2['properties']
        resource1_uri = resource1['URI']
        for item in candidates:
            if resource1_uri in [item['subject'], item['relation'], item['object']]:
                uri_matched = True
                if not does_tri_exit_in_list(item, result):    # perfectly linked uri
                    relative_hash[resource1['text']].add(resource2['text'])
                    relative_hash[resource2['text']].add(resource1['text'])
                    item['relatives'] = [resource2['URI'], resource1['URI']]
                    item['text'] = resource2['text']
                    item['URI'] = resource2['URI']
                    result.append(item)

        if not uri_matched:
            filtered_triples = get_topk_similar_triples(resource1_uri, resource2, keyword_embeddings, top_k=3)
            for item in filtered_triples:
                if not does_tri_exit_in_list(item, result):
                    result.append(item)
    return result


def filter_keyword_vs_keyword(linked_phrases_l, keyword_embeddings, relative_hash):
    result = []
    for i in itertools.combinations(linked_phrases_l, 2):
        resource1 = i[0]
        resource2 = i[1]
        candidates1 = resource1['categories'] + resource1['ontology_linked_values'] + \
                     resource1['linked_resources'] + resource1['properties']
        candidates2 = resource2['categories'] + resource2['ontology_linked_values'] + \
                     resource2['linked_resources'] + resource2['properties']

        exact_match = False
        for item1 in candidates1:
            for item2 in candidates2:
                if item1['keywords'] == item2['keywords']:
                    exact_match = True
                    relative_hash[resource1['text']].add(resource2['text'])
                    relative_hash[resource2['text']].add(resource1['text'])
                    if not does_tri_exit_in_list(item1, result):
                        item1['relatives'] = [resource1['URI'], resource2['URI']]
                        item1['text'] = resource1['text']
                        item1['URI'] = resource1['URI']
                        result.append(item1)
                    if not does_tri_exit_in_list(item2, result):
                        item2['relatives'] = [resource2['URI'], resource1['URI']]
                        item2['text'] = resource2['text']
                        item2['URI'] = resource2['URI']
                        result.append(item2)
        if not exact_match:
            top_k_pairs = get_most_close_pairs(resource1, resource2, keyword_embeddings, top_k=3)
            for item in top_k_pairs:
                if not does_tri_exit_in_list(item, result):
                    result.append(item)
    return result


def get_most_close_pairs(resource1, resource2, keyword_embeddings, top_k=5):
    candidates1 = resource1['categories'] + resource1['ontology_linked_values'] + \
                  resource1['linked_resources'] + resource1['properties']
    candidates2 = resource2['categories'] + resource2['ontology_linked_values'] + \
                  resource2['linked_resources'] + resource2['properties']
    tri_keywords_l1 = [' '.join(tri['keywords']) for tri in candidates1]
    tri_keywords_l2 = [' '.join(tri['keywords']) for tri in candidates2]
    embedding1 = keyword_embeddings[resource1['text']]
    embedding2 = keyword_embeddings[resource2['text']]
    if len(embedding1) == 0:
        embedding1 = bert_similarity.get_phrase_embedding(tri_keywords_l1)
        keyword_embeddings[resource1['text']] = embedding1
    if len(embedding2) == 0:
        embedding2 = bert_similarity.get_phrase_embedding(tri_keywords_l2)
        keyword_embeddings[resource2['text']] = embedding2

    out = pw.cosine_similarity(embedding1, embedding2).flatten()
    topk_idx = np.argsort(out)[::-1][:top_k]
    len2 = len(tri_keywords_l2)
    result = []
    for item in topk_idx:
        tri1 = candidates1[item//len2]
        tri2 = candidates2[item%len2]
        score = out[item]
        tri1['relatives'] = [resource1['URI'], resource2['URI']]
        tri1['text'] = resource1['text']
        tri1['URI'] = resource1['URI']
        tri1['score'] = float(score)
        tri2['relatives'] = [resource2['URI'], resource1['URI']]
        tri2['text'] = resource2['text']
        tri2['URI'] = resource2['URI']
        tri2['score'] = float(score)
        result.append(tri1)
        result.append(tri2)
    return result


def get_topk_similar_triples(single_phrase, linked_phrase, keyword_embeddings, top_k=2):
    # get embedding of single_phrase
    if single_phrase in keyword_embeddings:
        keyword_vec = keyword_embeddings[single_phrase]
        if len(keyword_vec) == 0:
            keyword_vec = bert_similarity.get_phrase_embedding([single_phrase])[0]
            keyword_embeddings[single_phrase] = keyword_vec
    else:
        short_phrase = dbpedia_virtuoso.keyword_extract(single_phrase)
        keyword_vec = bert_similarity.get_phrase_embedding([short_phrase])[0]
        keyword_embeddings[single_phrase] = keyword_vec

    # get embedding for linked phrase triple keywords
    candidates = linked_phrase['categories'] + linked_phrase['ontology_linked_values'] + \
                 linked_phrase['linked_resources'] + linked_phrase['properties']
    tri_keywords_l = [' '.join(tri['keywords']) for tri in candidates]
    triple_vec_l = keyword_embeddings[linked_phrase['text']]
    if len(triple_vec_l) == 0:
        triple_vec_l = bert_similarity.get_phrase_embedding(tri_keywords_l)
        keyword_embeddings[linked_phrase['text']] = triple_vec_l

    if keyword_vec == [] or triple_vec_l == []:   #failed to get phrase embeddings
        return []

    score = np.sum(keyword_vec * triple_vec_l, axis=1) / \
            (np.linalg.norm(keyword_vec) * np.linalg.norm(triple_vec_l, axis=1))
    topk_idx = np.argsort(score)[::-1][:top_k]
    result = []
    for idx in topk_idx:
        record = candidates[idx]
        record['score'] = float(score[idx])
        record['relatives'] = [linked_phrase['URI'], single_phrase]
        record['text'] = linked_phrase['text']
        record['URI'] = linked_phrase['URI']
        result.append(record)
        # print('>%s\t%s' % (score[idx], tri_keywords_l[idx]))
    return result

if __name__ == '__main__':
    # text1 = "Autonomous cars shift insurance liability toward manufacturers"
    text1 = "Magic Johnson did not play for the Lakers."
    # text1 = 'Don Bradman retired from soccer.'
    construct_subgraph(text1)
