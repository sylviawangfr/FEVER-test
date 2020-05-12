from utils.tokenizer_simple import *
from dbpedia_sampler import dbpedia_lookup
from dbpedia_sampler import dbpedia_virtuoso
from dbpedia_sampler import bert_similarity
import json
import itertools


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
    keyword_embeddings = {key:[ ] for key in phrases}

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

    filter_resource_vs_keyword(linked_phrases_l, relative_hash)
    filter_text_vs_keyword(not_linked_phrases_l, linked_phrases_l, relative_hash)
    filter_keyword_vs_keyword(linked_phrases_l, relative_hash)


    # print(json.dumps(filtered_triples, indent=4))


def filter_text_vs_keyword(not_linked_phrases_l, linked_phrases_l, relative_hash):
    result = []
    for one_phrase in not_linked_phrases_l:
        for linked_p in linked_phrases_l:
            filtered_triples = []
            text = linked_p['text']
            uri = linked_p['URI']
            candidates = linked_p['categories'] + linked_p['ontology_linked_values'] + \
                         linked_p['linked_resources'] + linked_p['properties']
            filtered_triples.append(get_topk_triples(one_phrase, candidates, top_k=3))
            for tri in filtered_triples:
                tri['relatives'] = [linked_p['URI'], one_phrase]
                tri['text'] = text
                tri['URI'] = uri
            relative_hash[one_phrase].add(linked_p)
            relative_hash[text].add(one_phrase)
            result.append(filtered_triples)
    return result


def does_tri_exit_in_list(tri, tri_l):
    for item in tri_l:
        if tri['subject'] == item['subject'] \
                and tri['object'] == item['object'] \
                and tri['relation'] == item['relation']:
            return True
    return False


def filter_resource_vs_keyword(linked_phrases_l, relative_hash):
    result = []
    for i in itertools.permutations(linked_phrases_l, 2):
        uri_matched = False
        resource1 = i[0]         # key
        resource2 = i[1]        # candidates
        resource1_uri_short = dbpedia_virtuoso.keyword_extract(resource1['URI'])
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
            filtered_triples = get_topk_triples(resource1_uri_short, candidates, top_k=3)
            for item in filtered_triples:
                if not does_tri_exit_in_list(item, result):
                    item['relatives'] = [resource2['URI'], resource1['URI']]
                    item['text'] = resource2['text']
                    item['URI'] = resource2['URI']
                    result.append(item)
    return result



def filter_keyword_vs_keyword(linked_phrases_l, relative_hash):
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
            candidates1_keywords = [' '.join(c['keywords']) for c in candidates1]
            candidates2_keywords = [' '.join(c['keywords']) for c in candidates2]
            top_k_pair = bert_similarity.get_most_close_pair(candidates1_keywords, candidates2_keywords, top_k=3)
            for item in top_k_pair:
                tri1 = candidates1[item[0]]
                tri2 = candidates2[item[1]]
                score = item[2]
                if not does_tri_exit_in_list(tri1, result):
                    tri1['relatives'] = [resource1['URI'], resource2['URI']]
                    tri1['text'] = resource1['text']
                    tri1['URI'] = resource1['URI']
                    tri1['score'] = score
                    result.append(tri1)
                if not does_tri_exit_in_list(tri2, result):
                    tri2['relatives'] = [resource2['URI'], resource1['URI']]
                    tri2['text'] = resource2['text']
                    tri2['URI'] = resource2['URI']
                    tri2['score'] = score
                    result.append(tri2)
    return result



def get_topk_triples(keyword, triple_l, top_k=2):
    topk_l = bert_similarity.get_topk_similar_phrases(keyword, [' '.join(tri['keywords']) for tri in triple_l],
                                              top_k=top_k)
    for i in topk_l:
        idx = i['idx']
        i['subject'] = triple_l[idx]['subject']
        i['relation'] = triple_l[idx]['relation']
        i['object'] = triple_l[idx]['object']
    return topk_l


if __name__ == '__main__':
    # text1 = "Autonomous cars shift insurance liability toward manufacturers"
    text2 = "Magic Johnson did not play for the Lakers."
    construct_subgraph(text2)
