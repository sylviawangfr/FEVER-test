from utils.tokenizer_simple import *
from dbpedia_sampler import dbpedia_lookup
from dbpedia_sampler import dbpedia_virtuoso
from dbpedia_sampler import bert_similarity
import copy


def get_phrases(text):
    chunks, ents = split_claim_spacy(text)
    entities = [en[0] for en in ents]
    capitalized_phrased = split_claim_regex(text)
    print(f"chunks: {chunks}")
    print(f"entities: {entities}")
    print(f"capitalized phrases: {capitalized_phrased}")
    phrases = list(set(chunks) | set(entities) | set(capitalized_phrased))
    print(f"phrases: {phrases}")
    return phrases

def entity_lookup(phrase):
    pass


def construct_subgraph(text):
    phrases = get_phrases(text)
    entity_graphs = []
    not_linked_phrases_l = []
    linked_phrases_l = []

    for p in phrases:
        linked_phrase = dict()
        resource_dict = dbpedia_lookup.lookup_resource(p)
        if resource_dict < 0:
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
            linked_phrases_l.append(linked_phrase)

    for no_link_p  in not_linked_phrases_l:
        for linked_p in linked_phrases_l:
            close_categories = get_topk_triples(no_link_p, linked_p['categories'])
            close_ontology_linked_values = get_topk_triples(no_link_p, linked_p['ontology_linked_values'])
            close_linked_resources = get_topk_triples(no_link_p, linked_p['linked_resources'])
            close_properties = get_topk_triples(no_link_p, linked_p['properties'])

    if len(linked_phrases_l)



def get_topk_triples(keyword, triple_l):
    topk_l = bert_similarity.get_topk_similar_phrases(keyword, triple_l['keywords'])
    for i in topk_l:
        idx = i['idx']
        i['subject'] = triple_l[idx]['subject']
        i['relation'] = triple_l[idx]['relation']
        i['object'] = triple_l[idx]['object']
        i['text'] = triple_l[idx]['text']
    return topk_l







if __name__ == '__main__':
    text1 = "Autonomous cars shift insurance liability toward manufacturers"
    text2 = "Magic Johnson did not play for the Lakers."
    get_phrases(text2)
