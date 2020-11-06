import numpy as np
import sklearn.metrics.pairwise as pw

import log_util
from dbpedia_sampler import bert_similarity
from dbpedia_sampler import dbpedia_triple_linker
from utils.tokenizer_simple import get_dependent_verb
from memory_profiler import profile
from bert_serving.client import BertClient
import gc

CANDIDATE_UP_TO = 150
SCORE_CONFIDENCE = 0.85

log = log_util.get_logger("claim_context")


def construct_subgraph_for_claim(claim_text, bc:BertClient=None):
    not_linked_phrases_l, linked_phrases_l = dbpedia_triple_linker.link_sentence(claim_text, '')
    phrases = not_linked_phrases_l + [i['text'] for i in linked_phrases_l]

    relative_hash = {key: set() for key in phrases}
    embeddings_hash = {i['text']: {'phrase': [], 'one_hop': []} for i in linked_phrases_l}
    embeddings_hash.update({'not_linked_phrases_l': [], 'linked_phrases_l': []})
    lookup_hash = {i['text']: {'URI': i['URI'], 'text': i['text'], 'outbounds': i['outbounds'],
                               'categories': i['categories']} for i in linked_phrases_l}

    all_phrases = not_linked_phrases_l + [i['text'] for i in linked_phrases_l]
    verb_d = get_dependent_verb(claim_text, all_phrases)
    merged_result = dbpedia_triple_linker.filter_text_vs_one_hop(not_linked_phrases_l, linked_phrases_l, embeddings_hash, verb_d, bc=bc)
    r1 = dbpedia_triple_linker.filter_date_vs_property(claim_text, not_linked_phrases_l, linked_phrases_l, verb_d)
    r2 = dbpedia_triple_linker.filter_resource_vs_keyword(linked_phrases_l, embeddings_hash, relative_hash, fuzzy_match=True, bc=bc)
    no_exact_found = []
    # only keyword-match on those no exact match triples
    for i in linked_phrases_l:
        relatives = relative_hash[i['text']]
        if len(relatives) == 0:
            no_exact_found.append(i)
    r3 = dbpedia_triple_linker.filter_keyword_vs_keyword(no_exact_found, embeddings_hash, relative_hash, fuzzy_match=True, bc=bc)
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