import difflib
import itertools
from utils.text_clean import convert_brc
import dateutil.parser as dateutil
import numpy as np
import sklearn.metrics.pairwise as pw

import log_util
from dbpedia_sampler import bert_similarity
from dbpedia_sampler import dbpedia_lookup
from dbpedia_sampler import dbpedia_spotlight
from dbpedia_sampler import dbpedia_virtuoso
from dbpedia_sampler.sentence_util import *
from bert_serving.client import BertClient
from utils import c_scorer, text_clean
from memory_profiler import profile
from dbpedia_sampler.uri_util import isURI
import time
import gc


CANDIDATE_UP_TO = 150
SCORE_CONFIDENCE_1 = 0.6
SCORE_CONFIDENCE_2 = 0.75
SCORE_CONFIDENCE_3 = 0.8

log = log_util.get_logger('dbpedia_triple_linker')

# @profile
def lookup_phrase(phrase):
    linked_phrase = dict()
    resource_dict = dbpedia_lookup.lookup_resource(phrase)
    if len(resource_dict) > 0:
        linked_phrase['text'] = phrase
        linked_phrase['links'] = []
    for i in resource_dict:
        media_dict = dict()
        if isinstance(i, dict):
            media_dict['categories'] = dbpedia_lookup.to_triples(i)
            media_dict['URI'] = i['URI']
            media_dict['text'] = phrase
            media_dict['exact_match'] = i['exact_match']
            linked_phrase['links'].append(media_dict)
    return linked_phrase


def lookup_doc_id(phrase, doc_ids):
    linked_phrase = dict()
    links = []
    for d in doc_ids:
        tmp_d = convert_brc(d).replace('_', ' ').lower()
        resource_dict = dbpedia_lookup.lookup_label_exact_match(tmp_d)
        for i in resource_dict:
            r_dict = dict()
            if isinstance(i, dict):
                r_dict['categories'] = dbpedia_lookup.to_triples(i)
                r_dict['URI'] = i['URI']
                r_dict['text'] = phrase
                r_dict['exact_match'] = (phrase.lower() == tmp_d)
                links.append(r_dict)
    if len(links) > 0:
        linked_phrase['text'] = phrase
        linked_phrase['links'] = links
        linked_phrase['text'] = phrase
    return linked_phrase


# @profile
def query_resource(uri):
    context = dict()
    outbounds = dbpedia_virtuoso.get_outbounds(uri)
    if len(outbounds) < 1:
        outbounds = dbpedia_virtuoso.get_disambiguates_outbounds(uri)
    context['outbounds'] = outbounds
    return context


# @profile
def link_sentence(sentence, extend_entity_docs=None, doc_title='', lookup_hash=None):
    # not_linked_phrases_l, linked_phrases_l = link_sent_to_resources(sentence, doc_title=doc_title, lookup_hash=lookup_hash)
    not_linked_phrases_l, linked_phrases_l = link_sent_to_resources2(sentence,
                                                                     extend_entity_docs=extend_entity_docs,
                                                                     doc_title=doc_title,
                                                                     lookup_hash=lookup_hash)
    add_categories(linked_phrases_l)
    return not_linked_phrases_l, linked_phrases_l


# @profile
def link_sent_to_resources(sentence, doc_title='', lookup_hash=None):
    sentence = text_clean.convert_brc(sentence)
    entities, chunks = get_phrases(sentence, doc_title)
    not_linked_phrases_l = []
    linked_phrases_l = []
    # if len(entities) < 2:
    #     phrases = list(set(entities) | set(chunks))
    # else:
    #     phrases = entities
    phrases = merge_chunks_with_entities(chunks, entities)
    log.debug(f"all phrases to be linked: {phrases}")

    for p in phrases:
        if text_clean.is_date_or_number(p):
            not_linked_phrases_l.append(p)
            continue
        if lookup_hash is not None and p in lookup_hash:
            linked_phrase = lookup_hash[p]
        else:
            linked_phrase = lookup_phrase(p)

        if len(linked_phrase) == 0:
            not_linked_phrases_l.append(p)
        else:
            linked_phrases_l.append(linked_phrase)

    # if with_spotlight:
    #     spotlight_links = dbpedia_spotlight.entity_link(sentence)
    #     spotlight_resources = []
    #     for i in spotlight_links:
    #         surface = i['surfaceForm']
    #         if surface in STOP_WORDS:
    #             break
    #         i_URI = i['URI']
    #         if len(list(filter(lambda x: (surface in x['text'] and i_URI == x['URI']), linked_phrases_l))) == 0:
    #             linked_i = dict()
    #             linked_i['text'] = surface
    #             linked_i['URI'] = i_URI
    #             spotlight_resources.append(linked_i)
    #     merged_links_step1 = merge_linked_l1_to_l2(spotlight_resources, linked_phrases_l)
    #     linked_phrases_l = merge_linked_l1_to_l2(merged_links_step1, [])
    # else:
    #     linked_phrases_l = merge_linked_l1_to_l2(linked_phrases_l, [])
    return not_linked_phrases_l, linked_phrases_l


def link_sent_to_resources2(sentence, extend_entity_docs=None, doc_title='', lookup_hash=None):
    def merge_links(linked_p, es_linked_p):
        if len(es_linked_p) == 0:
            return linked_p
        if len(linked_p) == 0:
            return es_linked_p

        links1 = linked_p['links']
        links2 = es_linked_p['links']
        for i in links2:
            if i['URI'] not in links1:
                links1.append(i)
        return linked_p

    sentence = text_clean.convert_brc(sentence)
    entities, chunks = get_phrases(sentence, doc_title)
    linked_phrases_l = []
    not_linked_phrases_l = []
    if len(entities) < 1:
        phrases = chunks
    else:
        phrases = entities
        for c in chunks:
            if c.count(' ') > 0:
                phrases.append(c)
            else:
                not_linked_phrases_l.append(c)

    for p in phrases:
        if text_clean.is_date_or_number(p):
            not_linked_phrases_l.append(p)
            continue
        if lookup_hash is not None and p in lookup_hash:
            linked_phrase = lookup_hash[p]
        else:
            linked_phrase = lookup_phrase(p)
            if extend_entity_docs is not None and p in extend_entity_docs:
                es_doc_links = extend_entity_docs[p]
                if len(es_doc_links) > 0:
                    linked_phrase = merge_links(linked_phrase, es_doc_links)
            if lookup_hash is not None and len(linked_phrase) > 0:
                lookup_hash.update({p: linked_phrase})

        if len(linked_phrase) == 0:
            not_linked_phrases_l.append(p)
        else:
            linked_phrases_l.append(linked_phrase)
    # filter_not_linked =
    return not_linked_phrases_l, linked_phrases_l


def add_categories(linked_ps):
    for i in linked_ps:
        for j in i['links']:
            if 'outbounds' not in j:
                j.update(query_resource(j['URI']))
            if 'categories' not in j:
                j['categories'] = dbpedia_virtuoso.get_categories(j['URI'])


def keyword_matching(text, str_l):
    keyword_matching_score = [difflib.SequenceMatcher(None, text, i).ratio() for i in str_l]
    sorted_matching_index = sorted(range(len(keyword_matching_score)), key=lambda k: keyword_matching_score[k], reverse=True)
    return keyword_matching_score, sorted_matching_index


# def merge_linked_l1_to_l2(l1, l2):
#     for i in l1:
#         text_i = i['text']
#         URI_i = i['URI']
#         is_dup = False
#         to_delete = []
#         for m in l2:
#             if not 'text' in m:
#                 continue
#             text_m = m['text']
#             URI_m = m['URI']
#             short_uri_i = dbpedia_virtuoso.uri_short_extract(URI_i)
#             short_uri_m = dbpedia_virtuoso.uri_short_extract(URI_m)
#             score_i, sorted_idx_i = keyword_matching(text_i.lower(), [short_uri_i.lower()])
#             score_m, sorted_idx_m = keyword_matching(text_m.lower(), [short_uri_m.lower()])
#             if URI_i == URI_m:
#                 if score_i[0] == 1:  # exact match
#                     to_delete.append(m)
#                     continue
#                 if score_m[0] == 1:  # exact match
#                     is_dup = True
#                     break
#                 if text_m in text_i:
#                     to_delete.append(m)
#                     continue
#                 if text_i in text_m:
#                     is_dup = True
#                     break
#             else:   # URI_i != URI_m:
#                 if text_i == text_m:
#                     if score_i[0] == 1:   # exact match, pick i
#                         to_delete.append(m)
#                         continue
#                     if score_m[0] == 1:
#                         is_dup = True
#                         break
#                     # if score[0] > score[1]:    Australia VS Australian VS Australians
#                     #     l2.remove(m)
#                     #     break
#                     # else:
#                     is_dup = True
#                     break
#                 if text_i in text_m:
#                     if score_i[0] < SCORE_CONFIDENCE_2:
#                         is_dup = True
#                         break
#                 # if text_m in text_i:
#                 #     l2.remove(m)
#                 #     break
#         if not is_dup:
#             l2.append(i)
#         for d in to_delete:
#             l2.remove(d)
#     return l2


def filter_date_vs_property(not_linked_phrases_l, linked_phrases_l, verb_d):
    if len(not_linked_phrases_l) < 1:
        return []
    all_date_phrases = []
    for p in not_linked_phrases_l:
        if text_clean.is_date(p):
            all_date_phrases.append(p)
    if len(all_date_phrases) < 1:
        return []

    all_date_properties = []
    all_uris = dict()
    for i in linked_phrases_l:
        for u in i['links']:
            if u['URI'] not in all_uris:
                all_uris.update({u['URI']: u})
    for res in all_uris.values():
        one_hop = get_one_hop(res)
        for tri in one_hop:
            if 'datatype' in tri and tri['datatype'] == 'date':
                tri['text'] = res['text']
                all_date_properties.append(tri)
    if len(all_date_properties) < 1:
        return []

    no_exact_match_phrase = []
    all_exact_match = []
    for i in all_date_phrases:
        date_i = dateutil.parse(i)
        p_exact_match = []
        for j in all_date_properties:
            short_obj = dbpedia_virtuoso.uri_short_extract(j['object'])
            if text_clean.is_date(short_obj) and (dateutil.parse(short_obj) == date_i):
                j['relatives'] = [j['text'], i]
                p_exact_match.append(j)
        if len(p_exact_match) < 1:
            no_exact_match_phrase.append(i)
        else:
            all_exact_match.extend(p_exact_match)

    similarity_match = []
    if len(no_exact_match_phrase) > 0:
        for i in no_exact_match_phrase:
            if i in verb_d:
                v = verb_d[i]['verb']
                one_hop_relation = [dbpedia_virtuoso.uri_short_extract(tri['relation'])
                                    for tri in all_date_properties]
                keyword_matching_score = [difflib.SequenceMatcher(None, v.lower(), property_value.lower()).ratio() for
                                          property_value in one_hop_relation]
                sorted_matching_index = sorted(range(len(keyword_matching_score)),
                                               key=lambda k: keyword_matching_score[k],
                                               reverse=True)
                if keyword_matching_score[sorted_matching_index[0]] < 0.5:
                    # similarity_match.extend(all_date_properties)
                    break

                for j in sorted_matching_index:
                    score = keyword_matching_score[j]
                    if score >= 0.5:
                        tmp_tri = all_date_properties[j]
                        tmp_tri['relatives'] = [tmp_tri['text'], i]
                        tmp_tri['URI'] = tmp_tri['subject']
                        similarity_match.append(tmp_tri)
                    else:
                        break

    similarity_match.extend(all_exact_match)
    return similarity_match


def filter_text_vs_one_hop(not_linked_phrases_l, linked_phrases_l, keyword_embeddings, verb_d, bc:BertClient=None):
    if len(not_linked_phrases_l) > CANDIDATE_UP_TO or len(not_linked_phrases_l) < 1:
        return []

    embedding1 = keyword_embeddings['not_linked_phrases_l']
    if len(embedding1) == 0:
        with_verbs = []
        for p in not_linked_phrases_l:
            if text_clean.is_date_or_number(p) and p in verb_d:
                with_verbs.append(verb_d[p]['verb'] + " " + p)
            else:
                with_verbs.append(p)
        embedding1 = bert_similarity.get_phrase_embedding(with_verbs, bc)
        if len(embedding1) > 0:
            keyword_embeddings['not_linked_phrases_l'] = embedding1
        else:
            return []

    result = []
    all_uris = dict()
    for i in linked_phrases_l:
        for u in i['links']:
            if u['URI'] not in all_uris:
                all_uris.update({u['URI']: u})
    for m in all_uris.values():
        tmp_result = similarity_between_nonlinked_and_linked(not_linked_phrases_l, embedding1, m, keyword_embeddings, bc)
        if len(tmp_result) > 0:
            result.extend(tmp_result)
    # only top 2 triples
    filted_result = filter_triples(result)
    return filted_result


def filter_triples(triples):
    relatives_l = []
    all_triples = []
    for tri in triples:
        rel = tri['relatives']
        if len(list(filter(lambda x: x == rel, relatives_l))) == 0:
            relatives_l.append(rel)
    for r in relatives_l:
        tri_r = []
        for t in triples:
            if t['relatives'] == r:
                tri_r.append(t)
        tri_r.sort(key=lambda k: k['score'], reverse=True)
        top_k =2
        for idx, i in enumerate(tri_r):
            if i['score'] > 0.9 or idx < top_k:
                all_triples.append(i)
    return all_triples


def similarity_between_nonlinked_and_linked(not_linked_phrases_l, phrase_list_embedding, linked_resource, keyword_embeddings, bc:BertClient=None):
    candidates = get_one_hop(linked_resource)
    if len(candidates) > CANDIDATE_UP_TO:
        return []
    embedding2 = keyword_embeddings[linked_resource['URI']]['one_hop']
    tri_keywords_l2 = [' '.join(tri['keywords']) for tri in candidates]
    if len(embedding2) == 0:
        embedding2 = bert_similarity.get_phrase_embedding(tri_keywords_l2, bc)
        keyword_embeddings[linked_resource['URI']]['one_hop'] = embedding2

    if len(embedding2) == 0:
        return []

    out = pw.cosine_similarity(phrase_list_embedding, embedding2).flatten()
    topk_idx = np.argsort(out)[::-1][:2]
    len2 = len(tri_keywords_l2)

    tmp_result = []
    for item in topk_idx:
        score = float(out[item])
        if score < float(SCORE_CONFIDENCE_3):
            break
        else:
            p1 = not_linked_phrases_l[item // len2]
            tri2 = candidates[item % len2]
            tri2['relatives'] = [p1, linked_resource['text']]
            tri2['text'] = linked_resource['text']
            tri2['URI'] = linked_resource['URI']
            tri2['score'] = score
            tri2['exact_match'] = linked_resource['exact_match']
            tmp_result.append(tri2)
    return tmp_result


def does_tri_exit_in_list(tri, tri_l):
    for item in tri_l:
        if tri['subject'] == item['subject'] \
                and tri['keywords'] == item['keywords']:
            return True
    return False


def does_node_exit_in_list(node_uri, tri_l):
    for item in tri_l:
        if node_uri == item['subject'] or node_uri == item['relation'] or node_uri == item['object']:
            return True
    return False


def filter_node_vs_two_hop(linked_phrases_l, isolated_nodes, keyword_embeddings, threshold=0.65, bc:BertClient=None):
    result = []
    topk = 2
    two_hop_hash = dict()
    for i_node in isolated_nodes:
        keyword_vec = lookup_or_update_phrase_embedding_hash(i_node['text'], keyword_embeddings, bc=bc)
        if len(keyword_vec) < 1:
            continue

        tmp_result = []
        for other_nodes in linked_phrases_l:
            if i_node['text'] in other_nodes['text'] or other_nodes['text'] in i_node['text']:
                continue
            other_nodes_l = other_nodes['links']
            for n in other_nodes_l:
                nodes_one_hop = get_one_hop(n)
                one_hop_triples = list(filter(lambda x: ("http://dbpedia.org/resource/" in x['object']), nodes_one_hop))
                for oht in one_hop_triples:
                    ohr = oht['object']
                    if len(two_hop_hash) > 0 and ohr in two_hop_hash:
                        two_hops = two_hop_hash[ohr]['two_hops']
                    else:
                        two_hops = dbpedia_virtuoso.get_outbounds(ohr)
                        two_hop_hash.update({ohr: {'two_hops': two_hops, 'embedding': []}})
                    if len(two_hops) > CANDIDATE_UP_TO or len(two_hops) < 1:
                        continue
                    try:
                        tri_keywords_l = [' '.join(tri['keywords']) for tri in two_hops]
                    except Exception as err:
                        log.error(err)
                        tri_keywords_l = []

                    triple_vec_l = two_hop_hash[ohr]['embedding']
                    if not len(triple_vec_l) == len(two_hops):
                        triple_vec_l = bert_similarity.get_phrase_embedding(tri_keywords_l, bc)
                        two_hop_hash[ohr]['embedding'] = triple_vec_l
                    if keyword_vec == [] or triple_vec_l == []:  # failed to get phrase embeddings
                        continue

                    score = np.sum(keyword_vec * triple_vec_l, axis=1) / \
                            (np.linalg.norm(keyword_vec) * np.linalg.norm(triple_vec_l, axis=1))
                    topk_idx = np.argsort(score)[::-1][:topk]
                    two_hop_records = []
                    for idx in topk_idx:
                        idx_score = float(score[idx])
                        if idx_score < float(threshold):
                            break
                        else:
                            two_hop_tri = two_hops[idx]
                            two_hop_tri['score'] = idx_score
                            two_hop_records.append(two_hop_tri)
                    if len(two_hop_records) > 0:
                        record = dict()
                        one_hop_record = list(filter(lambda x: (x['object'] == ohr), nodes_one_hop))
                        record['one_hop'] = one_hop_record
                        record['two_hop'] = two_hop_records
                        record['text'] = n['text'] + '_two_hop'
                        record['relatives'] = [i_node['text'], n['text']]
                        tmp_result.append(record)
                        # print('>%s\t%s' % (score[idx], tri_keywords_l[idx]))
        if len(tmp_result) > 0:
            result.extend(tmp_result)
    return result


def lookup_or_update_phrase_embedding_hash(keyword, embedding_hash, bc:BertClient=None):
    keyword_vec = []
    if keyword in embedding_hash:
        keyword_vec = embedding_hash[keyword]
    if len(keyword_vec) == 0:
        tmp_embedding = bert_similarity.get_phrase_embedding([keyword], bc)
        if len(tmp_embedding) > 0:
            keyword_vec = tmp_embedding[0]
            embedding_hash[keyword] = keyword_vec
    return keyword_vec


def lookup_or_update_onehop_embedding_hash(linked_resource, embedding_hash, bc:BertClient=None):
    keyword_vec = []
    if linked_resource['URI'] in embedding_hash:
        keyword_vec = embedding_hash[linked_resource['URI']]['one_hop']
    candidates = get_one_hop(linked_resource)
    if len(keyword_vec) == 0 or len(keyword_vec) != len(candidates):
        tri_keywords_l2 = [' '.join(tri['keywords']) for tri in candidates]
        keyword_vec = bert_similarity.get_phrase_embedding(tri_keywords_l2, bc)
        embedding_hash[linked_resource['URI']]['one_hop'] = keyword_vec
    return keyword_vec


# @profile
def filter_resource_vs_keyword(linked_phrases_l, keyword_embeddings,  fuzzy_match=False, bc:BertClient=None):
    result = []
    for i in itertools.permutations(linked_phrases_l, 2):
        resource1 = i[0]         # key
        resource2 = i[1]
        if resource1['text'] in resource2['text'] \
                or resource2['text'] in resource1['text']:
            continue

        uri_matched = False
        # candidates
        resource1_l = resource1['links']
        resource2_l = resource2['links']
        for re1 in resource1_l:
            re1_uri = re1['URI']
            for re2 in resource2_l:
                re2_uri = re2['URI']
                if re1_uri == re2_uri:
                    continue
                candidates = get_one_hop(re2)
                for item in candidates:
                    if re1_uri in [item['subject'], item['relation'], item['object']]:
                        uri_matched = True
                        if not does_tri_exit_in_list(item, result):    # perfectly linked uri
                            item['relatives'] = [re2['text'], re1['text']]
                            item['text'] = re2['text']
                            item['URI'] = re2['URI']
                            item['score'] = float(1)
                            item['exact_match'] = re1['exact_match'] | re2['exact_match']
                            result.append(item)

        if fuzzy_match and not uri_matched:
            if len(keyword_embeddings['linked_phrases_l']) < 1:
                linked_phrases_text_l = [i['text'] for i in linked_phrases_l]
                linked_text_embedding = bert_similarity.get_phrase_embedding(linked_phrases_text_l, bc)
                keyword_embeddings['linked_phrases_l'] = linked_text_embedding
                if len(linked_text_embedding) < 1:
                    continue
                for idx, i in enumerate(linked_phrases_l):
                    keyword_embeddings[i['text']] = linked_text_embedding[idx]

            filtered_triples = get_topk_similar_triples(resource1['text'], resource2, keyword_embeddings,
                                                        top_k=3, threshold=SCORE_CONFIDENCE_3, bc=bc)
            for item in filtered_triples:
                if not does_tri_exit_in_list(item, result):
                    result.append(item)
    result = filter_triples(result)
    return result


def filter_keyword_vs_keyword(isolated_nodes, linked_phrases_l, keyword_embeddings, fuzzy_match=False, bc:BertClient=None):
    result = []
    for i in itertools.product(isolated_nodes, linked_phrases_l):
        resource1 = i[0]
        resource2 = i[1]

        if resource1['text'] in resource2['text'] or resource2['text'] in resource1['text']:
            continue

        resource1_l = resource1['links']
        resource2_l = resource2['links']
        candidates1 = []
        candidates2 = []
        for i in resource1_l:
            candidates1.extend(get_one_hop(i))
        for j in resource2_l:
            candidates2.extend(get_one_hop(j))
        exact_match = False
        for item1 in candidates1:
            for item2 in candidates2:
                if item1['subject'] == item2['subject']:
                    continue
                if item1['keywords'] == item2['keywords']:
                    exact_match = True
                    if not does_tri_exit_in_list(item1, result):
                        item1['relatives'] = [resource1['text'], resource2['text']]
                        item1['text'] = resource1['text']
                        item1['URI'] = item1['subject']
                        item1['score'] = float(1)
                        item1['exact_match'] = i['exact_match']
                        result.append(item1)
                    if not does_tri_exit_in_list(item2, result):
                        item2['relatives'] = [resource2['text'], resource1['text']]
                        item2['text'] = resource2['text']
                        item2['URI'] = item2['subject']
                        item2['exact_match'] = j['exact_match']
                        item2['score'] = float(1)
                        result.append(item2)
        if fuzzy_match and not exact_match:
            for re1 in resource1_l:
                for re2 in resource2_l:
                    if re1['URI'] == re2['URI']:
                        continue
                    top_k_pairs = get_most_close_pairs(re1, re2, keyword_embeddings, top_k=3, bc=bc)
                    for item in top_k_pairs:
                        if not does_tri_exit_in_list(item, result):
                            result.append(item)
    result = filter_triples(result)
    return result


def get_most_close_pairs(resource1, resource2, keyword_embeddings, top_k=5, bc:BertClient=None):
    candidates1 = get_one_hop(resource1)
    candidates2 = get_one_hop(resource2)

    if len(candidates1) > CANDIDATE_UP_TO or len(candidates2) > CANDIDATE_UP_TO:
        return []

    tri_keywords_l1 = [' '.join(tri['keywords']) for tri in candidates1]
    tri_keywords_l2 = [' '.join(tri['keywords']) for tri in candidates2]
    embedding1 = lookup_or_update_onehop_embedding_hash(resource1, keyword_embeddings, bc=bc)
    embedding2 = lookup_or_update_onehop_embedding_hash(resource2, keyword_embeddings, bc=bc)

    if len(embedding1) == 0 or len(embedding2) == 0 \
            or not len(embedding1) == len(candidates1) \
            or not len(embedding2) == len(candidates2):
        return []

    out = pw.cosine_similarity(embedding1, embedding2).flatten()
    topk_idx = np.argsort(out)[::-1][:top_k]
    len2 = len(tri_keywords_l2)
    result = []
    for item in topk_idx:
        score = float(out[item])
        if score < float(SCORE_CONFIDENCE_3):
            break
        else:
            tri1 = candidates1[item//len2]
            tri2 = candidates2[item%len2]

            tri1['relatives'] = [resource1['text'], resource2['text']]
            tri1['text'] = resource1['text']
            tri1['URI'] = resource1['URI']
            tri1['exact_match'] = resource1['exact_match']
            tri1['score'] = score
            tri2['relatives'] = [resource2['text'], resource1['text']]
            tri2['text'] = resource2['text']
            tri2['URI'] = resource2['URI']
            tri2['score'] = score
            tri2['exact_match'] = resource2['exact_match']
            result.append(tri1)
            result.append(tri2)
    return result


# @profile
def get_topk_similar_triples(single_phrase, linked_phrase, keyword_embeddings, top_k=2, threshold=SCORE_CONFIDENCE_2, bc:BertClient=None):
    # get embedding of single_phrase

    keyword_vec = lookup_or_update_phrase_embedding_hash(single_phrase, keyword_embeddings, bc=bc)
    if len(keyword_vec) == 0:
        log.error(f'failed to get embedding for {single_phrase}')
        return []

    resource2_l = linked_phrase['links']
    result = []
    for res in resource2_l:
        candidates = get_one_hop(res)
        if len(candidates) > CANDIDATE_UP_TO or len(candidates) < 1:
            continue

        triple_vec_l = lookup_or_update_onehop_embedding_hash(res, keyword_embeddings, bc=bc)
        if len(triple_vec_l) == 0:   #failed to get phrase embeddings
            continue

        score = np.sum(keyword_vec * triple_vec_l, axis=1) / \
                (np.linalg.norm(keyword_vec) * np.linalg.norm(triple_vec_l, axis=1))
        topk_idx = np.argsort(score)[::-1][:top_k]
        for idx in topk_idx:
            idx_score = float(score[idx])
            if idx_score < float(threshold):
                break
            else:
                record = candidates[idx]
                record['score'] = idx_score
                record['relatives'] = [single_phrase, res['text']]
                record['text'] = res['text']
                record['URI'] = res['URI']
                record['exact_match'] = res['exact_match']
                result.append(record)
            # print('>%s\t%s' % (score[idx], tri_keywords_l[idx]))
    result.sort(key=lambda k: k['score'], reverse=True)
    return result


def get_one_hop(linked_dict):
    # one_hop = linked_dict['categories'] + linked_dict['inbounds'] + linked_dict['outbounds']
    one_hop = linked_dict['categories'] + linked_dict['outbounds']
    return one_hop


@profile
def test():
    s2 = "'History of art includes architecture, dance, sculpture, music, painting, poetry " \
             "literature, theatre, narrative, film, photography and graphic arts.'"

    s3 = "'Graphic arts - Graphic art further includes calligraphy , photography , painting , typography , " \
         "computer graphics , and bindery .'"

    s4 = "Homeland is an American spy thriller television series developed by Howard Gordon and Alex Gansa based on" \
         " the Israeli series Prisoners of War ( Original title חטופים Hatufim , literally `` Abductees '' ) , " \
         "which was created by Gideon Raff .."
    s5 = "He is best known for hosting the talent competition show American Idol , "
    s_l = [s2, s3, s4, s5]
    for i in s_l:
        x, y = link_sentence(i)
        del x
        del y
        gc.collect()
        time.sleep(5)

def test_similarity(ph1, ph2):
    bc = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=60000)
    phe = bert_similarity.get_phrase_embedding([ph1, ph2], bc)
    out = pw.cosine_similarity([phe[0]], [phe[1]]).flatten()
    return out


if __name__ == '__main__':
    si = test_similarity('United States', 'Germany')

    # test()
    # test()
    # embedding1 = bert_similarity.get_phrase_embedding(['Advertising'])
    # embedding2 = bert_similarity.get_phrase_embedding(['Pranksters'])
    # out = pw.cosine_similarity(embedding1, embedding2) # 0.883763313293457
    text = "Autonomous cars shift insurance liability toward manufacturers"
    # claim = "Roman Atwood is a content creator."
    # sentence1 = "Brett Atwood is a website editor , content strategist and former print and online journalist whose " \
    #            "writings have appeared in Billboard , Rolling Stone , Vibe , " \
    #            "The Hollywood Reporter and other publications "
    # sentence2 = "Roman Bernard Atwood (born May 28, 1983) is an American YouTube personality and prankster."
    # s1 = "Narrative can be organized in a number of thematic or formal categories : non-fiction -LRB- such as " \
    #      "definitively including creative non-fiction , biography , journalism , transcript poetry , " \
    #      "and historiography -RRB- ; fictionalization of historical events -LRB- such as anecdote , myth , " \
    #      "legend , and historical fiction -RRB- ; and fiction proper -LRB- such as literature in prose and " \
    #      "sometimes poetry , such as short stories , novels , and narrative poems and songs , and imaginary narratives " \
    #      "as portrayed in other textual forms , games , or live or recorded performances -RRB- ."
    # s2 = "'History of art includes architecture, dance, sculpture, music, painting, poetry " \
    #      "literature, theatre, narrative, film, photography and graphic arts.'"
    #
    # s3 = "'Graphic arts - Graphic art further includes calligraphy , photography , painting , typography , " \
    #      "computer graphics , and bindery .'"
    #
    # s4 = "Homeland is an American spy thriller television series developed by Howard Gordon and Alex Gansa based on" \
    #      " the Israeli series Prisoners of War ( Original title חטופים Hatufim , literally `` Abductees '' ) , " \
    #      "which was created by Gideon Raff .."
    # s5 = "He is best known for hosting the talent competition show American Idol , " \
    #      "as well as the syndicated countdown program American Top 40 and the KIIS-FM morning radio show On Air with Ryan Seacrest ."
    # s6 = "Mozilla Firefox ( or simply Firefox ) is a free and open-source web browser developed by the Mozilla Foundation and its subsidiary the Mozilla Corporation ."
    # s7 = "Firefox is a computer game."
    # s8 = "Where the Heart Is ( 2000 film ) - The filmstars Natalie Portman , Stockard Channing , Ashley Judd , and Joan Cusack with supporting roles done by James Frain , Dylan Bruno , Keith David , and Sally Field ."
    # no_l, l = link_sentence(s8, doc_title='')

    # all_phrases = no_l + [i['text'] for i in l]
    # verb_d = get_dependent_verb(s6, all_phrases)