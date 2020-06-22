from utils.tokenizer_simple import *
from dbpedia_sampler import dbpedia_lookup
from dbpedia_sampler import dbpedia_virtuoso
from dbpedia_sampler import bert_similarity
from dbpedia_sampler import dbpedia_spotlight
from utils import c_scorer
import log_util
import itertools
import numpy as np
import sklearn.metrics.pairwise as pw
import difflib


STOP_WORDS = ['they', 'i', 'me', 'you', 'she', 'he', 'it', 'individual', 'individuals', 'we', 'who', 'where', 'what',
              'which', 'when', 'whom', 'the', 'history']
CANDIDATE_UP_TO = 150
SCORE_CONFIDENCE = 0.85

log = log_util.get_logger('dbpedia_triple_linker')


def get_phrases(sentence, doc_title=''):
    log.debug(sentence)
    if doc_title != '' and c_scorer.SENT_DOC_TITLE in sentence and sentence.startswith(doc_title):
        title_and_sen = sentence.split(c_scorer.SENT_DOC_TITLE, 1)
        sent = title_and_sen[1]
    else:
        sent = sentence

    chunks, ents = split_claim_spacy(sent)
    entities = [en[0] for en in ents]
    capitalized_phrased = split_claim_regex(sent)
    log.debug(f"chunks: {chunks}")
    log.debug(f"entities: {entities}")
    log.debug(f"capitalized phrases: {capitalized_phrased}")
    merged_entities = merge_phrases_l1_to_l2(capitalized_phrased, entities)
    if not doc_title == '':
        merged_entities = list(set(merged_entities) | set([doc_title]))
    merged_entities = [i for i in merged_entities if i.lower() not in STOP_WORDS]
    other_chunks = list(set(chunks) - set(merged_entities))
    log.debug(f"merged entities: {merged_entities}")
    log.debug(f"other phrases: {other_chunks}")
    return merged_entities, other_chunks


def merge_phrases_l1_to_l2(l1, l2):
    for i in l1:
        is_dup = False
        for j in l2:
            if i in j:
                is_dup = True
                break
        if not is_dup:
            l2.append(i)
    merged = [i for i in l2 if i.lower() not in STOP_WORDS]
    return merged


def lookup_phrase(phrase):
    linked_phrase = dict()
    resource_dict = dbpedia_lookup.lookup_resource(phrase)
    if isinstance(resource_dict, dict):
        linked_phrase['categories'] = dbpedia_lookup.to_triples(resource_dict)
        linked_phrase['text'] = phrase
        linked_phrase['URI'] = resource_dict['URI']
    return linked_phrase


def query_resource(uri):
    context = dict()
    context['inbounds'] = dbpedia_virtuoso.get_inbounds(uri)
    context['outbounds'] = dbpedia_virtuoso.get_outbounds(uri)
    context['URI'] = uri
    return context


def link_sentence(sentence, doc_title=''):
    entities, chunks = get_phrases(sentence, doc_title)
    not_linked_phrases_l = []
    linked_phrases_l = []
    # if len(entities) < 2:
    #     phrases = list(set(entities) | set(chunks))
    # else:
    #     phrases = entities
    phrases = list(set(entities) | set(chunks))

    for p in phrases:
        linked_phrase = lookup_phrase(p)
        if len(linked_phrase) == 0:
            not_linked_phrases_l.append(p)
        else:
            linked_phrases_l.append(linked_phrase)

    spotlight_links = dbpedia_spotlight.entity_link(sentence)
    for i in spotlight_links:
        surface = i['surfaceForm']
        i_URI = i['URI']
        if len(list(filter(lambda x: (surface in x['text'] and i_URI == x['URI']), linked_phrases_l))) == 0:
            linked_i = dict()
            linked_i['text'] = surface
            linked_i['URI'] = i_URI
            linked_phrases_l.append(linked_i)

    linked_phrases_l = merge_linked_l1_to_l2(linked_phrases_l, [])
    for i in linked_phrases_l:
        i.update(query_resource(i['URI']))
        if 'categories' not in i:
            i['categories'] = dbpedia_virtuoso.get_categories(i['URI'])

    return not_linked_phrases_l, linked_phrases_l


def merge_linked_l1_to_l2(l1, l2):
    for i in l1:
        text_i = i['text']
        URI_i = i['URI']
        is_dup = False
        for m in l2:
            text_m = m['text']
            URI_m = m['URI']
            if URI_i == URI_m:
                if text_m in text_i:
                    l2.remove(m)
                    break
                if text_i in text_m:
                    is_dup = True
                    break
            if not URI_i == URI_m:
                if text_i == text_m:
                    i['text'] = text_i + ' ,'  # same text linked to different entities
                    break
                if text_i in text_m:
                    is_dup = True
                    break
                if text_m in text_i:
                    l2.remove(m)
                    break
        if not is_dup:
            l2.append(i)
    return l2


def filter_text_vs_keyword(not_linked_phrases_l, linked_phrases_l, keyword_embeddings):
    if len(not_linked_phrases_l) > CANDIDATE_UP_TO or len(not_linked_phrases_l) < 1:
        return []

    embedding1 = keyword_embeddings['not_linked_phrases_l']
    if len(embedding1) == 0:
        embedding1 = bert_similarity.get_phrase_embedding(not_linked_phrases_l)
        keyword_embeddings['not_linked_phrases_l'] = embedding1
    for i in linked_phrases_l:
        candidates2 = i['categories'] + i['inbounds'] + i['outbounds']
        if len(candidates2) > CANDIDATE_UP_TO:
            continue
        embedding2 = keyword_embeddings[i['text']]['one_hop']
        if len(embedding2) == 0:
            tri_keywords_l2 = [' '.join(tri['keywords']) for tri in candidates2]
            embedding2 = bert_similarity.get_phrase_embedding(tri_keywords_l2)
            keyword_embeddings[i['text']]['one_hop'] = embedding2

        if len(embedding1) == 0 or len(embedding2) == 0:
            return []

        out = pw.cosine_similarity(embedding1, embedding2).flatten()
        topk_idx = np.argsort(out)[::-1][:3]
        len2 = len(tri_keywords_l2)
        tmp_result = []
        for item in topk_idx:
            score = float(out[item])
            if score < float(SCORE_CONFIDENCE):
                break
            else:
                p1 = not_linked_phrases_l[item // len2]
                tri2 = candidates2[item % len2]
                tri2['relatives'] = [i['URI'], i['URI']]
                tri2['text'] = i['text']
                tri2['URI'] = i['URI']
                tri2['score'] = score
                tmp_result.append(tri2)
    result = sorted(tmp_result, key=lambda t: t['score'], reverse=True)[:5]
    return result


def does_tri_exit_in_list(tri, tri_l):
    for item in tri_l:
        if tri['subject'] == item['subject'] \
                and tri['relation'] == item['relation'] \
                and dbpedia_virtuoso.keyword_extract(tri['object']) == dbpedia_virtuoso.keyword_extract(item['object']):
            return True
    return False


def filter_resource_vs_keyword(linked_phrases_l, keyword_embeddings, relative_hash,  fuzzy_match=False):
    result = []
    for i in itertools.permutations(linked_phrases_l, 2):
        resource1 = i[0]         # key
        resource2 = i[1]
        if resource1['text'] in resource2['text'] or resource2['text'] in resource1['text']:
            continue

        uri_matched = False
        # candidates
        candidates = resource2['categories'] + resource2['inbounds'] + resource2['outbounds']
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

        if fuzzy_match and not uri_matched:
            filtered_triples = get_topk_similar_triples(resource1['text'], resource2, keyword_embeddings, top_k=3)
            for item in filtered_triples:
                if not does_tri_exit_in_list(item, result):
                    result.append(item)
    return result


def filter_keyword_vs_keyword(linked_phrases_l, keyword_embeddings, relative_hash, fuzzy_match=False):
    result = []
    for i in itertools.combinations(linked_phrases_l, 2):
        resource1 = i[0]
        resource2 = i[1]
        if resource1['text'] in resource2['text'] or resource2['text'] in resource1['text']:
            continue

        candidates1 = resource1['categories'] + resource1['inbounds'] + resource1['outbounds']
        candidates2 = resource2['categories'] + resource2['inbounds'] + resource2['outbounds']

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
        if fuzzy_match and not exact_match:
            top_k_pairs = get_most_close_pairs(resource1, resource2, keyword_embeddings, top_k=3)
            for item in top_k_pairs:
                if not does_tri_exit_in_list(item, result):
                    result.append(item)
    return result


def get_most_close_pairs(resource1, resource2, keyword_embeddings, top_k=5):
    candidates1 = resource1['categories'] + resource1['inbounds'] + resource1['outbounds']
    candidates2 = resource2['categories'] + resource2['inbounds'] + resource2['outbounds']

    if len(candidates1) > CANDIDATE_UP_TO or len(candidates2) > CANDIDATE_UP_TO:
        return []

    tri_keywords_l1 = [' '.join(tri['keywords']) for tri in candidates1]
    tri_keywords_l2 = [' '.join(tri['keywords']) for tri in candidates2]
    embedding1 = keyword_embeddings[resource1['text']]['one_hop']
    embedding2 = keyword_embeddings[resource2['text']]['one_hop']
    if len(embedding1) == 0:
        embedding1 = bert_similarity.get_phrase_embedding(tri_keywords_l1)
        keyword_embeddings[resource1['text']]['one_hop'] = embedding1
    if len(embedding2) == 0:
        embedding2 = bert_similarity.get_phrase_embedding(tri_keywords_l2)
        keyword_embeddings[resource2['text']]['one_hop'] = embedding2

    if len(embedding1) == 0 or len(embedding2) == 0:
        return []

    out = pw.cosine_similarity(embedding1, embedding2).flatten()
    topk_idx = np.argsort(out)[::-1][:top_k]
    len2 = len(tri_keywords_l2)
    result = []
    for item in topk_idx:
        score = float(out[item])
        if score < float(SCORE_CONFIDENCE):
            break
        else:
            tri1 = candidates1[item//len2]
            tri2 = candidates2[item%len2]

            tri1['relatives'] = [resource1['URI'], resource2['URI']]
            tri1['text'] = resource1['text']
            tri1['URI'] = resource1['URI']
            tri1['score'] = score
            tri2['relatives'] = [resource2['URI'], resource1['URI']]
            tri2['text'] = resource2['text']
            tri2['URI'] = resource2['URI']
            tri2['score'] = score
            result.append(tri1)
            result.append(tri2)
    return result


def get_topk_similar_triples(single_phrase, linked_phrase, keyword_embeddings, top_k=2):
    # get embedding of single_phrase
    if single_phrase in keyword_embeddings:
        keyword_vec = keyword_embeddings[single_phrase]['phrase']
        if len(keyword_vec) == 0:
            tmp_embedding = bert_similarity.get_phrase_embedding([single_phrase])
            if len(tmp_embedding) > 0:
                keyword_vec = tmp_embedding[0]
                keyword_embeddings[single_phrase]['phrase'] = keyword_vec
            else:
                return []
    else:
        log.error(f'{single_phrase} is not initiated in keyword_embeddings')
        return []

    # get embedding for linked phrase triple keywords
    candidates = linked_phrase['categories'] + linked_phrase['inbounds'] + linked_phrase['outbounds']

    if len(candidates) > CANDIDATE_UP_TO or len(candidates) < 1:
        return []
    try:
        tri_keywords_l = [' '.join(tri['keywords']) for tri in candidates]
    except Exception as err:
        log.error(err)

    triple_vec_l = keyword_embeddings[linked_phrase['text']]['one_hop']
    if len(triple_vec_l) == 0:
        triple_vec_l = bert_similarity.get_phrase_embedding(tri_keywords_l)
        keyword_embeddings[linked_phrase['text']]['one_hop'] = triple_vec_l

    if keyword_vec == [] or triple_vec_l == []:   #failed to get phrase embeddings
        return []

    score = np.sum(keyword_vec * triple_vec_l, axis=1) / \
            (np.linalg.norm(keyword_vec) * np.linalg.norm(triple_vec_l, axis=1))
    topk_idx = np.argsort(score)[::-1][:top_k]
    result = []
    for idx in topk_idx:
        idx_score = float(score[idx])
        if idx_score < float(SCORE_CONFIDENCE):
            break
        else:
            record = candidates[idx]
            record['score'] = idx_score
            record['relatives'] = [linked_phrase['URI'], single_phrase]
            record['text'] = linked_phrase['text']
            record['URI'] = linked_phrase['URI']
            result.append(record)
        # print('>%s\t%s' % (score[idx], tri_keywords_l[idx]))
    return result


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
    link_sentence(sentence2, doc_title='Roman Atwood')