import difflib
import itertools

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
import time
import gc


CANDIDATE_UP_TO = 150
SCORE_CONFIDENCE_1 = 0.6
SCORE_CONFIDENCE_2 = 0.85

log = log_util.get_logger('dbpedia_triple_linker')


def keyword_matching(text, str_l):
    keyword_matching_score = [difflib.SequenceMatcher(None, text, i).ratio() for i in str_l]
    sorted_matching_index = sorted(range(len(keyword_matching_score)), key=lambda k: keyword_matching_score[k], reverse=True)
    return keyword_matching_score, sorted_matching_index


def merge_linked_l1_to_l2(l1, l2):
    for i in l1:
        text_i = i['text']
        URI_i = i['URI']
        is_dup = False
        to_delete = []
        for m in l2:
            if not 'text' in m:
                continue
            text_m = m['text']
            URI_m = m['URI']
            short_uri_i = dbpedia_virtuoso.uri_short_extract(URI_i)
            short_uri_m = dbpedia_virtuoso.uri_short_extract(URI_m)
            score_i, sorted_idx_i = keyword_matching(text_i.lower(), [short_uri_i.lower()])
            score_m, sorted_idx_m = keyword_matching(text_m.lower(), [short_uri_m.lower()])
            if URI_i == URI_m:
                if score_i[0] == 1:  # exact match
                    to_delete.append(m)
                    continue
                if score_m[0] == 1:  # exact match
                    is_dup = True
                    break
                if text_m in text_i:
                    to_delete.append(m)
                    continue
                if text_i in text_m:
                    is_dup = True
                    break
            else:   # URI_i != URI_m:
                if text_i == text_m:
                    if score_i[0] == 1:   # exact match, pick i
                        to_delete.append(m)
                        continue
                    if score_m[0] == 1:
                        is_dup = True
                        break
                    # if score[0] > score[1]:    Australia VS Australian VS Australians
                    #     l2.remove(m)
                    #     break
                    # else:
                    is_dup = True
                    break
                if text_i in text_m:
                    if score_i[0] < SCORE_CONFIDENCE_2:
                        is_dup = True
                        break
                # if text_m in text_i:
                #     l2.remove(m)
                #     break
        if not is_dup:
            l2.append(i)
        for d in to_delete:
            l2.remove(d)
    return l2


def filter_date_vs_property(sents, not_linked_phrases_l, linked_phrases_l, verb_d):
    if len(not_linked_phrases_l) < 1:
        return []
    all_date_phrases = []
    for p in not_linked_phrases_l:
        if text_clean.is_date(p):
            all_date_phrases.append(p)
    if len(all_date_phrases) < 1:
        return []

    all_date_properties = []
    for res in linked_phrases_l:
        one_hop = get_one_hop(res)
        for tri in one_hop:
            if 'datatype' in tri and tri['datatype'] == 'date':
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

                for i in sorted_matching_index:
                    score = keyword_matching_score[i]
                    if score >= 0.5:
                        similarity_match.append(all_date_properties[sorted_matching_index[i]])

    similarity_match.extend(all_exact_match)
    return similarity_match


def filter_text_vs_one_hop(not_linked_phrases_l, linked_phrases_l, keyword_embeddings, verb_d, bc:BertClient=None):
    if len(not_linked_phrases_l) > CANDIDATE_UP_TO or len(not_linked_phrases_l) < 1:
        return []

    embedding1 = keyword_embeddings['not_linked_phrases_l']
    if len(embedding1) == 0:
        with_verbs = []
        for p in not_linked_phrases_l:
            if is_date_or_number(p) and p in verb_d:
                with_verbs.append(verb_d[p]['verb'] + " " + p)
            else:
                with_verbs.append(p)
        embedding1 = bert_similarity.get_phrase_embedding(with_verbs, bc)
        keyword_embeddings['not_linked_phrases_l'] = embedding1

    tmp_result = []
    for i in linked_phrases_l:
        candidates2 = get_one_hop(i)
        if len(candidates2) > CANDIDATE_UP_TO:
            continue
        embedding2 = keyword_embeddings[i['text']]['one_hop']
        if len(embedding2) == 0:
            tri_keywords_l2 = [' '.join(tri['keywords']) for tri in candidates2]
            embedding2 = bert_similarity.get_phrase_embedding(tri_keywords_l2, bc)
            keyword_embeddings[i['text']]['one_hop'] = embedding2

        if len(embedding1) == 0 or len(embedding2) == 0:
            return []

        out = pw.cosine_similarity(embedding1, embedding2).flatten()
        topk_idx = np.argsort(out)[::-1][:3]
        len2 = len(tri_keywords_l2)

        for item in topk_idx:
            score = float(out[item])
            if score < float(SCORE_CONFIDENCE_2):
                break
            else:
                p1 = not_linked_phrases_l[item // len2]
                tri2 = candidates2[item % len2]
                tri2['relatives'] = [i['URI'], i['URI']]
                tri2['text'] = i['text']
                tri2['URI'] = i['URI']
                tri2['score'] = score
                tmp_result.append(tri2)
    # result = sorted(tmp_result, key=lambda t: t['score'], reverse=True)[:5]
    return tmp_result


def does_tri_exit_in_list(tri, tri_l):
    for item in tri_l:
        if tri['subject'] == item['subject'] \
                and tri['relation'] == item['relation'] \
                and dbpedia_virtuoso.uri_short_extract(tri['object']) == dbpedia_virtuoso.uri_short_extract(item['object']):
            return True
    return False

# @profile
def filter_resource_vs_keyword(linked_phrases_l, keyword_embeddings, relative_hash,  fuzzy_match=False, bc:BertClient=None):
    result = []
    for i in itertools.permutations(linked_phrases_l, 2):
        resource1 = i[0]         # key
        resource2 = i[1]
        if resource1['text'] in resource2['text'] \
                or resource2['text'] in resource1['text'] \
                or resource1['URI'] == resource2['URI']:
            continue

        uri_matched = False
        # candidates
        candidates = get_one_hop(resource2)
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
            if len(keyword_embeddings['linked_phrases_l']) < 1:
                linked_phrases_text_l = [i['text'] for i in linked_phrases_l]
                linked_text_embedding = bert_similarity.get_phrase_embedding(linked_phrases_text_l, bc)
                keyword_embeddings['linked_phrases_l'] = linked_text_embedding
                if len(linked_text_embedding) < 1:
                    continue
                for idx, i in enumerate(linked_phrases_l):
                    keyword_embeddings[i['text']]['phrase'] = linked_text_embedding[idx]

            filtered_triples = get_topk_similar_triples(resource1['text'], resource2, keyword_embeddings, top_k=3)
            for item in filtered_triples:
                if not does_tri_exit_in_list(item, result):
                    result.append(item)
    return result


def filter_keyword_vs_keyword(linked_phrases_l, keyword_embeddings, relative_hash, fuzzy_match=False, bc:BertClient=None):
    result = []
    for i in itertools.combinations(linked_phrases_l, 2):
        resource1 = i[0]
        resource2 = i[1]
        if resource1['text'] in resource2['text'] \
                or resource2['text'] in resource1['text'] \
                or resource1['URI'] == resource2['URI']:
            continue

        candidates1 = get_one_hop(resource1)
        candidates2 = get_one_hop(resource2)
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
            top_k_pairs = get_most_close_pairs(resource1, resource2, keyword_embeddings, top_k=3, bc=bc)
            for item in top_k_pairs:
                if not does_tri_exit_in_list(item, result):
                    result.append(item)
    return result


def get_most_close_pairs(resource1, resource2, keyword_embeddings, top_k=5, bc:BertClient=None):
    candidates1 = get_one_hop(resource1)
    candidates2 = get_one_hop(resource2)

    if len(candidates1) > CANDIDATE_UP_TO or len(candidates2) > CANDIDATE_UP_TO:
        return []

    tri_keywords_l1 = [' '.join(tri['keywords']) for tri in candidates1]
    tri_keywords_l2 = [' '.join(tri['keywords']) for tri in candidates2]
    embedding1 = keyword_embeddings[resource1['text']]['one_hop']
    embedding2 = keyword_embeddings[resource2['text']]['one_hop']
    if not len(embedding1) == len(candidates1):
        embedding1 = bert_similarity.get_phrase_embedding(tri_keywords_l1, bc)
        keyword_embeddings[resource1['text']]['one_hop'] = embedding1
    if not len(embedding2) == len(candidates2):
        embedding2 = bert_similarity.get_phrase_embedding(tri_keywords_l2, bc)
        keyword_embeddings[resource2['text']]['one_hop'] = embedding2

    if len(embedding1) == 0 or len(embedding2) == 0:
        return []

    out = pw.cosine_similarity(embedding1, embedding2).flatten()
    topk_idx = np.argsort(out)[::-1][:top_k]
    len2 = len(tri_keywords_l2)
    result = []
    for item in topk_idx:
        score = float(out[item])
        if score < float(SCORE_CONFIDENCE_2):
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

# @profile
def get_topk_similar_triples(single_phrase, linked_phrase, keyword_embeddings, top_k=2, bc:BertClient=None):
    # get embedding of single_phrase
    if single_phrase in keyword_embeddings:
        keyword_vec = keyword_embeddings[single_phrase]['phrase']
        if len(keyword_vec) == 0:
            tmp_embedding = bert_similarity.get_phrase_embedding([single_phrase], bc)
            if len(tmp_embedding) > 0:
                keyword_vec = tmp_embedding[0]
                keyword_embeddings[single_phrase]['phrase'] = keyword_vec
            else:
                return []
    else:
        log.error(f'{single_phrase} is not initiated in keyword_embeddings')
        return []

    # get embedding for linked phrase triple keywords
    candidates = get_one_hop(linked_phrase)

    if len(candidates) > CANDIDATE_UP_TO or len(candidates) < 1:
        return []
    try:
        tri_keywords_l = [' '.join(tri['keywords']) for tri in candidates]
    except Exception as err:
        log.error(err)
        tri_keywords_l = []

    triple_vec_l = keyword_embeddings[linked_phrase['text']]['one_hop']
    if not len(triple_vec_l) == len(candidates):
        triple_vec_l = bert_similarity.get_phrase_embedding(tri_keywords_l, bc)
        keyword_embeddings[linked_phrase['text']]['one_hop'] = triple_vec_l

    if keyword_vec == [] or triple_vec_l == []:   #failed to get phrase embeddings
        return []

    score = np.sum(keyword_vec * triple_vec_l, axis=1) / \
            (np.linalg.norm(keyword_vec) * np.linalg.norm(triple_vec_l, axis=1))
    topk_idx = np.argsort(score)[::-1][:top_k]
    result = []
    for idx in topk_idx:
        idx_score = float(score[idx])
        if idx_score < float(SCORE_CONFIDENCE_2):
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



if __name__ == '__main__':
    test()
    test()
    # embedding1 = bert_similarity.get_phrase_embedding(['Advertising'])
    # embedding2 = bert_similarity.get_phrase_embedding(['Pranksters'])
    # out = pw.cosine_similarity(embedding1, embedding2) # 0.883763313293457
    # text = "Autonomous cars shift insurance liability toward manufacturers"
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