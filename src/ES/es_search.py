import itertools
from dbpedia_sampler.sentence_util import merge_chunks_with_entities
from functools import reduce
from utils.resource_manager import CountryNationality
from ES.es_queries import *


MEDIA = ['tv', 'film', 'book', 'novel', 'band', 'album', 'music', 'series', 'poem',
         'song', 'advertisement', 'company', 'episode', 'season', 'animator',
         'actor', 'singer', 'writer', 'drama', 'character']

def search_media(entities):
    r_list = []
    no_resource_found = []
    has_resource_found = []
    country_nationality = CountryNationality()
    for e in entities:
        if (not is_capitalized(e)) or country_nationality.is_nationality(e) or country_nationality.is_country(e):
            continue
        must = []
        should = []
        must.append({'multi_match': {'query': e, "type": "phrase",
                                         'fields': ['id^2'], 'analyzer': 'underscore_analyzer'}})
        for m in MEDIA:
            should.append({'multi_match': {'query': m, "type": "phrase",
                                           'fields': ['id'], 'analyzer': 'underscore_analyzer'}})
        search = Search(using=client, index=config.WIKIPAGE_INDEX)
        search = search.query(Q('bool', must=must, should=should)). \
                     highlight('lines', number_of_fragments=0). \
                     sort({'_score': {"order": "desc"}}). \
                     source(include=['id'])[0:10]

        response = search.execute()
        tmp_r = []
        if len(response['hits']['hits']) == 0:
            no_resource_found.append(e)
            if (' or ' in e) or (' and ' in e) or (' of ' in e) or (' by ' in e) or (' in ' in e) or ("\'s " in e):
                splits = split_combinations(e)
                for i in splits:
                    if i not in entities:
                        entities.append(i)
            continue

        for hit in response['hits']['hits']:
            score = hit['_score']
            id = hit['_source']['id']
            if 'highlight' in hit:
                lines = hit['highlight']['lines'][0]
                lines = lines.replace("</em> <em>", " ")
            else:
                lines = ""
            doc_dic = {'score': score, 'phrases': [e], 'id': id, 'lines': lines}
            tmp_r.append(doc_dic)
            has_resource_found.append(e)
        r_list.extend(tmp_r)
    r_list.sort(key=lambda x: x.get('score'), reverse=True)
    return r_list


def search_entity_combinations(entitie_subsets):
    def construct_must_and_should(must_l, should_l):
        must = []
        should = []
        for m in must_l:
            m = remove_the_a(m)
            must.append({'multi_match': {'query': m, "type": "phrase",
                                         'fields': ['id^2', 'lines'], 'slop': 3, 'analyzer': 'underscore_analyzer'}})
        for s in should_l:
            s = remove_the_a(s)
            must.append({'multi_match': {'query': s, "type": "phrase",
                                           'fields': ['id', 'lines'], 'slop': 3, 'analyzer': 'underscore_analyzer'}})
        return must, should

    try:
        r_list = []
        for entities in entitie_subsets:
            for ph in entities:
                tmp_r = []
                must_entity = [ph]
                should_entities = [i for i in entities if i != ph]
                must, should = construct_must_and_should(must_entity, should_entities)
                search = Search(using=client, index=config.WIKIPAGE_INDEX)
                search = search.query(Q('bool', must=must, should=should)). \
                             highlight('lines', number_of_fragments=0). \
                             sort({'_score': {"order": "desc"}}). \
                             source(include=['id'])[0:10]

                response = search.execute()
                for hit in response['hits']['hits']:
                    score = hit['_score']
                    id = hit['_source']['id']
                    if 'highlight' in hit:
                        lines = hit['highlight']['lines'][0]
                        lines = lines.replace("</em> <em>", " ")
                    else:
                        lines = ""
                    doc_dic = {'score': score, 'phrases': entities, 'id': id, 'lines': lines}
                    tmp_r.append(doc_dic)
                r_list.extend(tmp_r)

        r_list.sort(key=lambda x: x.get('score'), reverse=True)
        return r_list
    except:
        print(f"es entity error: {entitie_subsets}")
        return []



# in case there is no co-existing all phrases in one doc:
# search in pairs and merge
def has_overlap(str_l):
    has_dup = False
    for i in str_l:
        if len(list(filter(lambda x: (i in x and i != x), str_l))) > 0:
            has_dup = True
    return has_dup

def search_subsets(phrases):
    result = []
    phrase_set = set(phrases)
    covered_set = set([])
    searched_subsets = []
    l = len(phrases)
    if l > 1:
        l = 5 if l > 4 else l + 1
        for i in reversed(range(2, l)):
            sub_sets = itertools.combinations(phrases, i)
            for s in sub_sets:
                # if isSubset(s, searched_subsets) or has_overlap(s):
                if has_overlap(s):
                    # print("skip ", s)
                    continue

                r = search_doc(list(s))
                if len(r) > 0:
                    # print("has hits ", s)
                    covered_set = covered_set | set(s)
                    # searched_subsets.append(s)
                    result = result + r
                    # if phrase_set == covered_set:
                    #     return result, set([])
                # else:
                    # print("no hits ", s)

    not_covered = set(phrases) - covered_set
    # need to search those entities without any hits as well, for example e = phrases - covered_set

    return result, not_covered

def test():
    phrases = ['1']

def search_and_merge(entities, nouns):
    # print("entities:", entities)
    # print("nouns:", nouns)
    result0, not_covered0 = search_subsets(merge_chunks_with_entities(nouns, entities))

    result1, not_covered1 = search_subsets(entities)
    # print("done with r1")
    result2, not_covered2 = search_subsets(nouns)
    # print("done with r2")
    result3 = search_single_entity(not_covered1)
    # print("done with r3")
    result4 = search_single_entity(not_covered2)
    # print("done with r4")
    result5 = search_single_entity(entities)
    # print("done with r5")
    result6 = search_single_entity(nouns)

    result7 = search_single_entity(not_covered0)
    # print("done with r6")
    return merge_result(result0 + result1 + result3 + result2 + result4 + result5 + result6 + result7)


def search_and_merge2(entities_and_nouns):
    result0, not_covered0 = search_subsets(entities_and_nouns)
    result7 = search_single_entity(not_covered0)
    return merge_result(result0 + result7)


def search_and_merge4(entities, nouns):
    def get_subsets(phrase_l):
        all_subsets = []
        l = len(phrase_l)
        if l == 0:
            return all_subsets
        l = 4 if l > 3 else l
        for i in reversed(range(1, l + 1)):
            sub_sets = [list(c) for c in itertools.combinations(phrase_l, i)]
            all_subsets.extend(sub_sets)
        filtered = []
        for s in all_subsets:
            has_dup = False
            for item in s:
                if len(list(filter(lambda x: item in x and x != item, s))) > 0:
                    has_dup = True
                    break
            if not has_dup:
                filtered.append(s)
        return filtered

    if len(entities) > 0 and len(nouns) > 0:
        result_media = search_media(entities)
        entity_subsets = get_subsets(entities)
        result = search_entity_combinations(entity_subsets)
        result.extend(result_media)
        # for i in entities:
        #     if i in nouns:
        #         nouns.remove(i)
        nouns_subsets = get_subsets(nouns)
        covered_set = set()
        if len(entity_subsets) > 0 and len(nouns_subsets) > 0:
            product = itertools.product(entity_subsets, nouns_subsets)
            for i in product:
                new_subset = []
                new_subset.extend(i[0])
                new_subset.extend(i[1])
                if has_overlap(new_subset):
                    continue
                r = search_doc(new_subset)
                if len(r) > 0:
                    covered_set = covered_set | set(new_subset)
                    result.extend(r)
    elif len(entities) == 0 and len(nouns) > 0:
        result = search_and_merge2(nouns)
    elif len(nouns) == 0 and len(entities) > 0:
        result_media = search_media(entities)
        entity_subsets = get_subsets(entities)
        result = search_entity_combinations(entity_subsets)
        result.extend(result_media)
    else:
        return []
    # merged = merge_result(result)
    # truncated = truncate_result(result)
    # merged = merge_result2(truncated)
    merged = merge_result2(result)
    return merged


# entity, keywords
def search_and_merge3(entity_context_l):
    context_r = []
    for c in entity_context_l:
        re_tmp = search_doc_dbpedia_context(c)
        context_r.extend(re_tmp)
    return merge_result(context_r)


def merge_result(result):
    merged = []
    for i in result:
        score = i['score']
        id = i['id']
        phrases = i['phrases']
        if not has_doc_id(id, merged):
            merged.append(i)
        else:
            for m in merged:
                if id == m.get("id") and phrases == m["phrases"] and score > m.get("score"):
                    merged.remove(m)
                    merged.append(i)
                    break
    merged.sort(key=lambda x: x.get('score'), reverse=True)
    return merged


def merge_result2(result):
    ids_l = []
    for i in result:
        if i['id'] not in ids_l:
            ids_l.append(i['id'])
    docs2phrases = dict()
    for i in result:
        i_id = i['id']
        i_idx = ids_l.index(i_id)
        if i_idx in docs2phrases:
            # for record in docs2phrases[i_idx]:
            #     if record['phrases'] == i['phrases'] and record['score'] <= i['score']:
            #         docs2phrases[i_idx].remove(record)
            #         break
            docs2phrases[i_idx].append(i)
        else:
            docs2phrases.update({i_idx: [i]})
    merged = []
    for d in docs2phrases:
        new_score = reduce(lambda x, y: x + y, [i['score'] for i in docs2phrases[d]])
        new_phrases = reduce(lambda x, y:  list(set(x) | set(y)), [i['phrases'] for i in docs2phrases[d]])
        new_r = {'score': new_score, 'id': ids_l[d], 'phrases': new_phrases, 'lines': ""}
        merged.append(new_r)
    merged.sort(key=lambda x: x.get('score'), reverse=True)
    return merged


def merge_result3(result):
    ids_l = []
    for i in result:
        if i['id'] not in ids_l:
            ids_l.append(i['id'])
    docs2phrases = dict()
    for i in result:
        i_id = i['id']
        i_idx = ids_l.index(i_id)
        if i_idx in docs2phrases:
            for record in docs2phrases[i_idx]:
                if record['phrases'] == i['phrases'] and record['score'] <= i['score']:
                    docs2phrases[i_idx].remove(record)
                    break
            docs2phrases[i_idx].append(i)
        else:
            docs2phrases.update({i_idx: [i]})
    merged = []
    for d in docs2phrases:
        new_score = 0
        new_phrases = set()
        for i in docs2phrases[d]:
            if len(set(i['phrases']) | new_phrases) > len(new_phrases) or len(i['phrases']) == 1:
                new_score += i['score']
                new_phrases = set(i['phrases']) | new_phrases
        new_r = {'score': new_score, 'id': ids_l[d], 'phrases': list(new_phrases), 'lines': ""}
        merged.append(new_r)
    merged.sort(key=lambda x: x.get('score'), reverse=True)
    return merged



def truncate_result(merged):
    phrases_l = []
    for i in merged:
        if i['phrases'] not in phrases_l:
            phrases_l.append(i['phrases'])
    phrases2docs = dict()
    for i in merged:
        i_p = i['phrases']
        i_idx = phrases_l.index(i_p)
        if i_idx in phrases2docs:
            phrases2docs[i_idx].append(i)
        else:
            phrases2docs.update({i_idx: [i]})
    phrase_top_k = 4
    truncated = []
    for p in phrases2docs:
        p_docs = phrases2docs[p]
        p_docs.sort(key=lambda x: x.get('score'), reverse=True)
        truncated.extend(p_docs[:phrase_top_k])
    truncated.sort(key=lambda x: x.get('score'), reverse=True)
    return truncated


def has_phrase_covered(phrase_set1, phrase_set2):
    covered = True
    for i in phrase_set1:
        covered = False
        for j in phrase_set2:
            if i in j:
                covered = True
                break
        if not covered:
            return covered
    return covered


def has_doc_id(id, merged_list):
    for i in merged_list:
        if i.get("id") == id:
            return True
    return False


def isSubset(subset, big_set):
    sub = set(subset)
    for i in big_set:
        if sub.issubset(i):
            return True
        else:
            return False


# in case there is no phrase pairs existing in any doc:
# find common attributes for each entity/ontology and search [<entity, attibute>] and merge
def search_single_entity(phrases):
    result = []
    sub_set = itertools.combinations(phrases, 1)
    for s in sub_set:
        r = search_doc(s)
        if len(r) > 0:
            # print("has hits ", s)
            result = result + r
        # else:
        #     print("no hits ", s)
    return result


def test_search_claim(claim):
    nouns, entities = split_claim_spacy(claim)
    cap_phrases = split_claim_regex(claim)
    ents = [i[0] for i in entities]
    nouns = list(set(nouns) | set(cap_phrases))
    first = search_and_merge(ents, nouns)
    print(first)


if __name__ == '__main__':
    search_entity_combinations(['Ireland', 'Saxony'])
    search_doc_id("Pablo_Andrés_González")
    # t = normalize("""["Mariano Gonza\u0301lez", "Mariano Gonza\u0301lez"]""")

    # search_doc_id_and_keywords("Cordillera Domeyko", ["parent Mountain Peak", "Andes"])
    # search_doc_id_and_keywords("Pablo_Andrés_González", ["brother", "Mariano González"])
    # t = read_json(config.PRO_ROOT / "src/ES/wikipage_mapping.json")
    # print(test_search_id("Trouble with the Curve"))
    # print(has_phrase_covered(['a c', 'b', 'c'], ['a b c', 'c d']))
    # print(search_doc(['Fox 2000 Pictures', 'Soul Food']))
    # test_search_claim("Lisa Kudrow was in Romy and Michele's High School Reunion (1997), The Opposite of Sex (1998), Analyze This (1999) and its sequel Analyze That (2002), Dr. Dolittle 2 (2001), Wonderland (2003), Happy Endings (2005), P.S. I Love You (2007), Bandslam (2008), Hotel for Dogs (2009), Easy A (2010), Neighbors (2014), its sequel Neighbors 2: Sorority Rising (2016) and The Girl on the Train (2016).")
    # get_all_doc_ids()
    # get_all_sent_by_doc_id_es("Andre_Markgraaff")
    # get_evidence_es("Andre_Markgraaff", 13)

