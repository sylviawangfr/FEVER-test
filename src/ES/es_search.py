import itertools

from elasticsearch import Elasticsearch as es
from elasticsearch_dsl import Search, Q

from utils.file_loader import *
from utils.tokenizer_simple import *
from dbpedia_sampler.sentence_util import merge_chunks_with_entities

client = es([{'host': config.ELASTIC_HOST, 'port': config.ELASTIC_PORT,
              'timeout': 60, 'max_retries': 5, 'retry_on_timeout': True}])


# ES match_phrase on entities
def search_doc(phrases):
    try:
        search = Search(using=client, index=config.WIKIPAGE_INDEX)
        must = []
        should = []
        for ph in phrases:
            ph = remove_the_a(ph)
            # should.append({'multi_match': {'query': ph, "type": "most_fields",
            #                                'fields': ['id^2', 'lines'], 'analyzer': 'underscore_analyzer'}})
            must.append({'multi_match': {'query': ph, "type": "phrase",
                                         'fields': ['id^2', 'lines'], 'slop': 3, 'analyzer': 'underscore_analyzer'}})

        search = search.query(Q('bool', must=must, should=should)). \
                 highlight('lines', number_of_fragments=0). \
                 sort({'_score': {"order": "desc"}}). \
                 source(include=['id'])[0:15]

        response = search.execute()
        r_list = []

        for hit in response['hits']['hits']:
            score = hit['_score']
            id = hit['_source']['id']
            if 'highlight' in hit:
                lines = hit['highlight']['lines'][0]
                lines = lines.replace("</em> <em>", " ")
            else:
                lines = ""
            doc_dic = {'score': score, 'phrases': phrases, 'id': id, 'lines': lines}
            r_list.append(doc_dic)

        return r_list
    except:
        return []

# entity, keywords
def search_doc_dbpedia_context(context_dict):
    try:
        search = Search(using=client, index=config.WIKIPAGE_INDEX)
        must = []
        should = []

        # must.append({'match_phrase': {'lines': ph}})
        for i in context_dict['keywords']:
            should.append({'multi_match': {'query': i, "type": "phrase", 'fields': ['id^2', 'lines'],
                                           'slop': 3, 'analyzer': 'underscore_analyzer'}})
        must.append({'multi_match': {'query': context_dict['entity'], "type": "phrase",
                                     'fields': ['id^2', 'lines'], 'slop': 3, 'analyzer': 'underscore_analyzer'}})

        search = search.query(Q('bool', must=must, should=should)). \
                 highlight('lines', number_of_fragments=0). \
                 sort({'_score': {"order": "desc"}}). \
                 source(include=['id'])[0:5]

        response = search.execute()
        r_list = []
        phrases = context_dict['keywords']
        phrases.append(context_dict['entity'])
        for hit in response['hits']['hits']:
            score = hit['_score']
            id = hit['_source']['id']
            if 'highlight' in hit:
                lines = hit['highlight']['lines'][0]
                lines = lines.replace("</em> <em>", " ")
            else:
                lines = ""
            doc_dic = {'score': score, 'phrases': phrases, 'id': id, 'lines': lines}
            r_list.append(doc_dic)

        return r_list
    except:
        return []


def search_doc_id_and_keywords(possible_id, keywords):
    search = Search(using=client, index=config.WIKIPAGE_INDEX)
    must = []
    should = []
    must.append(
        {'match_phrase': {'id': {'query': possible_id, 'analyzer': 'underscore_analyzer', 'boost': 2}}})
    # should.append({'match_phrase': {'lines': {'query': possible_id.replace("_", " "), 'analyzer': 'underscore_analyzer'}}})
    if len(keywords) == 2:
        relation = keywords[0]
        should.append({'match': {'lines': {'query': relation, 'analyzer': 'wikipage_analyzer'}}})
    if len(keywords) > 0:
        obj = keywords[-1]
        should.append({'match_phrase': {'lines': {'query': obj, 'slop': 3, 'analyzer': 'wikipage_analyzer'}}})
    search = search.query(Q('bool', must=must, should=should)). \
                 highlight('lines', number_of_fragments=0, fragment_size=150). \
                 sort({'_score': {"order": "desc"}}). \
                 source(include=['id'])[0:5]

    response = search.execute()
    r_list = []
    sentences_list = []
    phrases = [possible_id]
    phrases.extend(keywords)
    for hit in response['hits']['hits']:
        score = hit['_score']
        id = hit['_source']['id']
        if 'highlight' in hit:
            lines = hit['highlight']['lines'][0]
            lines = lines.replace("</em> <em>", " ")
            lines_json_l = json.loads(normalize(lines))
            match_count = [i['sentences'].count("<em>") for i in lines_json_l]
            sorted_matching_index = sorted(range(len(match_count)), key=lambda k: match_count[k],
                                           reverse=True)
            highest_match_count = -1
            for idx in sorted_matching_index:
                if match_count[idx] > 0 and match_count[idx] >= highest_match_count:
                    h_links_s = set()
                    line_num = lines_json_l[idx]['line_num']
                    sentence = lines_json_l[idx]['sentences']
                    h_links = lines_json_l[idx]['h_links']
                    sentence = sentence.replace("</em>", "").replace("<em>", "")
                    highest_match_count = match_count[idx]
                    for h in h_links:
                        h = h.replace("<em>", "").replace("</em>", "")
                        h_links_s.add(h)
                    sentences_list.append(
                            {'sid': f'{possible_id}<SENT_LINE>{line_num}', 'text': sentence, 'h_links': list(h_links_s)})
                else:
                    break
        for s in sentences_list:
            doc_dic = {'score': score, 'phrases': phrases, 'doc_id': id, 'sid': s['sid'], 'text': s['text'], 'h_links': s['h_links']}
            r_list.append(doc_dic)
    return r_list


def search_doc_id_and_keywords_in_sentences(possible_id, subject, keywords):
    search = Search(using=client, index=config.FEVER_SEN_INDEX)
    must = []
    should = []
    must.append(
        {'term': {'doc_id_keyword': possible_id}})
    must.append({'multi_match': {'query': subject,
                                 'fields': ['doc_id', 'text'], 'analyzer': 'underscore_analyzer'}})
    if len(keywords) == 2:
        relation = keywords[0]
        should.append({'match': {'text': {'query': relation, 'analyzer': 'wikipage_analyzer'}}})
    if len(keywords) > 0:
        obj = keywords[-1]
        must.append({'multi_match': {'query': obj,
                                     'fields': ['doc_id', 'text'], 'analyzer': 'underscore_analyzer'}})

    search = search.query(Q('bool', must=must, should=should)). \
                 highlight('text', number_of_fragments=0, fragment_size=150). \
                 sort({'_score': {"order": "desc"}}). \
                 source(include=['doc_id', 'sid', 'text', 'h_links'])[0:5]

    response = search.execute()
    r_list = []
    phrases = [possible_id]
    phrases.extend(keywords)
    top_n = 1
    for hit in response['hits']['hits']:
        if top_n < 1:
            break
        score = hit['_score']
        doc_id = hit['_source']['doc_id']
        sid = hit['_source']['sid']
        text = hit['_source']['text']
        h_links = list(set(json.loads(normalize(hit['_source']['h_links']))))
        doc_dic = {'doc_id': doc_id, 'score': score, 'phrases': phrases, 'sid': sid, 'text': text, 'h_links': h_links}
        r_list.append(doc_dic)
        top_n -= 1
    return r_list


def search_doc_id(possible_id):
    try:
        body = {
            "query": {
                "match_phrase": {
                    "id": {
                        "query": possible_id,
                        "analyzer": "underscore_analyzer"
                    }
                }},
            "sort": {"_score": {"order": "desc"}},
            "_source": ["id"],
            "size": 10
        }
        response = client.search(index=config.WIKIPAGE_INDEX, body=body)
        r_list = []
        for hit in response['hits']['hits']:
            score = hit['_score']
            id = hit['_source']['id']
            doc_dic = {'score': score, 'phrases': [possible_id], 'id': id, 'lines': ""}
            r_list.append(doc_dic)
        return r_list
    except Exception as e:
        print(e)
        return []


def remove_the_a(ph):
    if (not is_capitalized(ph)) and \
            (ph.lower().startswith('the ')
             or ph.lower().startswith("a ")
             or ph.lower().startswith("an ")):
        ph = ph.split(' ', 1)[1]
    return ph


def search_entity_combinations(entities):
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
        return []



# in case there is no co-existing all phrases in one doc:
# search in pairs and merge
def has_overlap(str_l):
    for i in str_l:
        if len(list(filter(lambda x: (i in x), str_l))) > 1:
            return True
    return False

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
    if len(entities) == 0:
        return search_and_merge2(nouns)
    if len(nouns) == 0:
        return search_entity_combinations(entities)

    def get_subsets(phrase_l):
        all_subsets = []
        l = len(phrase_l)
        if l == 0:
            return all_subsets
        l = 4 if l > 3 else l
        for i in reversed(range(1, l + 1)):
            sub_sets = [list(c) for c in itertools.combinations(phrase_l, i)]
            all_subsets.extend(sub_sets)
        return all_subsets

    entity_subsets = get_subsets(entities)
    nouns_subsets = get_subsets(nouns)
    covered_set = set()
    result = search_entity_combinations(entities)
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
        # not_covered = set(entities) - covered_set
        # result7 = search_single_entity(not_covered)
        # return merge_result(result + result7)
    return merge_result(result)



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
        if not has_doc_id(id, merged):
            merged.append(i)
        else:
            for m in merged:
                if id == m.get("id") and score > m.get("score"):
                    merged.remove(m)
                    merged.append(i)
                    break
    merged.sort(key=lambda x: x.get('score'), reverse=True)
    return merged


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


def test_search_id(text):
    try:
        search = Search(using=client, index=config.WIKIPAGE_INDEX)
        must = []
        if text.lower().startswith('the ') or text.lower().startswith("a ") or text.lower().startswith("an "):
            text = text.split(' ', 1)[1]
        # must.append({'match_phrase': {'lines': ph}})
        # should.append({'match_phrase': {'id': {'query': ph, 'boost': 2}}})
        must.append({'multi_match': {'query': text, "type": "phrase", 'fields': ['id^2']}})

        search = search.query(Q('bool', must=must)). \
                 highlight('lines', number_of_fragments=0). \
                 sort({'_score': {"order": "desc"}}). \
                 source(include=['id'])[0:5]

        response = search.execute()
        r_list = []

        for hit in response['hits']['hits']:
            score = hit['_score']
            id = hit['_source']['id']
            if 'highlight' in hit:
                lines = hit['highlight']['lines'][0]
                lines = lines.replace("</em> <em>", " ")
            else:
                lines = ""
            doc_dic = {'score': score, 'phrases': text, 'id': id, 'lines': lines}
            r_list.append(doc_dic)

        r_list.sort(key=lambda x: x.get('score'), reverse=True)
        return r_list
    except:
        return []

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

