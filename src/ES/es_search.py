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
            if ph.lower().startswith('the ') or ph.lower().startswith("a ") or ph.lower().startswith("an "):
                ph = ph.split(' ', 1)[1]
        # must.append({'match_phrase': {'lines': ph}})
            should.append({'multi_match': {'query': ph, "type": "most_fields", 'fields': ['id^2', 'lines']}})
            must.append({'multi_match': {'query': ph, "type": "phrase", 'fields': ['id^2', 'lines'], 'slop': 3}})

        search = search.query(Q('bool', must=must, should=should)). \
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
            doc_dic = {'score': score, 'phrases': phrases, 'id': id, 'lines': lines}
            r_list.append(doc_dic)

        return r_list
    except:
        return []


def search_doc_id(possible_id):
    try:
        search = Search(using=client, index=config.WIKIPAGE_INDEX)
        search = search.query('match_phrase', id=possible_id). \
                 sort({'_score': {"order": "desc"}}). \
                 source(include=['id'])[0:10]
        response = search.execute()
        r_list = []
        for hit in response['hits']['hits']:
            score = hit['_score']
            id = hit['_source']['id']
            doc_dic = {'score': score, 'phrases': [possible_id], 'id': id, 'lines': ""}
            r_list.append(doc_dic)
        return r_list
    except:
        return []


# in case there is no co-existing all phrases in one doc:
# search in pairs and merge

def search_subsets(phrases):
    l = len(phrases)
    result = []
    phrase_set = set(phrases)
    covered_set = set([])
    searched_subsets = []

    if l > 1:
        l = 5 if l > 4 else l + 1
        for i in reversed(range(2, l)):
            sub_sets = itertools.combinations(phrases, i)
            for s in sub_sets:
                if isSubset(s, searched_subsets):
                    # print("skip ", s)
                    continue

                r = search_doc(list(s))
                if len(r) > 0:
                    # print("has hits ", s)
                    covered_set = covered_set | set(s)
                    searched_subsets.append(s)
                    result = result + r
                    if phrase_set == covered_set:
                        return result, set([])
                # else:
                    # print("no hits ", s)

    not_covered = set(phrases) - covered_set
    # need to search those entities without any hits as well, for example e = phrases - covered_set

    return result, not_covered


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


def merge_result(result):
    merged = []
    for i in result:
        score, ph, id, lines = i.values()
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

        return r_list
    except:
        return []

if __name__ == '__main__':
    t = read_json(config.PRO_ROOT / "src/ES/wikipage_mapping.json")
    print(test_search_id("Trouble with the Curve"))
    # print(has_phrase_covered(['a c', 'b', 'c'], ['a b c', 'c d']))
    # print(search_doc(['Fox 2000 Pictures', 'Soul Food']))
    # test_search_claim("Lisa Kudrow was in Romy and Michele's High School Reunion (1997), The Opposite of Sex (1998), Analyze This (1999) and its sequel Analyze That (2002), Dr. Dolittle 2 (2001), Wonderland (2003), Happy Endings (2005), P.S. I Love You (2007), Bandslam (2008), Hotel for Dogs (2009), Easy A (2010), Neighbors (2014), its sequel Neighbors 2: Sorority Rising (2016) and The Girl on the Train (2016).")
    # get_all_doc_ids()
    # get_all_sent_by_doc_id_es("Andre_Markgraaff")
    # get_evidence_es("Andre_Markgraaff", 13)

