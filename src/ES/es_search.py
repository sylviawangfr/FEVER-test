from elasticsearch import Elasticsearch as es
from elasticsearch_dsl import Search, Q
from elasticsearch_dsl.query import MultiMatch
import config
import itertools
from utils.tokenizer_simple import *

client = es([{'host': config.ELASTIC_HOST, 'port': config.ELASTIC_PORT}])


# ES match_phrase on entities
def search_doc(phrases):
    search = Search(using=client, index=config.WIKIPAGE_INDEX)
    must = []
    # should = []
    for ph in phrases:
        if ph.startswith('the ') and ph.startswith("a "):
            ph = ph.split(' ', 1)[1]
        # must.append({'match_phrase': {'lines': ph}})
        # should.append({'match_phrase': {'id': {'query': ph, 'boost': 2}}})
        must.append({'multi_match': {'query': ph, 'fields': ['id^2', 'lines']}})

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
        doc_dic = {'score': score, 'phrases': phrases, 'id': id, 'lines': lines}
        r_list.append(doc_dic)

    return r_list


# in case there is no co-existing all phrases in one doc:
# search in pairs and merge

def search_subsets(phrases):
    l = len(phrases)
    result = []
    phrase_set = set(phrases)
    covered_set = set([])
    searched_subsets = []

    if l > 1:
        for i in reversed(range(2, l + 1)):
            sub_set = itertools.combinations(phrases, i)
            for s in sub_set:
                if isSubset(s, searched_subsets):
                    print("skip ", s)
                    continue

                r = search_doc(list(s))
                if len(r) > 0:
                    print("has hits ", s)
                    covered_set = covered_set | set(s)
                    searched_subsets.append(s)
                    result = result + r
                    if phrase_set == covered_set:
                        return result, set([])
                else:
                    print("no hits ", s)

    not_covered = set(phrases) - covered_set
    # need to search those entities without any hits as well, for example e = phrases - covered_set

    return result, not_covered


def search_and_merge(entities, nouns):
    # print("entities:", entities)
    # print("nouns:", nouns)
    ents_list = [i[0] for i in entities]
    result1, not_covered1 = search_subsets(ents_list)
    # print("done with r1")
    result2, not_covered2 = search_subsets(nouns)
    # print("done with r2")
    result3 = search_single_entity(not_covered1)
    # print("done with r3")
    result4 = search_single_entity(not_covered2)
    # print("done with r4")
    result5 = search_single_entity(ents_list)
    # print("done with r5")
    result6 = search_single_entity(nouns)
    # print("done with r6")

    return merge_result(result1 + result3 + result2 + result4 + result5 + result6)


def merge_result(result):
    merged = []
    for i in result:
        score, ph, id, lines = i.values()
        if not has_doc_id(id, merged):
            merged.append(i)
        else:
            ph_set = set(ph)
            for m in merged:
                m_set = set(m.get("phrases"))
                if m_set.issubset(ph_set) and id == m.get("id"):
                    merged.remove(m)
                    merged.append(i)
    return merged


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
            print("has hits ", s)
            result = result + r
        else:
            print("no hits ", s)
    return result


def test():
    claim = "Hot Right Now is mistakenly attributed to DJ Fresh."
    nouns, entities = split_claim_spacy(claim)
    cap_phrases = split_claim_regex(claim)

    ents = [i[0] for i in entities]

    #
    print("search nouns:", nouns)
    # result1, c1 = search_subsets(nouns)
    # for i in result1:
    #     print(i)
    #
    print("search entities:", ents)
    # result2, c2 = search_subsets(ents)
    # for i in result2:
    #     print(i)
    #
    # print("search single nouns:", nouns)
    # result3 = search_single_entity(nouns)
    # for i in result3:
    #     print(i)

    print("search single entites:", ents)
    result4 = search_single_entity(ents)
    for i in result4:
        print(i)

    result = result4

    print("all results:")
    for i in result:
        print(i)

    merged = merge_result(result)
    print("merged:")
    for i in merged:
        print(i)


if __name__ == '__main__':
    # print(search_doc(['Fox 2000 Pictures', 'Soul Food']))
    test()
