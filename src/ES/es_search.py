from elasticsearch import Elasticsearch as es
from elasticsearch_dsl import Search, Q
import config
import itertools

client = es([{'host': config.ELASTIC_HOST, 'port': config.ELASTIC_PORT}])


# ES match_phrase on entities
def search_doc(phrases):
    search = Search(using=client, index=config.WIKIPAGE_INDEX)
    match_phases = [Q('match_phrase', lines=ph) for ph in phrases]
    search = search.query('bool', must=match_phases).\
    highlight('lines', number_of_fragments=0).\
    sort({'_score': {"order": "desc"}}).\
    source(include=['id'])[0:5]

    response = search.execute()
    r_list = []

    for hit in response['hits']['hits']:
        score = hit['_score']
        id = hit['_source']['id']
        lines = hit['highlight']['lines'][0]
        lines = lines.replace("</em> <em>", " ")
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
        for i in reversed(range(2,l)):
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
                        return result
                else:
                    print("no hits ", s)

    not_covered = set(phrases) - covered_set
# need to search those entities without any hits as well, for example e = phrases - covered_set
    if l == 1 or not_covered != []:
        result = result + connect_search_entity(covered_set, not_covered)

    return result


def search_and_merge(entities, nouns):
    ents_list = [i[0] for i in entities]
    result = search_subsets(ents_list) + search_subsets(nouns)
    return merge_result(result)


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
def connect_search_entity(subset1, sebset2):
    return []


def test():
    result1 = search_subsets(['Colin Kaepernick', 'a starting quarterback', 'the 49ers', '63rd season', 'the National Football League'])
    print("search nouns:")
    for i in result1:
        print(i)
    result2 = search_subsets(['Colin Kaepernick', 'the 49ers 63rd season', 'the National Football League'])
    print("search entities:")
    for i in result2:
        print(i)
    result = result1 + result2

    merged = merge_result(result)
    print("merged:")
    for i in merged:
        print(i)



if __name__ == '__main__':
    # print(search_doc(['Fox 2000 Pictures', 'Soul Food']))
    test()