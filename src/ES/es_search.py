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
        doc_dic = {'score': score, 'id': id, 'lines': lines}
        r_list.append(doc_dic)

    return r_list


# in case there is no co-existing all phrases in one doc:
# search in pairs and merge

def search_and_merge(phrases):
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


def isSubset(subset, big_set):
    sub = set(subset)
    big = set(big_set)
    for i in big:
        if sub.issubset(i):
            return True
        else:
            return False

# in case there is no phrase pairs existing in any doc:
# find common attributes for each entity/ontology and search [<entity, attibute>] and merge
def connect_search_entity(subset1, sebset2):
    return []


if __name__ == '__main__':
    # print(search_doc(['Fox 2000 Pictures', 'Soul Food']))
    for s in search_and_merge(['Colin Kaepernick', 'a starting quarterback', 'the 49ers', '63rd season', 'the National Football League']):
        print(s)
