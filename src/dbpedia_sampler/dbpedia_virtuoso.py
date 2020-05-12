from SPARQLWrapper import SPARQLWrapper, JSON
import json
import re
import validators
import config

DEFAULT_GRAPH = "http://dbpedia.org"
PREFIX_DBO = "http://dbpedia.org/ontology/"
PREFIX_SCHEMA = "http://www.w3.org/2001/XMLSchema"
PREFIX_SUBCLASSOF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
PREFIX_DBR = "http://dbpedia.org/resource/"


def get_triples(query_str):
    sparql = SPARQLWrapper(config.DBPEDIA_GRAPH_URL, defaultGraph=DEFAULT_GRAPH)
    sparql.setTimeout(5)
    sparql.setQuery(query_str)
    sparql.setReturnFormat(JSON)
    triples = []
    try:
        results = sparql.query().convert()
        for re in results["results"]["bindings"]:
            tri = dict()
            tri['subject'] = re['subject']['value']
            tri['relation'] = re['relation']['value']
            tri['object'] = re['object']['value']
            triples.append(tri)
        print(json.dumps(triples, indent=4))
        return triples
    except Exception as err:
        print("failed to query dbpedia virtuoso...")
        print(err)
        return triples


def get_categories_one_hop_child(ontology_uri):
    query_str_inbound = f"SELECT distinct ?subject (<{PREFIX_SUBCLASSOF}> AS ?relation) (<{ontology_uri}> AS ?object) " \
        "FROM <http://dbpedia.org> WHERE {" \
        f"?subject <{PREFIX_SUBCLASSOF}> <{ontology_uri}> .}}"
    tris = get_triples(query_str_inbound)
    for tri in tris:
        subj_split = keyword_extract(tri['subject'])
        tri['keywords'] = [subj_split]
    return tris


def get_categories_one_hop_parent(ontology_uri):
    query_str_outbound = f"SELECT distinct (<{ontology_uri}> AS ?subject) (<{PREFIX_SUBCLASSOF}> AS ?relation) ?object " \
        "FROM <http://dbpedia.org> WHERE {" \
        f"<{ontology_uri}> <{PREFIX_SUBCLASSOF}> ?object .}}"
    tris = get_triples(query_str_outbound)
    for tri in tris:
        obj_split = keyword_extract(tri['object'])
        tri['keywords'] = [obj_split]
    return tris

def get_properties(resource_uri):
    query_str = f"SELECT distinct (<{resource_uri}> AS ?subject) ?relation ?object " \
        "FROM <http://dbpedia.org> WHERE {" \
        f"<{resource_uri}> ?relation ?object . " \
        "filter contains(str(?relation), \'http://dbpedia.org/property/\')}"
    tris = get_triples(query_str)
    for tri in tris:
        rel_split = keyword_extract(tri['relation'].split('^^')[0])
        subj_split = keyword_extract(tri['subject'].split('^^')[0])
        tri['keywords'] = [rel_split, subj_split]
    return tris


def get_ontology_linked_values_outbound(resource_uri):
    query_str = f"PREFIX dbo: <{PREFIX_DBO}> " \
        f"SELECT distinct (<{resource_uri}> AS ?subject) ?relation ?object " \
        "FROM <http://dbpedia.org> WHERE { " \
        f"<{resource_uri}> ?relation ?object . " \
        "filter contains(str(?relation), 'ontology') " \
        "filter (?relation not in (" \
        "dbo:thumbnail, " \
        "dbo:abstract, " \
        "dbo:wikiPageID, " \
        "dbo:wikiPageRevisionID, " \
        "dbo:wikiPageExternalLink))}"
    tris = get_triples(query_str)
    for tri in tris:
        rel_split = keyword_extract(tri['relation'])
        obj_split = keyword_extract(tri['object'])
        tri['keywords'] = [rel_split, obj_split]
    return tris


def get_ontology_linked_values_inbound(resource_uri):
    query_str = f"PREFIX dbo: <{PREFIX_DBO}> " \
        f"SELECT distinct ?subject ?relation (<{resource_uri}> AS ?object) " \
        "FROM <http://dbpedia.org> WHERE { " \
        f"?subject ?relation <{resource_uri}> . " \
        "filter contains(str(?relation), 'ontology') " \
        "filter (!contains(str(?relation), \'wiki\'))"  \
        "filter (?relation not in (" \
        "dbo:thumbnail, " \
        "dbo:abstract))}"
    tris = get_triples(query_str)
    for tri in tris:
        rel_split = keyword_extract(tri['relation'])
        subj_split = keyword_extract(tri['subject'])
        tri['keywords'] = [rel_split, subj_split]
    return tris


def get_one_hop_resource_inbound(resource_uri):
    query_str_inbound = f"SELECT distinct ?subject ?relation (<{resource_uri}> AS ?object) " \
        f"WHERE {{ ?subject ?relation <{resource_uri}> . " \
        "filter (!contains(str(?relation), \'wiki\')) " \
        "filter (!contains(str(?relation), \'ontology\')) " \
        "filter contains(str(?subject), \'http://dbpedia.org/resource/\')}"
    tris = get_triples(query_str_inbound)
    for tri in tris:
        rel_split = keyword_extract(tri['relation'])
        subj_split = keyword_extract(tri['subject'])
        tri['keywords'] = [rel_split, subj_split]
    return tris


def get_one_hop_resource_outbound(resource_uri):
    query_str_outbound = f"SELECT distinct (<{resource_uri}> AS ?subject) ?relation ?object " \
        f"WHERE {{ <{resource_uri}> ?relation ?object . " \
        "filter (!contains(str(?relation), \'wiki\')) " \
        "filter (!contains(str(?relation), \'ontology\')) " \
        "filter contains(str(?object), \'http://dbpedia.org/resource/\')}"
    tris = get_triples(query_str_outbound)
    for tri in tris:
        rel_split = keyword_extract(tri['relation'])
        obj_split = keyword_extract(tri['object'])
        tri['keywords'] = [rel_split, obj_split]
    return tris



def get_similar_properties(resource1, resource2):
    pass


def get_relevant_triples(triple, keyword):
    pass


def keyword_extract(uri):
    lastword = uri.split('/')[-1]
    words = wildcase_split(lastword)
    phrases = []
    for w in words:
        ph = camel_case_split(w)
        phrases.append(' '.join([ww for ww in ph]))
    one_phrase = ' '.join(p for p in phrases)
    return one_phrase

def wildcase_split(str):
    p_l = re.findall(r'(?:\d+\.\d+)|(?:\d+)|(?:[a-zA-Z]+)', str)
    return list(filter(None, p_l))


def camel_case_split(str):
    return re.findall(r'(?:\d+\.\d+)|(?:\d+)|(?:^[a-z]+)|(?:[A-Z]+)(?:[a-z]*|[A-Z]*(?=[A-Z]|$))', str)


# "Johnson in 2007"^^<http://www.w3.org/1999/02/22-rdf-syntax-ns#langString>
def property_split(str):
    value_or_type = str.split('^^')
    return value_or_type[0]


def isURI(str):
    return validators.url(str)


if __name__ == "__main__":
    res = "http://dbpedia.org/resource/Magic_Johnson"
    on = "http://dbpedia.org/ontology/City"
    o1 = get_categories_one_hop_child(on)
    o2 =get_categories_one_hop_parent(on)
    get_properties(res)
    get_one_hop_resource_inbound(res)
    get_one_hop_resource_outbound(res)
    get_ontology_linked_values_inbound(res)
    get_ontology_linked_values_outbound(res)

    # get_keyword(re)
    # str = ['birthPlace', 'USA', 'Magic_Johnson', '112.3', 'USA_flag']
    # on = "http://dbpedia.org/resource/Los_Angeles_Lakers"
    t = get_one_hop_resource_inbound(res)
    print(t)



