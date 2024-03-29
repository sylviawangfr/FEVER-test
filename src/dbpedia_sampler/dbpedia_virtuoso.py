import re
from datetime import datetime

from SPARQLWrapper import SPARQLWrapper, JSON
from dbpedia_sampler.uri_util import *
from utils.tokenizer_simple import count_words
from memory_profiler import profile
import config
import log_util

DEFAULT_GRAPH = "http://dbpedia.org"
PREFIX_DBO = "http://dbpedia.org/ontology/"
PREFIX_SCHEMA = "http://www.w3.org/2001/XMLSchema"
PREFIX_SUBCLASSOF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
PREFIX_DBR = "http://dbpedia.org/resource/"
PREFIX_TYPE_OF = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
PREFIX_TYPE = 'http://purl.org/dc/terms/subject'
RECORD_LIMIT = 200

log = log_util.get_logger('dbpedia_virtuoso')


def get_triples(query_str):
    # log.debug("virtuoso query str: " + query_str)
    start = datetime.now()
    sparql = SPARQLWrapper(config.DBPEDIA_GRAPH_URL, defaultGraph=DEFAULT_GRAPH)
    sparql.setTimeout(5)
    sparql.setQuery(query_str)
    sparql.setReturnFormat(JSON)
    triples = []
    try:
        response = sparql.query()
        results = response.convert()
        if len(results["results"]["bindings"]) > 500:
            log.debug('extra large bindings in DBpedia, ignore')
            return triples
        for record in results["results"]["bindings"]:
            if 0 < len(record['object']['value']) < 100:
                tri = dict()
                tri['subject'] = record['subject']['value']
                tri['relation'] = record['relation']['value']
                tri['object'] = record['object']['value']
                if 'datatype' in record['object']:
                    tri['datatype'] = record['object']['datatype'].split('#')[-1]
                else:
                    tri['datatype'] = 'uri'

                triples.append(tri)
            else:
                log.debug(f"extra long obj or empty str: {record['object']['value']}")
        # print(json.dumps(triples, indent=4))
        response.response.close()
        log.debug(f"sparql time: {(datetime.now() - start).seconds}")
        return triples
    except Exception as err:
        log.warning("failed to query dbpedia virtuoso...")
        log.error(err)
        log.debug(f"sparql time: {(datetime.now() - start).seconds}")
        return triples


# def get_categories(resource):
#     query_str_inbound = f"SELECT distinct (<{resource}> AS ?subject) (<{PREFIX_TYPE_OF}> AS ?relation) ?object " \
#         "FROM <http://dbpedia.org> WHERE {" \
#         f"<{resource}> <{PREFIX_TYPE_OF}> ?object ." \
#         "filter contains(str(?object), 'http://dbpedia.org/ontology/') " \
#         "filter (!contains(str(?object), \'wiki\'))} LIMIT 500"
#     tris = get_triples(query_str_inbound)
#     for tri in tris:
#         obj_split = uri_short_extract(tri['object'])
#         tri['keywords'] = [obj_split]
#         tri['keyword1'] = 'type'
#         tri['keyword2'] = obj_split
#     return tris


def get_categories2(resource):
    query_str_inbound = f"PREFIX PREFIX_TYPE_OF: <{PREFIX_TYPE_OF}> "\
        f"PREFIX PREFIX_TYPE: <{PREFIX_TYPE}> " \
        f"SELECT distinct (<{resource}> AS ?subject) ?relation ?object " \
        "FROM <http://dbpedia.org> WHERE { " \
        f"<{resource}> ?relation ?object . " \
        "filter (?relation in (" \
        f"PREFIX_TYPE_OF:, PREFIX_TYPE:)) " \
        "filter (!contains(str(?object), \'wiki\'))} LIMIT 500"
    tris = get_triples(query_str_inbound)
    for tri in tris:
        obj_split = uri_short_extract(tri['object'])
        tri['keywords'] = [obj_split]
        tri['keyword1'] = 'type'
        tri['keyword2'] = obj_split
    return tris


# def get_categories_one_hop_child(ontology_uri):
#     query_str_inbound = f"SELECT distinct ?subject (<{PREFIX_SUBCLASSOF}> AS ?relation) (<{ontology_uri}> AS ?object) " \
#         "FROM <http://dbpedia.org> WHERE {" \
#         f"?subject <{PREFIX_SUBCLASSOF}> <{ontology_uri}> .}} LIMIT 500"
#     tris = get_triples(query_str_inbound)
#     for tri in tris:
#         subj_split = uri_short_extract(tri['subject'])
#         tri['keywords'] = [subj_split]
#     return tris


# def get_categories_one_hop_parent(ontology_uri):
#     query_str_outbound = f"SELECT distinct (<{ontology_uri}> AS ?subject) (<{PREFIX_SUBCLASSOF}> AS ?relation) ?object " \
#         "FROM <http://dbpedia.org> WHERE {" \
#         f"<{ontology_uri}> <{PREFIX_SUBCLASSOF}> ?object .}} LIMIT 500"
#     tris = get_triples(query_str_outbound)
#     for tri in tris:
#         obj_split = uri_short_extract(tri['object'])
#         tri['keywords'] = [obj_split]
#     return tris


# def get_properties(resource_uri):
#     query_str = f"SELECT distinct (<{resource_uri}> AS ?subject) ?relation ?object " \
#         "FROM <http://dbpedia.org> WHERE {" \
#         f"<{resource_uri}> ?relation ?object . " \
#         "filter contains(str(?relation), \'http://dbpedia.org/property/\')} LIMIT 500"
#     tris = get_triples(query_str)
#     to_delete = []
#     for tri in tris:
#         obj_split = uri_short_extract(tri['object'].split('^^')[0])
#         if does_reach_max_length(obj_split):
#             to_delete.append(tri)
#             continue
#         else:
#             rel_split = uri_short_extract(tri['relation'].split('^^')[0])
#             tri['keywords'] = [rel_split, obj_split]
#             tri['keyword1'] = rel_split
#             tri['keyword2'] = obj_split
#     for i in to_delete:
#         tris.remove(i)
#     return tris


def get_resource_wiki_page(resource_uri):
    query_str = f"PREFIX dbr: <{PREFIX_DBR}> " \
        f"SELECT distinct (<{resource_uri}> AS ?subject) " \
        f"(<http://xmlns.com/foaf/0.1/isPrimaryTopicOf> AS ?relation) ?object " \
        "FROM <http://dbpedia.org> WHERE { " \
        f"<{resource_uri}> <http://xmlns.com/foaf/0.1/isPrimaryTopicOf> ?object . " \
        "} LIMIT 10"
    tris = get_triples(query_str)
    wikis = [i['object'] for i in tris]
    return wikis


# def get_ontology_linked_values_outbound(resource_uri):
#     query_str = f"PREFIX dbo: <{PREFIX_DBO}> " \
#         f"SELECT distinct (<{resource_uri}> AS ?subject) ?relation ?object " \
#         "FROM <http://dbpedia.org> WHERE { " \
#         f"<{resource_uri}> ?relation ?object . " \
#         "filter contains(str(?relation), 'ontology') " \
#         "filter (?relation not in (" \
#         "dbo:thumbnail, " \
#         "dbo:abstract, " \
#         "dbo:wikiPageID, " \
#         "dbo:wikiPageRevisionID, " \
#         "dbo:wikiPageExternalLink))} LIMIT 500"
#     tris = get_triples(query_str)
#     to_delete = []
#     for tri in tris:
#         obj_split = uri_short_extract(tri['object'])
#         if does_reach_max_length(obj_split):
#             to_delete.append(tri)
#             continue
#         else:
#             rel_split = uri_short_extract(tri['relation'])
#             tri['keywords'] = [rel_split, obj_split]
#             tri['keyword1'] = rel_split
#             tri['keyword2'] = obj_split
#     # print(f"outbound re: {len(tris)}")
#     for i in to_delete:
#         tris.remove(i)
#     return tris

#
# def get_ontology_linked_values_inbound(resource_uri):
#     query_str = f"PREFIX dbo: <{PREFIX_DBO}> " \
#         f"SELECT distinct ?subject ?relation (<{resource_uri}> AS ?object) " \
#         "FROM <http://dbpedia.org> WHERE { " \
#         f"?subject ?relation <{resource_uri}> . " \
#         "filter contains(str(?relation), 'ontology') " \
#         "filter (!contains(str(?relation), \'wiki\'))" \
#         "filter (?relation not in (" \
#         "dbo:thumbnail, " \
#         "dbo:abstract))} LIMIT 500"
#     tris = get_triples(query_str)
#     for tri in tris:
#         rel_split = uri_short_extract(tri['relation'])
#         subj_split = uri_short_extract(tri['subject'])
#         tri['keywords'] = [subj_split, rel_split]
#         tri['keyword1'] = rel_split
#         tri['keyword2'] = subj_split
#     log.debug(f"inbound re: {len(tris)}")
#     return tris


def get_outbounds2(resource_uri):
    # "filter (!contains(str(?object), \'wikidata\')) " \
    query_str = f"SELECT distinct (<{resource_uri}> AS ?subject) ?relation ?object " \
        "FROM <http://dbpedia.org> WHERE { " \
        f" <{resource_uri}> ?relation ?object . " \
        "filter (regex(?relation, " \
        "'^(?!.*?(wiki|comment|label|sameAs|exactMatch|abstract|align|image|photo|wasDerivedFrom|logo|width|height|dimension|isPrimaryTopicOf|property/[0-9]+)).*$')) " \
        "filter(regex(?object, '^(?!.*?(Thing|Agent|wikidata)).*$')) " \
        "} LIMIT 500"
    tris = get_triples(query_str)
    to_delete = []
    for tri in tris:
        obj_split = uri_short_extract(tri['object'])
        rel_split = uri_short_extract(tri['relation'])
        if obj_split == '' or rel_split == '' or does_reach_max_length(obj_split):
            to_delete.append(tri)
            continue
        else:
            if rel_split == 'subject':
                obj_split = obj_split.replace('Category ', '')
                tri['keywords'] = [obj_split]
                tri['keyword1'] = 'category'
                tri['keyword2'] = obj_split
                continue
            if rel_split == 'see Also':
                tri['keywords'] = [obj_split]
                tri['keyword1'] = 'see also'
                tri['keyword2'] = obj_split
                continue
            tri['keywords'] = [rel_split, obj_split]
            tri['keyword1'] = rel_split
            tri['keyword2'] = obj_split
    log.debug(f"outbound re: {len(tris)}")
    for i in to_delete:
        tris.remove(i)
    return tris


# @profile
# def get_outbounds(resource_uri):
#     query_str_outbound = f"PREFIX dbo: <http://dbpedia.org/ontology/> " \
#         f"PREFIX dbp: <http://dbpedia.org/property/> " \
#         f"SELECT distinct (<{resource_uri}> AS ?subject) ?relation ?object " \
#         "FROM <http://dbpedia.org> WHERE { " \
#         f"<{resource_uri}> ?relation ?object . " \
#         "filter (!contains(str(?relation), 'wiki')) " \
#         "filter (!contains(str(?object), 'http://www.w3.org/2002/07/owl')) " \
#         "filter (!contains(str(?object), 'http://www.wikidata.org/entity')) " \
#         "filter (?relation not in (dbo:thumbnail, dbo:abstract)) " \
#         "filter (?relation not in (dbp:width, dbp:icon, dbp:image, dbp:align, dbp:float, dbp:direction, dbp:imagewidth, dbp:iconWidth) " \
#         "|| contains(str(?object), 'http://dbpedia.org/resource/'))} LIMIT 500"
#     tris = get_triples(query_str_outbound)
#     to_delete = []
#     for tri in tris:
#         obj_split = uri_short_extract(tri['object'])
#         if does_reach_max_length(obj_split) or obj_split == '':
#             to_delete.append(tri)
#             continue
#         else:
#             rel_split = uri_short_extract(tri['relation'])
#             if rel_split == 'subject':
#                 obj_split = obj_split.replace('Category ', '')
#                 tri['keywords'] = [obj_split]
#                 tri['keyword1'] = 'category'
#                 tri['keyword2'] = obj_split
#                 continue
#             if rel_split == 'rdf schema see Also':
#                 tri['keywords'] = [obj_split]
#                 tri['keyword1'] = 'see also'
#                 tri['keyword2'] = obj_split
#                 continue
#             tri['keywords'] = [rel_split, obj_split]
#             tri['keyword1'] = rel_split
#             tri['keyword2'] = obj_split
#     log.debug(f"outbound re: {len(tris)}")
#     for i in to_delete:
#         tris.remove(i)
#     return tris


def get_resource_redirect(resource_uri):
    query_str_outbound = "PREFIX disambiguates: <http://dbpedia.org/ontology/wikiPageDisambiguates> " \
                         "PREFIX redirects: <http://dbpedia.org/ontology/wikiPageRedirects> " \
        f"SELECT distinct (<{resource_uri}> AS ?subject) ?relation ?object " \
                         "where { " \
        f"<{resource_uri}> ?relation ?object. " \
                         "filter (?relation in (disambiguates:, redirects:)) " \
                         "} LIMIT 5"
    tris = get_triples(query_str_outbound)
    objs = []
    for tri in tris:
        obj_split = uri_short_extract2(tri['object'])
        objs.append(obj_split)
    objs = list(set(objs))
    return objs


def get_disambiguates_outbounds2(resource_uri):
    query_str_outbound = f"PREFIX dbo: <http://dbpedia.org/ontology/> " \
        "PREFIX disambiguates: <http://dbpedia.org/ontology/wikiPageDisambiguates> " \
        "PREFIX redirects: <http://dbpedia.org/ontology/wikiPageRedirects> " \
        "SELECT distinct ?subject ?relation ?object where { " \
        f"<{resource_uri}> ?x ?subject. " \
        "?subject ?relation ?object. " \
        "filter (?x in (disambiguates:, redirects:)) " \
        "filter (regex(?relation, " \
        "'^(?!.*?(wiki|comment|label|sameAs|exactMatch|abstract|align|image|photo|wasDerivedFrom|isPrimaryTopicOf|property/[0-9]+)).*$')) " \
        "filter(regex(?object, '^(?!.*?(Thing|Agent|wikidata)).*$')) " \
        "} LIMIT 500"
    tris = get_triples(query_str_outbound)
    to_delete = []
    for tri in tris:
        obj_split = uri_short_extract(tri['object'])
        if does_reach_max_length(obj_split) or obj_split == '':
            to_delete.append(tri)
            continue
        else:
            rel_split = uri_short_extract(tri['relation'])
            if rel_split == 'subject':
                obj_split = obj_split.replace('Category ', '')
                tri['keywords'] = [obj_split]
                tri['keyword1'] = 'category'
                tri['keyword2'] = obj_split
            if rel_split == 'rdf schema see Also':
                tri['keywords'] = [obj_split]
                tri['keyword1'] = 'see also'
                tri['keyword2'] = obj_split
                continue
            tri['keywords'] = [rel_split, obj_split]
            tri['keyword1'] = rel_split
            tri['keyword2'] = obj_split
    log.debug(f"outbound re: {len(tris)}")
    for i in to_delete:
        tris.remove(i)
    return tris


# def get_disambiguates_outbounds(resource_uri):
#     query_str_outbound = f"PREFIX dbo: <http://dbpedia.org/ontology/> " \
#         "PREFIX disambiguates: <http://dbpedia.org/ontology/wikiPageDisambiguates> " \
#         "PREFIX redirects: <http://dbpedia.org/ontology/wikiPageRedirects> " \
#         "SELECT distinct ?subject ?relation ?object where { " \
#         f"<{resource_uri}> ?x ?subject. " \
#         "?subject ?relation ?object. " \
#         "filter (?x in (disambiguates:, redirects:)) " \
#         "filter (!contains(str(?relation), 'wiki')) " \
#         "filter (?relation not in (dbo:thumbnail, dbo:abstract)) " \
#         "filter (contains(str(?relation), 'ontology') " \
#         "|| contains(str(?object), 'http://dbpedia.org/resource/') " \
#         "|| contains(str(?relation), 'http://dbpedia.org/property/'))} " \
#         "LIMIT 500"
#     tris = get_triples(query_str_outbound)
#     to_delete = []
#     for tri in tris:
#         obj_split = uri_short_extract(tri['object'])
#         if does_reach_max_length(obj_split) or obj_split == '':
#             to_delete.append(tri)
#             continue
#         else:
#             rel_split = uri_short_extract(tri['relation'])
#             if rel_split == 'subject':
#                 obj_split = obj_split.replace('Category ', '')
#                 tri['keywords'] = [obj_split]
#                 tri['keyword1'] = 'category'
#                 tri['keyword2'] = obj_split
#             if rel_split == 'rdf schema see Also':
#                 tri['keywords'] = [obj_split]
#                 tri['keyword1'] = 'see also'
#                 tri['keyword2'] = obj_split
#                 continue
#             tri['keywords'] = [rel_split, obj_split]
#             tri['keyword1'] = rel_split
#             tri['keyword2'] = obj_split
#     log.debug(f"outbound re: {len(tris)}")
#     for i in to_delete:
#         tris.remove(i)
#     return tris


# def get_inbounds(resource_uri):
#     query_str_inbound = f"PREFIX dbo: <http://dbpedia.org/ontology/> " \
#         f"SELECT distinct ?subject ?relation (<{resource_uri}> AS ?object) " \
#         "FROM <http://dbpedia.org> " \
#         f"WHERE {{ ?subject ?relation <{resource_uri}> . " \
#         "filter (!contains(str(?relation), 'wiki')) " \
#         "filter (?relation not in (dbo:thumbnail, dbo:abstract)) " \
#         "filter (contains(str(?relation), 'ontology') " \
#         "|| contains(str(?subject), 'http://dbpedia.org/resource/') " \
#         "|| contains(str(?relation), 'http://dbpedia.org/property/'))} LIMIT 500"
#     tris = get_triples(query_str_inbound)
#     for tri in tris:
#         rel_split = uri_short_extract(tri['relation'])
#         subj_split = uri_short_extract(tri['subject'])
#         if does_reach_max_length(subj_split):
#             print('here')
#         tri['keywords'] = [subj_split, rel_split]
#     return tris


# def get_one_hop_resource_inbound(resource_uri):
#     query_str_inbound = f"SELECT distinct ?subject ?relation (<{resource_uri}> AS ?object) " \
#         f"WHERE {{ ?subject ?relation <{resource_uri}> . " \
#         "filter (!contains(str(?relation), \'wiki\')) " \
#         "filter (!contains(str(?relation), \'ontology\')) " \
#         "filter contains(str(?subject), \'http://dbpedia.org/resource/\')} LIMIT 500"
#     tris = get_triples(query_str_inbound)
#     for tri in tris:
#         rel_split = uri_short_extract(tri['relation'])
#         subj_split = uri_short_extract(tri['subject'])
#         if does_reach_max_length(subj_split):
#             print('here')
#         tri['keywords'] = [subj_split, rel_split]
#     return tris


def get_one_hop_resource_outbound(resource_uri):
    query_str_outbound = f"SELECT distinct (<{resource_uri}> AS ?subject) ?relation ?object " \
        f"WHERE {{ <{resource_uri}> ?relation ?object . " \
        "filter (!contains(str(?relation), \'wiki\')) " \
        "filter (!contains(str(?relation), \'ontology\')) " \
        "filter contains(str(?object), \'http://dbpedia.org/resource/\')} LIMIT 500"
    tris = get_triples(query_str_outbound)
    for tri in tris:
        rel_split = uri_short_extract(tri['relation'])
        obj_split = uri_short_extract(tri['object'])
        if does_reach_max_length(obj_split):
            print('here')
        tri['keywords'] = [rel_split, obj_split]
        tri['keyword1'] = rel_split
        tri['keyword2'] = obj_split
    return tris


def does_reach_max_length(text):
    if count_words(text) > 25:
        return True


if __name__ == "__main__":
    # res = "http://dbpedia.org/resource/Magic_Johnson"
    res = "http://dbpedia.org/resource/Bombay"
    # res = get_outbounds('http://dbpedia.org/resource/Charlie_Chaplin')
    get_resource_redirect(res)
    get_outbounds2(res)
    # get_categories2(res)
    # print(get_disambiguates_outbounds2(res))
    # print(get_properties(res))
    # on = "http://dbpedia.org/ontology/City"
    # o1 = get_categories_one_hop_child(on)
    # o2 =get_categories_one_hop_parent(on)
    # get_properties(res)
    # get_one_hop_resource_inbound(res)
    # get_one_hop_resource_outbound(res)
    # get_ontology_linked_values_inbound(res)
    # get_ontology_linked_values_outbound(res)

    # get_keyword(re)
    # str = ['birthPlace', 'USA', 'Magic_Johnson', '112.3', 'USA_flag']
    # on = "http://dbpedia.org/resource/Los_Angeles_Lakers"
    # t = get_one_hop_resource_inbound(res)
    # print(t)
