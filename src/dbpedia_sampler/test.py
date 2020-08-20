import xmltodict
from memory_profiler import profile
import urllib.request
import gc
import datetime as datetime
from dbpedia_sampler.bert_similarity import get_phrase_embedding
from dbpedia_sampler.dbpedia_spotlight import entity_link
from dbpedia_sampler.dbpedia_virtuoso import get_disambiguates_outbounds, get_outbounds
from dbpedia_sampler.dbpedia_lookup import lookup_resource_app_query, lookup_resource_app_label, lookup_resource_ref_count


@profile
def ip():
    r = urllib.request.urlopen('http://localhost:5005/lookup-application/api/search?label=USA')
    b = r.read().decode('utf-8')
    # x = xmltodict.parse(b)
    r.close()
    # del b
    # print(len(b))
    return b


# @profile
def test1():
    for i in range(30):
        result = ip()
        del result
        gc.collect()


def test3():
    start = datetime.now()
    p1 = ['Neil Armstrong', 'moon buggy', 'human', 'rocket', 'Naval installations', 'Military terminology']
    get_phrase_embedding(p1)
    print(f"embedding time: {(datetime.now() - start).milliseconds}")

    start = datetime.now()
    entity_link("President Obama on Monday will call for a new minimum tax rate for individuals making more "
                            "than $1 million a year to ensure that they pay at least the same percentage of their earnings "
                            "as other taxpayers, according to administration officials.")
    print(f"spotlight time: {(datetime.now() - start).milliseconds}")

    start = datetime.now()
    res1 = "http://dbpedia.org/resource/Magic_Johnson"
    res2 = "http://dbpedia.org/resource/Tap_dancer"
    get_outbounds(res1)
    get_disambiguates_outbounds(res2)
    print(f"virtuoso time: {(datetime.now() - start).milliseconds}")

    t = ['Howard Eugene Johnson', 'cultists', 'Italian', 'Even', 'Giada Pamela De Laurentiis', 'American',
         'Bloomington', 'music school', 'China capital city']
    start = datetime.now()
    for i in t:
        s = lookup_resource_ref_count(i)
        t = lookup_resource_app_query(i)
        x = lookup_resource_app_label(i)
        del s
        del t
        del x
    print(f"lookup time: {(datetime.now() - start).milliseconds}")

    start = datetime.now()
    for i in t:
        t = lookup_resource_app_query(i)
        x = lookup_resource_app_label(i)
        del t
        del x
    print(f"lookup-app time: {(datetime.now() - start).milliseconds}")


@profile
def test2():
    x = '''<?xml version="1.0" encoding="utf-8"?> 
    <ArrayOfResult xmlns="http://lookup.dbpedia.org/" 
    xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    </ArrayOfResult>'''
    for i in range(50):
        t = xmltodict.parse(x)
        del t
        gc.collect()

    for i in range(50):
        t = xmltodict.parse(x)
        del t
        gc.collect()

    return

test3()