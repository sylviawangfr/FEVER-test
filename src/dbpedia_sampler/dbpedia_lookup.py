import requests
import xmltodict
import json
import os
import config
from dbpedia_sampler.dbpedia_virtuoso import keyword_extract


def lookup_resource(text_phrase):
    url = config.DBPEDIA_LOOKUP_URL + text_phrase
    print(url)
    response = requests.get(url, timeout=2)
    if response.status_code is 200:
        results = xmltodict.parse(response.text)
        if len(results['ArrayOfResult']) <= 3:
            return -3
        else:
            top_match = results['ArrayOfResult']['Result'][0]
            record = dict()
            record['Label'] = top_match['Label']
            record['URI'] = top_match['URI']
            catgr = []
            cls_l = top_match['Classes']['Class']
            for c in cls_l:
                if 'http://dbpedia.org/ontology/' in c['URI']:
                    catgr.append(c['URI'])
            record['Classes'] = catgr

        print(json.dumps(record, indent=4))
        return record
        # print(json.dumps(results, indent=4))
    else:
        print('failed to connect DBpedia lookup')
        return -1


def to_triples(record_json):
    subject = record_json['URI']
    relation = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
    triples = []
    for i in record_json['Classes']:
        tri = dict()
        tri['subject'] = subject
        tri['relation'] = relation
        tri['object'] = i
        tri['keywords'] = [keyword_extract(i)]
        triples.append(tri)
    print(json.dumps(triples, indent=4))
    return triples


if __name__ == "__main__":
    to_triples(lookup_resource('Berlin'))