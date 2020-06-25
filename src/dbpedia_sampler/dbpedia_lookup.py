import requests
import xmltodict
import json
import config
from dbpedia_sampler.dbpedia_virtuoso import uri_short_extract
import difflib
from datetime import datetime
import log_util


log = log_util.get_logger('lookup_resource')


def lookup_resource(text_phrase):
    start = datetime.now()
    url = config.DBPEDIA_LOOKUP_URL + text_phrase
    log.debug(f"lookup url: {url}")
    response = requests.get(url, timeout=5)
    if response.status_code is 200:
        results = xmltodict.parse(response.text)
        if len(results['ArrayOfResult']) <= 3:
            return -3
        else:
            re = results['ArrayOfResult']['Result']
            if isinstance(re, dict):
                top_match = re
            else:
                # keyword_matching = [difflib.SequenceMatcher(None, text_phrase, i['Label']).ratio() for i in re]
                # sorted_matching_index = sorted(range(len(keyword_matching)), key=lambda k: keyword_matching[k], reverse=True)
                # top_match = re[sorted_matching_index[0]]
                top_match = re[0]
                close_matches = []
                for i in re:
                    tmp_label = i['Label']
                    if tmp_label == text_phrase or text_phrase in tmp_label or tmp_label in text_phrase:
                        close_matches.append(i)
                if len(close_matches) > 0:
                    keyword_matching = [difflib.SequenceMatcher(None, text_phrase, i['Label']).ratio() for i in close_matches]
                    sorted_matching_index = sorted(range(len(keyword_matching)), key=lambda k: keyword_matching[k], reverse=True)
                    top_match = close_matches[sorted_matching_index[0]]
            double_check_matching_score = difflib.SequenceMatcher(None, text_phrase, top_match['Label']).ratio()
            if double_check_matching_score < 0.15:
                log.warning(f"failed to connect DBpedia lookup, matching score is too low: {text_phrase} VS {top_match['Label']}")
                return -1

            record = dict()
            record['Label'] = top_match['Label']
            record['URI'] = top_match['URI']
            catgr = []
            if not top_match['Classes'] is None:
                cl = top_match['Classes']['Class']
                cls_l = [cl] if isinstance(cl, dict) else cl
            else:
                cls_l = []
            for c in cls_l:
                if ('http://dbpedia.org/ontology/' in c['URI'] \
                        or 'http://schema.org/' in c['URI']) \
                        and not '/Agent' in c['URI']:
                    catgr.append(c['URI'])                           # or 'http://www.w3.org/2002/07/owl' in c['URI'] \
            record['Classes'] = catgr

        # log.debug(json.dumps(record, indent=4))
        log.debug(f"lookup time: {(datetime.now() - start).seconds}")
        return record
        # print(json.dumps(results, indent=4))
    else:
        log.warning('failed to connect DBpedia lookup')
        log.debug(f"lookup time: {(datetime.now() - start).seconds}")
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
        tri['keywords'] = [uri_short_extract(i)]
        triples.append(tri)
    # print(json.dumps(triples, indent=4))
    return triples


if __name__ == "__main__":
    lookup_resource('music')
    lookup_resource('film')
    log.warning("test debug")
    log.info("test info")

