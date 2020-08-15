import requests
import xmltodict
import config
from dbpedia_sampler.dbpedia_virtuoso import uri_short_extract
import difflib
from datetime import datetime
import log_util
from memory_profiler import profile

log = log_util.get_logger('lookup_resource')


def lookup_resource(text_phrase):
    lookup_rec = combine_lookup(text_phrase)
    if len(lookup_rec) < 1:
        return []
    else:
        record = dict()
        record['Label'] = lookup_rec['Label']
        record['URI'] = lookup_rec['URI']
        catgr = []
        if not lookup_rec['Classes'] is None:
            cl = lookup_rec['Classes']['Class']
            cls_l = [cl] if isinstance(cl, dict) else cl
        else:
            cls_l = []
        for c in cls_l:
            if ('http://dbpedia.org/ontology/' in c['URI'] \
                or 'http://schema.org/' in c['URI']) \
                    and not '/Agent' in c['URI']:
                catgr.append(c['URI'])  # or 'http://www.w3.org/2002/07/owl' in c['URI'] \
        record['Classes'] = catgr
        return record


def get_keyword_matching_ratio_top(text_phrase, lookup_records, threshold=0.6):
    top_match = []
    try:
        keyword_matching_score = [difflib.SequenceMatcher(None, text_phrase,
                                                          i['Label'] if i['Label'] is not None else '').ratio()
                                  for i in lookup_records]
    except Exception as err:
        log.error(err)
        print(text_phrase)
    if keyword_matching_score is None:
        return top_match

    sorted_matching_index = sorted(range(len(keyword_matching_score)), key=lambda k: keyword_matching_score[k],
                                   reverse=True)
    top_score = keyword_matching_score[sorted_matching_index[0]]
    if top_score > threshold:
        top_match = lookup_records[sorted_matching_index[0]]
    return top_match


def combine_lookup(text_phrase):
    lookup_matches = lookup_resource_ref_count(text_phrase)
    exact_match = has_exact_match(text_phrase, lookup_matches)
    if exact_match is not None:
        log.debug(f"DBpedia lookup phrase: {text_phrase}, matching: {exact_match['URI']}")
        return exact_match
    else:
        lookup_app_matches_label = lookup_resource_app_label(text_phrase)
        lookup_app_matches_query = lookup_resource_app_query(text_phrase)
        lookup_app_matches = lookup_app_matches_label
        for i in lookup_app_matches_query:
            if len(list(filter(lambda x: (x['Label'] == i['Label']), lookup_app_matches))) < 1:
                lookup_app_matches.append(i)

        exact_match = has_exact_match(text_phrase, lookup_app_matches)
        if exact_match is not None:
            log.debug(f"DBpedia lookup-app phrase: {text_phrase}, matching: {exact_match['URI']}")
            return exact_match

    top_match = []
    # token_count = len(text_clean.easy_tokenize(text_phrase))
    token_count = text_phrase.count(' ') + 1
    if token_count < 2 and len(lookup_matches) > 0:         # use lookup refCount
        partial_matches = []
        for i in lookup_matches:
            tmp_label = i['Label'].lower()
            text_phrase_lower = text_phrase.lower()
            if text_phrase_lower in tmp_label or tmp_label in text_phrase_lower:
                partial_matches.append(i)
        if len(partial_matches) > 0:
            top_match = get_keyword_matching_ratio_top(text_phrase, partial_matches, 0.6)
        else:
            first_match = lookup_matches[0]
            tmp_match = get_keyword_matching_ratio_top(text_phrase, lookup_matches, 0.6)
            if len(tmp_match) > 0:
                top_match = tmp_match
            else:
                top_match = get_keyword_matching_ratio_top(text_phrase, [first_match], 0.155)
        if len(top_match) > 0:
            log.debug(f"DBpedia lookup phrase: {text_phrase}, matching: {top_match['URI']}")

    if len(lookup_app_matches) > 0 and (token_count > 1 or len(top_match) < 1):        # use lookup-app Label+comments
        top_match = get_keyword_matching_ratio_top(text_phrase, lookup_app_matches, 0.6)
        if len(top_match) > 0:
            log.debug(f"DBpedia lookup-app phrase: {text_phrase}, matching: {top_match['URI']}")

    if len(top_match) < 1:
        log.debug(f"failed to query DBpedia_lookup and lookup-app, matching score is too low: {text_phrase}")

    return top_match


def has_exact_match(text_phrase, lookup_records):
    for i in lookup_records:
        if i['Label'] is not None and text_phrase.lower() == i['Label'].lower():
            return i
    return None


def lookup_resource_app_query(text_phrase):
    url = config.DBPEDIA_LOOKUP_APP_URL_QUERY + text_phrase
    return lookup_resource_app(text_phrase, url)


def lookup_resource_app_label(text_phrase):
    url = config.DBPEDIA_LOOKUP_APP_URL_LABEL + text_phrase
    return lookup_resource_app(text_phrase, url)


def lookup_resource_app(text_phrase, url):
    start = datetime.now()
    close_matches = []
    response = requests.get(url, timeout=5)
    if response.status_code is 200:
        results1 = xmltodict.parse(response.text)
        if len(results1['ArrayOfResults']) <= 3:
            log.debug(f"lookup phrase: {text_phrase}, no matching found by lookup-app-query.")
            return close_matches
        else:
            re = results1['ArrayOfResults']['Result']
            if isinstance(re, dict):
                close_matches.append(re)
            else:
                for i in re:
                    if 'Label' in i and i['Label'] is not None:
                        tmp_label = i['Label'].lower()
                        text_phrase_lower = text_phrase.lower()
                        if tmp_label == text_phrase_lower:
                            close_matches.append(i)
                            break
                if len(close_matches) < 1:
                    close_matches = re
    else:
        log.error(f"failed to query lookup-app, response code:{response.status_code}")
    log.debug(f"lookup-app time: {(datetime.now() - start).seconds}")
    return close_matches


def lookup_resource_ref_count(text_phrase):
    start = datetime.now()
    if '%' in text_phrase:
        return []
    url = config.DBPEDIA_LOOKUP_URL + text_phrase
    # log.debug(f"lookup url: {url}")
    response = requests.get(url, timeout=5)
    close_matches = []

    if response.status_code is not 200:
        log.error(f"failed to query lookup, response code: {response.status_code}, phrase: {text_phrase})")
        return []
    else:
        results = xmltodict.parse(response.text)
        if len(results['ArrayOfResult']) <= 3:
            log.debug(f"lookup phrase: {text_phrase}, no matching found by lookup ref.")
            return []
        else:
            re = results['ArrayOfResult']['Result']
            if isinstance(re, dict):
                close_matches.append(re)
            else:
                for i in re:
                    tmp_label = i['Label'].lower()
                    text_phrase_lower = text_phrase.lower()
                    if tmp_label == text_phrase_lower:
                        close_matches.append(i)
                        break
                if len(close_matches) < 1:
                    close_matches = re
    log.debug(f"lookup time: {(datetime.now() - start).seconds}")
    return close_matches


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
    lookup_resource('Howard Eugene Johnson')
    lookup_resource('cultists')
    lookup_resource('Italian')
    lookup_resource('Even')
    lookup_resource('Giada Pamela De Laurentiis')
    lookup_resource('American')
    lookup_resource('UK')
    lookup_resource('Bloomington')
    lookup_resource('Indiana')
    lookup_resource('film')
    lookup_resource('music')

