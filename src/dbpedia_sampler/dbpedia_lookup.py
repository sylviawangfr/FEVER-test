import difflib
from datetime import datetime

import urllib.request
from urllib.parse import quote, quote_plus
import xmltodict

import config
import log_util
from dbpedia_sampler.uri_util import uri_short_extract
from utils.file_loader import read_json_rows
from utils.tokenizer_simple import split_claim_regex
from memory_profiler import profile
import time
import gc


log = log_util.get_logger('lookup_resource')

# @profile
def lookup_resource(text_phrase):
    lookup_rec = combine_lookup(text_phrase)
    if len(lookup_rec) < 1:
        return []
    else:
        unwrapped_rec = [unwrap_record(i) for i in lookup_rec]
        return unwrapped_rec

def unwrap_record(lookup_rec):
    record = dict()
    record['Label'] = lookup_rec['Label']
    record['URI'] = lookup_rec['URI']
    record['exact_match'] = lookup_rec['exact_match']
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


def lookup_resource_no_filter(text_phrase):
    lookup_matches_ref = lookup_resource_ref_count(text_phrase)
    lookup_app_matches_label = lookup_resource_app_label(text_phrase)
    lookup_app_matches_query = lookup_resource_app_query(text_phrase)
    merged = []
    merge_resources(merged, lookup_matches_ref)
    merge_resources(merged, lookup_app_matches_label)
    merge_resources(merged, lookup_app_matches_query)
    keyword_matching_score = [difflib.SequenceMatcher(None, text_phrase,
                                                      i['Label'] if i['Label'] is not None else '').ratio()
                              for i in merged]
    sorted_matching_index = sorted(range(len(keyword_matching_score)), key=lambda k: keyword_matching_score[k],
                                   reverse=True)
    result = [merged[i] for i in sorted_matching_index]
    return result[:20]


def merge_resources(merged_l, to_merge):
    for i in to_merge:
        if len(list(filter(lambda x: (x['URI'] == i['URI']), merged_l))) < 1:
            merged_l.append(i)


def score_bewteen_phrases(phrase1, phrase2):
    return difflib.SequenceMatcher(None, phrase1 if phrase1 is not None else '',
                                   phrase2 if phrase2 is not None else '').ratio()

def get_keyword_matching_ratio_top(text_phrase, lookup_records, threshold=0.6):
    result = []
    keyword_matching_score = []
    try:
        keyword_matching_score = [difflib.SequenceMatcher(None, text_phrase.lower(),
                                                          i['Label'].lower() if i['Label'] is not None else '').ratio()
                                  for i in lookup_records]
    except Exception as err:
        log.error(err)
        print(text_phrase)
    if keyword_matching_score is None or len(keyword_matching_score) < 1:
        return result

    sorted_matching_index = sorted(range(len(keyword_matching_score)), key=lambda k: keyword_matching_score[k],
                                   reverse=True)
    # for i in sorted_matching_index:
    score = keyword_matching_score[sorted_matching_index[0]]
    #     partial_match = lookup_records[i]
    if score > threshold:
        result.append(lookup_records[sorted_matching_index[0]])

            # if partial_match['Label'].lower() in text_phrase.lower() or text_phrase.lower() in partial_match['Label'].lower():
            #     partial_match = lookup_records[i]
            #     result.append(partial_match)

    # if len(result) < 1 and keyword_matching_score[sorted_matching_index[0]] > threshold:
    #     result.append(lookup_records[sorted_matching_index[0]])
    return result


# @profile
def combine_lookup(text_phrase):
    lookup_matches_ref = lookup_resource_ref_count(text_phrase)
    # if exact_match is not None:
    #     log.debug(f"DBpedia lookup phrase: {text_phrase}, matching: {exact_match['URI']}")
    #     return exact_match, media_match
    # else:
    lookup_app_matches_label = lookup_resource_app_label(text_phrase)
    lookup_app_matches_query = lookup_resource_app_query(text_phrase)
    lookup_app_matches = []
    for i in lookup_matches_ref:
        if len(list(filter(lambda x: (x['Label'] == i['Label']), lookup_app_matches))) < 1:
            lookup_app_matches.append(i)
    for i in lookup_app_matches_query:
        if len(list(filter(lambda x: (x['Label'] == i['Label']), lookup_app_matches))) < 1:
            lookup_app_matches.append(i)
    for i in lookup_app_matches_label:
        if len(list(filter(lambda x: (x['Label'] == i['Label']), lookup_app_matches))) < 1:
            lookup_app_matches.append(i)

    lookup_app_matches.sort(key=lambda k: int(k['Refcount']), reverse=True)
    exact_match = get_exact_match(text_phrase, lookup_app_matches)
    media_match = get_media_subset_match(text_phrase, lookup_app_matches)
    if exact_match is not None:
        log.debug(f"DBpedia lookup-app phrase: {text_phrase}, matching: {exact_match['URI']}")
        exact_match['exact_match'] = True
        result = [exact_match]
        for i in media_match:
            if i['URI'] != exact_match['URI']:
                i['exact_match'] = False
                result.append(i)
        top_ref = lookup_app_matches[0]
        if len(list(filter(lambda x: (x['URI'] == top_ref['URI']), result))) < 1 \
                and score_bewteen_phrases(text_phrase, top_ref['Label']) > 0.3:
            top_ref['exact_match'] = False
            result.append(top_ref)
        # print(f"link_phrase: {text_phrase}, count links: {len(result)}")
        return result 

    top_match = []
    # token_count = len(text_clean.easy_tokenize(text_phrase))
    token_count = text_phrase.count(' ') + 1
    if token_count < 2 and len(lookup_matches_ref) > 0:         # use lookup refCount
        partial_matches = []
        for i in lookup_matches_ref:
            tmp_label = i['Label'].lower()
            text_phrase_lower = text_phrase.lower()
            if text_phrase_lower in tmp_label or tmp_label in text_phrase_lower:
                partial_matches.append(i)
        if len(partial_matches) > 0:
            top_match = get_keyword_matching_ratio_top(text_phrase, partial_matches, 0.6)
        else:
            first_match = lookup_matches_ref[0]
            tmp_match = get_keyword_matching_ratio_top(text_phrase, lookup_matches_ref, 0.6)
            if len(tmp_match) > 0:
                top_match = tmp_match
            else:
                top_match = get_keyword_matching_ratio_top(text_phrase, [first_match], 0.155)
        if len(top_match) > 0:
            log.debug(f"DBpedia lookup phrase: {text_phrase}, matching: {top_match}")

    if len(lookup_app_matches) > 0 and (token_count > 1 or len(top_match) < 1):        # use lookup-app Label+comments
        top_match = get_keyword_matching_ratio_top(text_phrase, lookup_app_matches, 0.6)
        top_ref = lookup_app_matches[0]
        if len(list(filter(lambda x: (x['URI'] == top_ref['URI']), top_match))) < 1 \
                and score_bewteen_phrases(text_phrase, top_ref['Label']) > 0.3:
            top_match.append(top_ref)

        log.debug(f"DBpedia lookup phrase: {text_phrase}, matching: {top_match}")

    if len(top_match) < 1:
        log.debug(f"failed to query DBpedia_lookup and lookup, matching score is too low: {text_phrase}")
        
    result = []
    if len(top_match) > 0:
        result.extend(top_match)
    for i in media_match:
        if len(list(filter(lambda x: (x['URI'] == i['URI']), result))) < 1:
            result.append(i)
    for t in result:
        t['exact_match'] = False
    # print(f"link_phrase: {text_phrase}, count links: {len(result)}")
    return result


def get_exact_match(text_phrase, lookup_records):
    for i in lookup_records:
        if i['Label'] is not None and text_phrase.lower() == i['Label'].lower():
            return i
    return None


def get_media_subset_match(text_phrase, lookup_records):
    media_match = []
    media = ['tv', 'film', 'book', 'novel', 'band', 'album', 'music', 'series', 'poem', 'song', 'advertisement', 'company',
             'episode', 'season', 'animator', 'actor', 'singer', 'writer', 'drama', 'character']
    for i in lookup_records:
        label = i['Label']
        if label is not None and text_phrase.lower() in label.lower() and any([j in label.lower() for j in media]):
            media_match.append(i)
    return media_match

def lookup_resource_app_query(text_phrase):
    url = config.DBPEDIA_LOOKUP_APP_URL_QUERY + quote(text_phrase)
    return lookup_resource_app(text_phrase, url)


def lookup_resource_app_label(text_phrase):
    url = config.DBPEDIA_LOOKUP_APP_URL_LABEL + quote(text_phrase)
    return lookup_resource_app(text_phrase, url)

# @profile
def lookup_resource_app(text_phrase, url):
    start = datetime.now()
    close_matches = []
    try:
        response = urllib.request.urlopen(url, timeout=5)
    except Exception as err:
        log.error(err)
        log.warning(f"{text_phrase}: {url}")
        return close_matches
    # response = requests.get(url, timeout=5) # cause memory leak
    if response.status == 200:
        xml = response.read().decode('utf-8')
        try:
            results1 = xmltodict.parse(xml)
        except Exception as err:
            log.error(err)
            log.warning(f"{text_phrase}: {url}")
            response.close()
            return close_matches
        if len(results1['ArrayOfResults']) <= 3:
            log.debug(f"lookup phrase: {text_phrase}, no matching found by lookup-app-query.")
        else:
            re = results1['ArrayOfResults']['Result']
            if isinstance(re, dict):
                close_matches.append(re)
            else:
                # for i in re:
                #     if 'Label' in i and i['Label'] is not None:
                #         tmp_label = i['Label'].lower()
                #         text_phrase_lower = text_phrase.lower()
                #         if tmp_label == text_phrase_lower:
                #             close_matches.append(i)
                # if len(close_matches) < 1:
                close_matches = re
    else:
        log.error(f"failed to query lookup-app, response code:{response.status_code}")
    response.close()
    log.debug(f"lookup-app time: {(datetime.now() - start).seconds}")
    return close_matches

# @profile
def lookup_resource_ref_count(text_phrase):
    start = datetime.now()
    if '%' in text_phrase:
        return []
    url = config.DBPEDIA_LOOKUP_URL + quote(text_phrase)
    # log.debug(f"lookup url: {url}")
    close_matches = []
    try:
        response = urllib.request.urlopen(url, timeout=5)
    except Exception as err:
        log.error(err)
        log.warning(f"{text_phrase}: {url}")
        return close_matches
    if response.status != 200:
        log.error(f"failed to query lookup, response code: {response.status_code}, phrase: {text_phrase})")
    else:
        try:
            xml = response.read().decode('utf-8')
            results = xmltodict.parse(xml)
        except Exception as err:
            log.error(err)
            log.warning(f"{text_phrase}: {url}")
            response.close()
            return close_matches

        if len(results['ArrayOfResult']) <= 3:
            log.debug(f"lookup phrase: {text_phrase}, no matching found by lookup ref.")
        else:
            re = results['ArrayOfResult']['Result']
            if isinstance(re, dict):
                close_matches.append(re)
            else:
                # for i in re:
                #     tmp_label = i['Label'].lower()
                #     text_phrase_lower = text_phrase.lower()
                #     if tmp_label == text_phrase_lower:
                #         close_matches.append(i)
                # if len(close_matches) < 1:
                close_matches = re
    response.close()
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


# @profile
STOP_WORDS = ['they', 'i', 'me', 'you', 'she', 'he', 'it', 'individual', 'individuals', 'year', 'years', 'day', 'night',
               'we', 'who', 'where', 'what', 'days', 'him', 'her', 'here', 'there', 'a', 'for',
              'which', 'when', 'whom', 'the', 'history', 'morning', 'afternoon', 'evening', 'night', 'first', 'second',
              'third']
def test():
    j = read_json_rows(config.FEVER_DEV_JSONL)[300:400]
    for i in j:
        claim = i['claim']
        ent = split_claim_regex(claim)[0]
        if ent and ent.lower() not in STOP_WORDS:
            top1, media = lookup_resource(ent)
            print(claim)
            print(ent)
            print(top1)
            print(media)
            print("------------------")



if __name__ == "__main__":
    # for i in range(1):
    #     test()
    #     gc.collect()
    # test()
    lookup_resource("A monk")
    lookup_resource("Winter's Tale")

    # lookup_resource_no_filter('Tool')
    lookup_resource('Western Conference Southwest Division')
    lookup_resource("the league 's Western Conference Southwest Division")
    lookup_resource('NBA')
    lookup_resource('a member club')
    lookup_resource('The Pelicans')
    lookup_resource('the National Basketball Association')
    # lookup_resource('Bret Easton Ellis')
    # lookup_resource('Robert Palmer')
    # lookup_resource('American')
    # lookup_resource('UK')
    lookup_resource('Western Conference Southwest Division')
    # lookup_resource('Indiana')
    # lookup_resource('film')
    # lookup_resource('music')

