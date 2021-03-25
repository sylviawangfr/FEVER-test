from elasticsearch import Elasticsearch as es
from elasticsearch_dsl import Search, Q

from utils.file_loader import *
from utils.tokenizer_simple import *


client = es([{'host': config.ELASTIC_HOST, 'port': config.ELASTIC_PORT,
              'timeout': 60, 'max_retries': 5, 'retry_on_timeout': True}])


def remove_the_a(ph):
    if (not is_capitalized(ph)) and \
            (ph.lower().startswith('the ')
             or ph.lower().startswith("a ")
             or ph.lower().startswith("an ")):
        ph = ph.split(' ', 1)[1]
    return ph


# ES match_phrase on entities
def search_doc(phrases):
    try:
        search = Search(using=client, index=config.WIKIPAGE_INDEX)
        must = []
        should = []
        for ph in phrases:
            ph = remove_the_a(ph)
            # should.append({'multi_match': {'query': ph, "type": "most_fields",
            #                                'fields': ['id^2', 'lines'], 'analyzer': 'underscore_analyzer'}})
            must.append({'multi_match': {'query': ph, "type": "phrase",
                                         'fields': ['id^2', 'lines'], 'slop': 3, 'analyzer': 'underscore_analyzer'}})

        search = search.query(Q('bool', must=must, should=should)). \
                 highlight('lines', number_of_fragments=0). \
                 sort({'_score': {"order": "desc"}}). \
                 source(include=['id'])[0:10]

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
    except:
        return []

# entity, keywords
def search_doc_dbpedia_context(context_dict):
    try:
        search = Search(using=client, index=config.WIKIPAGE_INDEX)
        must = []
        should = []

        # must.append({'match_phrase': {'lines': ph}})
        for i in context_dict['keywords']:
            should.append({'multi_match': {'query': i, "type": "phrase", 'fields': ['id^2', 'lines'],
                                           'slop': 3, 'analyzer': 'underscore_analyzer'}})
        must.append({'multi_match': {'query': context_dict['entity'], "type": "phrase",
                                     'fields': ['id^2', 'lines'], 'slop': 3, 'analyzer': 'underscore_analyzer'}})

        search = search.query(Q('bool', must=must, should=should)). \
                 highlight('lines', number_of_fragments=0). \
                 sort({'_score': {"order": "desc"}}). \
                 source(include=['id'])[0:5]

        response = search.execute()
        r_list = []
        phrases = context_dict['keywords']
        phrases.append(context_dict['entity'])
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
    except:
        return []


def search_doc_id_and_keywords(possible_id, keywords):
    search = Search(using=client, index=config.WIKIPAGE_INDEX)
    must = []
    should = []
    must.append(
        {'match_phrase': {'id': {'query': possible_id, 'analyzer': 'underscore_analyzer', 'boost': 2}}})
    # should.append({'match_phrase': {'lines': {'query': possible_id.replace("_", " "), 'analyzer': 'underscore_analyzer'}}})
    if len(keywords) == 2:
        relation = keywords[0]
        should.append({'match': {'lines': {'query': relation, 'analyzer': 'wikipage_analyzer'}}})
    if len(keywords) > 0:
        obj = keywords[-1]
        should.append({'match_phrase': {'lines': {'query': obj, 'slop': 3, 'analyzer': 'wikipage_analyzer'}}})
    search = search.query(Q('bool', must=must, should=should)). \
                 highlight('lines', number_of_fragments=0, fragment_size=150). \
                 sort({'_score': {"order": "desc"}}). \
                 source(include=['id'])[0:5]

    response = search.execute()
    r_list = []
    sentences_list = []
    phrases = [possible_id]
    phrases.extend(keywords)
    for hit in response['hits']['hits']:
        score = hit['_score']
        id = hit['_source']['id']
        if 'highlight' in hit:
            lines = hit['highlight']['lines'][0]
            lines = lines.replace("</em> <em>", " ")
            lines_json_l = json.loads(normalize(lines))
            match_count = [i['sentences'].count("<em>") for i in lines_json_l]
            sorted_matching_index = sorted(range(len(match_count)), key=lambda k: match_count[k],
                                           reverse=True)
            highest_match_count = -1
            for idx in sorted_matching_index:
                if match_count[idx] > 0 and match_count[idx] >= highest_match_count:
                    h_links_s = set()
                    line_num = lines_json_l[idx]['line_num']
                    sentence = lines_json_l[idx]['sentences']
                    h_links = lines_json_l[idx]['h_links']
                    sentence = sentence.replace("</em>", "").replace("<em>", "")
                    highest_match_count = match_count[idx]
                    for h in h_links:
                        h = h.replace("<em>", "").replace("</em>", "")
                        h_links_s.add(h)
                    sentences_list.append(
                            {'sid': f'{possible_id}<SENT_LINE>{line_num}', 'text': sentence, 'h_links': list(h_links_s)})
                else:
                    break
        for s in sentences_list:
            doc_dic = {'score': score, 'phrases': phrases, 'doc_id': id, 'sid': s['sid'], 'text': s['text'], 'h_links': s['h_links']}
            r_list.append(doc_dic)
    return r_list


def search_doc_id_and_keywords_in_sentences(possible_id, subject, keywords):
    search = Search(using=client, index=config.FEVER_SEN_INDEX)
    must = []
    should = []
    must.append(
        {'term': {'doc_id_keyword': possible_id}})
    must.append({'multi_match': {'query': subject, 'type': 'phrase', 'slop': 3,
                                 'fields': ['doc_id', 'text'], 'analyzer': 'underscore_analyzer'}})
    if len(keywords) == 2:
        relation = keywords[0]
        should.append({'match': {'text': {'query': relation, 'analyzer': 'wikipage_analyzer'}}})
    if len(keywords) > 0:
        obj = keywords[-1]
        must.append({'multi_match': {'query': obj,
                                     'fields': ['text'], 'analyzer': 'underscore_analyzer'}})

    search = search.query(Q('bool', must=must, should=should)). \
                 highlight('text', number_of_fragments=0, fragment_size=150). \
                 sort({'_score': {"order": "desc"}}). \
                 source(include=['doc_id', 'sid', 'text', 'h_links'])[0:5]

    response = search.execute()
    r_list = []
    phrases = [possible_id]
    phrases.extend(keywords)
    top_n = 3
    for hit in response['hits']['hits']:
        if top_n < 1:
            break
        score = hit['_score']
        doc_id = hit['_source']['doc_id']
        sid = hit['_source']['sid']
        text = hit['_source']['text']
        h_links = list(set(json.loads(normalize(hit['_source']['h_links']))))
        doc_dic = {'doc_id': doc_id, 'score': score, 'phrases': phrases, 'sid': sid, 'text': text, 'h_links': h_links}
        r_list.append(doc_dic)
        top_n -= 1
    return r_list


def search_docid_subject_object_in_sentences(possible_id, subj, rel, obj):
    search = Search(using=client, index=config.FEVER_SEN_INDEX)
    must = []
    should = []
    must.append(
        {'term': {'doc_id_keyword': possible_id}})
    must.append({'multi_match': {'query': subj, 'type': 'phrase', 'slop': 3,
                                 'fields': ['doc_id', 'text'], 'analyzer': 'underscore_analyzer'}})
    must.append({'multi_match': {'query': obj, 'type': 'phrase', 'slop': 3,
                                 'fields': ['doc_id', 'text'], 'analyzer': 'underscore_analyzer'}})
    should.append({'match': {'text': {'query': rel, 'analyzer': 'wikipage_analyzer'}}})

    search = search.query(Q('bool', must=must, should=should)). \
                 highlight('text', number_of_fragments=0, fragment_size=150). \
                 sort({'_score': {"order": "desc"}}). \
                 source(include=['doc_id', 'sid', 'text', 'h_links'])[0:5]

    response = search.execute()
    r_list = []
    phrases = [possible_id, subj, rel, obj]
    top_n = 3
    for hit in response['hits']['hits']:
        if top_n < 1:
            break
        score = hit['_score']
        doc_id = hit['_source']['doc_id']
        sid = hit['_source']['sid']
        text = hit['_source']['text']
        h_links = list(set(json.loads(normalize(hit['_source']['h_links']))))
        doc_dic = {'doc_id': doc_id, 'score': score, 'phrases': phrases, 'sid': sid, 'text': text, 'h_links': h_links}
        r_list.append(doc_dic)
        top_n -= 1
    return r_list


def search_doc_id(possible_id):
    try:
        body = {
            "query": {
                "match_phrase": {
                    "id": {
                        "query": possible_id,
                        "analyzer": "underscore_analyzer"
                    }
                }},
            "sort": {"_score": {"order": "desc"}},
            "_source": ["id"],
            "size": 10
        }
        response = client.search(index=config.WIKIPAGE_INDEX, body=body)
        r_list = []
        for hit in response['hits']['hits']:
            score = hit['_score']
            id = hit['_source']['id']
            doc_dic = {'score': score, 'phrases': [possible_id], 'id': id, 'lines': ""}
            r_list.append(doc_dic)
        return r_list
    except Exception as e:
        print(e)
        return []


def test_search_id(text):
    try:
        search = Search(using=client, index=config.WIKIPAGE_INDEX)
        must = []
        if text.lower().startswith('the ') or text.lower().startswith("a ") or text.lower().startswith("an "):
            text = text.split(' ', 1)[1]
        # must.append({'match_phrase': {'lines': ph}})
        # should.append({'match_phrase': {'id': {'query': ph, 'boost': 2}}})
        must.append({'multi_match': {'query': text, "type": "phrase", 'fields': ['id^2']}})

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
            doc_dic = {'score': score, 'phrases': text, 'id': id, 'lines': lines}
            r_list.append(doc_dic)

        r_list.sort(key=lambda x: x.get('score'), reverse=True)
        return r_list
    except:
        return []