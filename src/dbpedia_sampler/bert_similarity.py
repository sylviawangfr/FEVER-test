from bert_serving.client import BertClient
import config
import numpy as np


bc = BertClient(config.BERT_SERVICE_URL)

def get_phrase_embedding(phrases):
    return bc.encode(phrases)


def get_topk_similar_phrases(keyword_phrase, phrases_l):
    keyword_vec = bc.encode([keyword_phrase])[0]
    phrases_vecs = bc.encode(phrases_l)
    score = np.sum(keyword_vec * phrases_l, axis=1) / np.linalg.norm(phrases_vecs, axis=1)
    top2_idx = np.argsort(score)[::-1][:2]
    result = []
    for idx in top2_idx:
        record = dict()
        record['idx'] = idx
        record['keyword'] = phrases_l[idx]
        record['score'] = score[idx]
        result.append(record)
        print('>%s\t%s' % (score[idx], phrases_l[idx]))
    return result