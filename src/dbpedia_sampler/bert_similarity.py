from bert_serving.client import BertClient
import config
import numpy as np
import sklearn.metrics.pairwise as pw


def get_phrase_embedding(phrases):
    bc = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=20000)
    return bc.encode(phrases)

def get_most_close_pair(phrases1_l, phrases2_l, top_k=5):
    embedding1 = get_phrase_embedding(phrases1_l)
    embedding2 = get_phrase_embedding(phrases2_l)
    out = pw.cosine_similarity(embedding1, embedding2).flatten()
    topk_idx = np.argsort(out)[::-1][:top_k]
    len2 = len(phrases2_l)
    topk_k_pair = [(item//len2, item%len2, out[item]) for item in topk_idx]
    return topk_k_pair


def get_topk_similar_phrases(keyword_phrase, phrases_l, top_k=2):
    keyword_vec = get_phrase_embedding([keyword_phrase])[0]
    phrases_vecs = get_phrase_embedding(phrases_l)
    score = np.sum(keyword_vec * phrases_vecs, axis=1) / np.linalg.norm(phrases_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:top_k]
    result = []
    for idx in topk_idx:
        record = dict()
        record['idx'] = idx
        record['keywords'] = phrases_l[idx]
        record['score'] = score[idx]
        result.append(record)
        print('>%s\t%s' % (score[idx], phrases_l[idx]))
    return result


if __name__ == '__main__':
    # get_topk_similar_phrases('Neil Armstrong', ['space', 'rocket', 'spacecraft', 'Birthday', 'astronaut',  'resident', 'top rank', 'text book', 'Country', 'student',
    #                                    'artist group', 'city'], top_k=10)
    p1 = ['Neil Armstrong', 'moon buggy', 'human', 'rocket']
    p2 = ['spacecraft', 'Birthday', 'game', 'fire', 'man']
    get_most_close_pair(p1, p2)

