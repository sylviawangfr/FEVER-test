from bert_serving.client import BertClient
import config



def get_phrase_embedding(phrases):
    bc = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=20000)
    return bc.encode(phrases)


if __name__ == '__main__':
    # get_topk_similar_phrases('Neil Armstrong', ['space', 'rocket', 'spacecraft', 'Birthday', 'astronaut',  'resident', 'top rank', 'text book', 'Country', 'student',
    #                                    'artist group', 'city'], top_k=10)
    p1 = ['Neil Armstrong', 'moon buggy', 'human', 'rocket']
    p2 = ['spacecraft', 'Birthday', 'game', 'fire', 'man']


