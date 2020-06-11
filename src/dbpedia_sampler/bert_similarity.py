from bert_serving.client import BertClient
import config
import difflib



def get_phrase_embedding(phrases):
    try:
        bc = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=100000)
        return bc.encode(phrases)
    except Exception as err:
        print("failed to get embedding for phrases...")
        print(err)
        return []


if __name__ == '__main__':
    p1 = ['Neil Armstrong', 'moon buggy', 'human', 'rocket']
    p2 = ['spacecraft', 'Birthday', 'game', 'fire', 'man']
    keyword_matching = [difflib.SequenceMatcher(None, 'Neil Armstrong', i['Label']).ratio() for i in p2]
    sorted_matching_index = sorted(range(len(keyword_matching)), key=lambda k: keyword_matching[k], reverse=True)


