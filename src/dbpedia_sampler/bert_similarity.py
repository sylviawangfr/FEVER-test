from bert_serving.client import BertClient
import config
import difflib
from datetime import datetime
import log_util

log = log_util.get_logger('bert_similarity')


def get_phrase_embedding(phrases):
    try:
        start = datetime.now()
        log.debug(f"embedding: {phrases}")
        if phrases is None or len(phrases) < 1:
            return []

        bc = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=60000)
        re = bc.encode(phrases)
        log.debug(f"embedding time: {(datetime.now() - start).seconds}")
        return re
        # return []
    except Exception as err:
        log.warning(f"failed to get embedding for phrases...{phrases}")
        log.error(err)
        return []


if __name__ == '__main__':
    p1 = ['Neil Armstrong', 'moon buggy', 'human', 'rocket']
    p2 = ['spacecraft', 'Birthday', 'game', 'fire', 'man']
    keyword_matching = [difflib.SequenceMatcher(None, 'Neil Armstrong', i['Label']).ratio() for i in p2]
    sorted_matching_index = sorted(range(len(keyword_matching)), key=lambda k: keyword_matching[k], reverse=True)


