import spotlight
import json
import config
from datetime import datetime
import log_util


CONFIDENCE = 0.4
log = log_util.get_logger('dbpedia_spotlight')


def entity_link(sentence):
    start = datetime.now()
    log.debug("spotlight sent: " + sentence)
    try:
        annotations = spotlight.annotate(config.DBPEDIA_SPOTLIGHT_URL, sentence,
                                     confidence=CONFIDENCE,
                                     support=20)
    except Exception as err:
        log.error(err)
        return []

    # pretty_data = json.dumps(annotations, indent=4)
    # log.debug(pretty_data)
    entity_list = []
    for item in annotations:
        ent = dict()
        ent['URI'] = item['URI']
        ent['surfaceForm'] = item['surfaceForm']
        entity_list.append(ent)
    log.debug(f"spotlight time: {(datetime.now() - start).seconds}")
    return entity_list




if __name__ == "__main__":
    # entity_link("President Obama on Monday will call for a new minimum tax rate for individuals making more "
    #             "than $1 million a year to ensure that they pay at least the same percentage of their earnings "
    #             "as other taxpayers, according to administration officials.")
    # text1 = "Roman Atwood is a content creator."
    text1 = "Magic Johnson did not play for the Lakers."
    # text1 = 'Don Bradman retired from soccer.'
    entity_link(text1)

