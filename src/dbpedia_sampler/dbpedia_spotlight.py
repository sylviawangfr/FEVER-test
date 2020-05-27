import spotlight
import json
import config

CONFIDENCE = 0.4

def entity_link(sentence):
    annotations = spotlight.annotate(config.DBPEDIA_SPOTLIGHT_URL, sentence,
                                     confidence=CONFIDENCE,
                                     support=20)
    pretty_data = json.dumps(annotations, indent=4)
    print(pretty_data)
    entity_list = []
    for item in annotations:
        ent = dict()
        ent['URI'] = item['URI']
        ent['surfaceForm'] = item['surfaceForm']
        entity_list.append(ent)
    print(entity_list)
    return entity_list




if __name__ == "__main__":
    # entity_link("President Obama on Monday will call for a new minimum tax rate for individuals making more "
    #             "than $1 million a year to ensure that they pay at least the same percentage of their earnings "
    #             "as other taxpayers, according to administration officials.")
    text1 = "Autonomous cars shift insurance liability toward manufacturers"
    # text1 = "Magic Johnson did not play for the Lakers."
    # text1 = 'Don Bradman retired from soccer.'
    entity_link(text1)

