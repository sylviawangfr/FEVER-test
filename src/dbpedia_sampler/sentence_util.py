import log_util
from utils import c_scorer, text_clean, file_loader
from utils.tokenizer_simple import *
import config
from memory_profiler import profile

STOP_WORDS = ['the', 'they', 'i', 'me', 'you', 'she', 'he', 'it', 'individual', 'individuals', 'year', 'years', 'day', 'night',
               'we', 'who', 'where', 'what', 'days', 'him', 'her', 'here', 'there', 'a', 'for', 'anything', 'everything',
              'which', 'when', 'whom', 'the', 'history', 'morning', 'afternoon', 'evening', 'night', 'first', 'second',
              'third', 'life', 'all', 'part']

DO_NOT_LINK = ['American', 'German', 'Spanish']


log = log_util.get_logger('dbpedia_triple_linker')


# @profile
def get_phrases(sentence, doc_title=''):
    log.debug(sentence)
    if doc_title != '' and c_scorer.SENT_DOC_TITLE in sentence and sentence.startswith(doc_title):
        title_and_sen = sentence.split(c_scorer.SENT_DOC_TITLE, 1)
        sent = title_and_sen[1]
    else:
        sent = sentence
    entity_and_capitalized, nouns = get_ents_and_phrases(sent)
    # if not doc_title == '':
    #     entity_and_capitalized = list(set(entity_and_capitalized) | {doc_title})
    return entity_and_capitalized, nouns


def remove_the_a(ph):
    if (not is_capitalized(ph)) and \
            (ph.lower().startswith('the ')
             or ph.lower().startswith("a ")
             or ph.lower().startswith("an ")):
        ph = ph.split(' ', 1)[1]
    return ph


def get_ents_and_phrases(sentence):
    doc_noun = nlp_eng_spacy(sentence)
    noun_tokens = []
    for token in doc_noun:
        if token.pos_.lower() in ['propn', 'noun'] or token.dep_ == 'dobj':
            noun_tokens.append(token.text)
    nouns_chunks = [remove_the_a(chunk.text) for chunk in doc_noun.noun_chunks if chunk.text.lower() not in STOP_WORDS]
    ents = [remove_the_a(ent.text) for ent in doc_noun.ents if ent.text.lower() not in STOP_WORDS]
    capitalized_phrased = list(set(split_claim_regex(sentence)))

    nouns = list(set(nouns_chunks))
    entity_and_capitalized = []
    for i in capitalized_phrased:
        if text_clean.is_date_or_number(i) and i not in nouns:
            nouns.append(i)
            continue
        else:
            if i.lower() not in STOP_WORDS:
                entity_and_capitalized.append(i)
            if sentence.startswith(i) and (i.startswith('The ') or i.startswith('A ')):
                if i.startswith('The '):
                    remove_the = i.replace('The ', '', 1)
                if i.startswith('A '):
                    remove_the = i.replace('A ', '', 1)
                if i.startswith('An '):
                    remove_the = i.replace('An ', '', 1)
                if remove_the not in entity_and_capitalized and remove_the.lower() not in STOP_WORDS:
                    entity_and_capitalized.append(remove_the)
            continue

    for i in nouns_chunks:
        if text_clean.is_date_or_number(i):
            continue
        if len(list(filter(lambda x: (i in x), entity_and_capitalized))) > 0 and i in nouns:
            for x in entity_and_capitalized:
                if (' or ' in x) or (' and ' in x) or (' of ' in x):
                    splits = split_combinations(x)
                    if i in splits and i not in entity_and_capitalized:
                        entity_and_capitalized.append(i)
                        break
                if (x.startswith('In ') or x.startswith('At ') or x.startswith('From ') or x.startswith('After ')) \
                        and i in x \
                        and i not in entity_and_capitalized:
                    entity_and_capitalized.append(i)
                    break
            nouns.remove(i)
            continue
        if len(list(filter(lambda x: (x in i and (' ' in x)), entity_and_capitalized))) > 0 \
                and i in nouns:
            nouns.remove(i)
            continue
        if len(list(filter(lambda x: (x in i and sentence.startswith(x) and ' ' not in x), entity_and_capitalized))) > 0 \
                and i not in entity_and_capitalized \
                and i in nouns:
            nouns.remove(i)
            entity_and_capitalized.append(i)
            continue
        if (not is_capitalized(i)) \
                and len(list(filter(lambda x: (x in i or i in x), entity_and_capitalized))) < 1 \
                and 2 > i.count(' ') > 0:
            entity_and_capitalized.append(i)
            if i in nouns:
                nouns.remove(i)
            continue
    for i in ents:
        if (len(list(filter(lambda x: (i in x or x in i), entity_and_capitalized))) < 1 \
                and i not in entity_and_capitalized \
                and i not in nouns) or (text_clean.is_date_or_number(i) and i not in nouns):
            nouns.append(i)
    for i in noun_tokens:
        if len(list(filter(lambda x: (i in x and is_capitalized(x)), entity_and_capitalized))) < 1 and i not in nouns:
            nouns.append(i)
    entity_and_capitalized = [i for i in entity_and_capitalized if i.lower() not in STOP_WORDS]
    # entity_and_capitalized = merge_phrases_l1_to_l2(entity_and_capitalized, [])
    nouns = [i for i in nouns if i.lower() not in STOP_WORDS]
    return entity_and_capitalized, nouns



def get_phrases_and_nouns_merged(sentence):
    doc_noun = nlp_eng_spacy(sentence)
    noun_tokens = []
    for token in doc_noun:
        if token.pos_.lower() in ['propn', 'noun']:
            noun_tokens.append(token.text)
        # if (token.dep_ not in ["aux", 'auxpass']) \
        #         and (token.pos_ in ["VERB", "AUX"]) \
        #         and not (token.pos_ == "AUX" and token.dep_ == "ROOT"):
        #     verbs.append(token.text)
    nouns_chunks = [chunk.text for chunk in doc_noun.noun_chunks]
    ents = [ent.text for ent in doc_noun.ents]
    capitalized_phrased = split_claim_regex(sentence)
    merged = list(set(capitalized_phrased))
    for i in nouns_chunks:
        if len(list(filter(lambda x: (i in x), capitalized_phrased))) < 1 and i not in merged:
            merged.append(i)
    for i in noun_tokens:
        if len(list(filter(lambda x: (i in x), capitalized_phrased))) < 1 and i not in merged:
            merged.append(i)
    for i in ents:
        if len(list(filter(lambda x: (i in x), capitalized_phrased))) < 1 and i not in merged:
            merged.append(i)
    # merged = capitalized_phrased
    # for i in nouns_chunks:
    #     if i not in capitalized_phrased:
    #         merged.append(i)
    # for i in noun_tokens:
    #     if i not in capitalized_phrased:
    #         merged.append(i)
    # for i in ents:
    #     if i not in capitalized_phrased:
    #         merged.append(i)
    # to_delete = []
    # for i in merged:
    #     if 'the ' + i in merged or 'a ' + i in merged:
    #         to_delete.append(i)
    # for i in to_delete:
    #     merged.remove(i)
    merged = [i for i in merged if i.lower() not in STOP_WORDS]
    return merged


def delete_ents_from_chunks(ents: list, chunks: list):
    to_delete = []
    for i in ents:
        for j in chunks:
            if j in i or j.lower() in STOP_WORDS:
                to_delete.append(j)
    chunks = list(set(chunks) - set(to_delete))
    return chunks


def merge_phrases_l1_to_l2(l1, l2):
    to_delete = []
    for i in l1:
        is_dup = False
        for j in l2:
            if i in j:
                is_dup = True
                break
            if j in i:
                to_delete.append(j)
        if not is_dup:
            l2.append(i)
    l2 = list(set(l2) - set(to_delete))
    for i in l2:
        if i.lower() in STOP_WORDS:
            l2.remove(i)
    return l2


def merge_chunks_with_entities(chunks, ents):
    merged = ents
    for c in chunks:
        if len(list(filter(lambda x: (c in x), ents))) < 1:
            merged.append(c)
    return merged


if __name__ == '__main__':
    print(get_ents_and_phrases("Giada at Home was only available on DVD."))
    # print(get_ents_and_phrases('The horse has changed in size as it evolved.'))
    # print(get_ents_and_phrases('Camden, New Jersey is a large human settlement.'))
    # print(get_ents_and_phrases('Soyuz was part of a space program.'))
    # print(get_ents_and_phrases('The New Orleans Pelicans play in the Eastern Conference of the NBA.'))

    data = file_loader.read_json_rows(config.RESULT_PATH / "hardset2021/es_doc_10.log")
    for i in data:
        claim = i['claim']
        a, b = get_ents_and_phrases(claim)
        print(claim)
        print(i['evidence'])
        print(a)
        print(b)
        print('*'*10)

