import log_util
from utils import c_scorer, text_clean
from utils.tokenizer_simple import *
from memory_profiler import profile

STOP_WORDS = ['they', 'i', 'me', 'you', 'she', 'he', 'it', 'individual', 'individuals', 'year', 'years', 'day', 'night',
               'we', 'who', 'where', 'what', 'days', 'him', 'her','here', 'there', 'a', 'for',
              'which', 'when', 'whom', 'the', 'history', 'morning', 'afternoon', 'evening', 'night', 'first', 'second',
              'third']


log = log_util.get_logger('dbpedia_triple_linker')

# @profile
def get_phrases(sentence, doc_title=''):
    log.debug(sentence)
    if doc_title != '' and c_scorer.SENT_DOC_TITLE in sentence and sentence.startswith(doc_title):
        title_and_sen = sentence.split(c_scorer.SENT_DOC_TITLE, 1)
        sent = title_and_sen[1]
    else:
        sent = sentence

    chunks, ents = split_claim_spacy(sent)
    entities = [en[0] for en in ents]
    capitalized_phrased = split_claim_regex(sent)
    log.debug(f"chunks: {chunks}")
    log.debug(f"entities: {entities}")
    log.debug(f"capitalized phrases: {capitalized_phrased}")
    merged_entities = merge_phrases_l1_to_l2(capitalized_phrased, entities)
    if not doc_title == '':
        merged_entities = list(set(merged_entities) | set([doc_title]))
    merged_entities = [i for i in merged_entities if i.lower() not in STOP_WORDS]
    other_chunks = delete_ents_from_chunks(merged_entities, chunks)
    log.debug(f"merged entities: {merged_entities}")
    log.debug(f"other phrases: {other_chunks}")
    return merged_entities, other_chunks


def get_phrases_and_nouns(sentence):
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
    # merged_nouns = [i for i in capitalized_phrased]
    # for i in nouns_chunks:
    #     if len(list(filter(lambda x: (i in x), capitalized_phrased))) < 1:
    #         merged_nouns.append(i)
    # for i in noun_tokens:
    #     if len(list(filter(lambda x: (i in x), capitalized_phrased))) < 1:
    #         merged_nouns.append(i)
    # for i in ents:
    #     if len(list(filter(lambda x: (i in x), capitalized_phrased))) < 1:
    #         merged_nouns.append(i)
    merged = capitalized_phrased
    for i in nouns_chunks:
        if i not in capitalized_phrased:
            merged.append(i)
    for i in noun_tokens:
        if i not in capitalized_phrased:
            merged.append(i)
    for i in ents:
        if i not in capitalized_phrased:
            merged.append(i)
    return merged
    to_delete = []
    for i in merged_nouns:
        if 'the ' + i in merged_nouns or 'a ' + i in merged_nouns:
            to_delete.append(i)
    for i in to_delete:
        merged_nouns.remove(i)
    return merged_nouns


def delete_ents_from_chunks(ents: list, chunks: list):
    to_delete = []
    for i in ents:
        for j in chunks:
            if j in i or j.lower() in STOP_WORDS:
                to_delete.append(j)
    chunks = list(set(chunks) - set(to_delete))
    return chunks


def is_date_or_number(phrase):
    return text_clean.is_date(phrase) or text_clean.is_number(phrase)


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


