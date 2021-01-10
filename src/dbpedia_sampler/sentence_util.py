import log_util
from utils import c_scorer, text_clean, file_loader
from utils.tokenizer_simple import *
import config
from memory_profiler import profile

STOP_WORDS = ['the', 'they', 'i', 'me', 'you', 'she', 'he', 'it', 'individual', 'individuals', 'year', 'years', 'day', 'night',
               'we', 'who', 'where', 'what', 'days', 'him', 'her', 'here', 'there', 'a', 'for',
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
    entity_and_capitalized, nouns = get_ents_and_phrases(sent)
    if not doc_title == '':
        entity_and_capitalized = list(set(entity_and_capitalized) | {doc_title})
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
        if token.pos_.lower() in ['propn', 'noun']:
            noun_tokens.append(token.text)
    nouns_chunks = [remove_the_a(chunk.text) for chunk in doc_noun.noun_chunks]
    ents = [remove_the_a(ent.text) for ent in doc_noun.ents]
    capitalized_phrased = list(set(split_claim_regex(sentence)))

    for i in ents:
        if len(list(filter(lambda x: (i in x or x in i), capitalized_phrased))) < 1 \
                and i not in capitalized_phrased \
                and i not in nouns_chunks:
            nouns_chunks.append(i)

    nouns = list(set(nouns_chunks))
    entity_and_capitalized = [i for i in capitalized_phrased if i.lower() not in STOP_WORDS]
    for i in nouns_chunks:
        if len(list(filter(lambda x: (i in x), entity_and_capitalized))) > 0 and i in nouns:
            nouns.remove(i)
    for i in nouns_chunks:
        if len(list(filter(lambda x: (x in i and sentence.startswith(x) and ' ' not in x), entity_and_capitalized))) > 0 and i in nouns:
            nouns.remove(i)
            entity_and_capitalized.append(i)
    for i in noun_tokens:
        if len(list(filter(lambda x: (i in x), capitalized_phrased))) < 1 and i not in nouns:
            nouns.append(i)
    entity_and_capitalized = [i for i in entity_and_capitalized if i.lower() not in STOP_WORDS]
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


if __name__ == '__main__':
    # get_ents_and_phrases("The human brain contains a hypothalamus.")
    get_ents_and_phrases("Turin's Juventus Stadium is the home arena for Juventus F.C.")

    data = file_loader.read_json_rows(config.RESULT_PATH / "extend_20210106/candidate_docs2.log")[3:50]
    for i in data:
        claim = i['claim']
        a, b = get_ents_and_phrases(claim)
        print(claim)
        print(a)
        print(b)
        print('*'*10)

