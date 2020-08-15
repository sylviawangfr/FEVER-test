import spacy
from spacy.symbols import nsubj, dobj, pobj, VERB, ADP, AUX, acomp
from spacy import displacy
import config
from spacy.tokens import Doc, Span, Token
import regex
from nltk.stem.wordnet import WordNetLemmatizer
from utils.tokenizer_simple import nlp_eng_spacy as nlp

# used part of codes from https://github.com/NSchrading/intro-spacy-nlp.git

def split_claim_spacy(text):
    # doc_multi = nlp_multi(text)
    doc_noun = nlp(text)
    nouns = [chunk.text for chunk in doc_noun.noun_chunks]
    ents = ([(ent.text, ent.label_) for ent in doc_noun.ents])
    # print(nouns)
    # print(ents)
    return nouns, ents


def split_claim_regex(text):
    # get capital phrases
    # REGEX = r'(?<![.])([A-Z]+[\w]*\s)*([A-Z][\w]+)'
    REGEX = r'([a-z0-9]*[A-Z]+[\w]*(\s)*' \
            r'(of\s)*(to\s)*(for\s)*(at\s)*(in\s)*(on\s)*(from\s)*(and\s)*(with\s)*(the\s)*' \
            r'(-?)(\d*\s)*)*(?<!-\s)([A-Z]+[\w]*(\s\d+)*)'
    regexp = regex.compile(REGEX)
    matches = [m for m in regexp.finditer(text)]
    tokens = [matches[i].group() for i in range(len(matches))]
    return tokens


def get_phrase_token_indice(sent_token_l, phrase_token_l):
    i = 0
    j = 0
    idx = []
    while i < len(sent_token_l):
        tmp_i = i
        while j < len(phrase_token_l) and i < len(sent_token_l):
            if not sent_token_l[i] == phrase_token_l[j]:
                i = i + 1
                j = 0
            else:
                i = i + 1
                j = j + 1
        if j == len(phrase_token_l):
            idx.append((i - len(phrase_token_l), i))
            j = 0
        else:
            i = tmp_i + 1
            j = 0
    return idx


def merge_phrases_as_span(sent, phrase_l):
    doc_to_merge = nlp(sent)
    for ph in phrase_l:
        phrase = nlp(ph)
        doc_tokens = [token.text for token in doc_to_merge]
        phrase_tokens = [token.text for token in phrase]
        try:
            phrase_idx = get_phrase_token_indice(doc_tokens, phrase_tokens)
            if len(phrase_idx) > 0:
                idx = phrase_idx[0]
                with doc_to_merge.retokenize() as retokenizer:
                    retokenizer.merge(doc_to_merge[idx[0]:idx[1]], attrs={"LEMMA": ph})
            else:
                continue
        except Exception as err:
            print(err)
    return doc_to_merge


def get_dependent_verb(sent, phrase_l):
    doc_merged = merge_phrases_as_span(sent, phrase_l)
    # displacy.serve(doc_merged, style='dep')
    # svg = displacy.render(doc_merged, style="dep")
    # output_path = config.LOG_PATH / 'sentence.svg'
    # output_path.open("w", encoding="utf-8").write(svg)
    phs = dict()
    for ph in phrase_l:
        for possible_phrase in doc_merged:
            if possible_phrase.text == ph:
                one_p = {'dep': '', 'verb': ''}
                if possible_phrase.dep == nsubj:
                    one_p['dep'] = 'subj'
                if possible_phrase.dep == dobj or possible_phrase.dep == pobj:
                    one_p['dep'] = 'obj'
                if possible_phrase.head.pos == VERB:
                    one_p['verb'] = possible_phrase.head.text
                else:
                    if possible_phrase.head.head.pos == VERB:
                        one_p['verb'] = possible_phrase.head.head.text
                    else:
                        try:
                            next_w = next(possible_phrase.head.rights)
                            if next_w.pos == VERB:
                                one_p['verb'] = next_w.text
                        except StopIteration:
                            pass
                phs[ph] = one_p
    return phs

SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd", "pobj"]

def get_triple(sent, phrase_l):
    doc_merged = merge_phrases_as_span(sent, phrase_l)
    # displacy.serve(doc_merged, style='dep')
    # svg = displacy.render(doc_merged, style="dep")
    svos = findSVOs(doc_merged)
    print(svos)


def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs

def getObjsFromConjunctions(objs):
    moreObjs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs

def getVerbsFromConjunctions(verbs):
    moreVerbs = []
    for verb in verbs:
        rightDeps = {tok.lower_ for tok in verb.rights}
        if "and" in rightDeps:
            moreVerbs.extend([tok for tok in verb.rights if tok.pos_ == "VERB"])
            if len(moreVerbs) > 0:
                moreVerbs.extend(getVerbsFromConjunctions(moreVerbs))
    return moreVerbs

def findSubs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB" or head.pos_ == "AUX":
        subs = [tok for tok in head.lefts if tok.dep_ in SUBJECTS]
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == "NOUN":
        return [head], isNegated(tok)
    return [], False

def isNegated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False

def findSVs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs

def getObjsFromPrepositions(deps):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and (dep.dep_ == "prep" or dep.dep_ == "agent"):
            objs.extend([tok for tok in dep.rights if tok.dep_ in OBJECTS or (tok.pos_ == "PRON" and tok.lower_ == "me")])
    return objs

def getObjsFromAttrs(deps):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(getObjsFromPrepositions(rights))
                    if len(objs) > 0:
                        return v, objs
    return None, None


def getObjFromAComp(deps):
    for dep in deps:
        if dep.pos_ == "ADJ" and dep.dep_ == "acomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None


def getObjFromXComp(deps):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None

def getAllSubs(v):
    verbNegated = isNegated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
    return subs, verbNegated

def getAllObjs(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    objs.extend(getObjsFromPrepositions(rights))

    #potentialNewVerb, potentialNewObjs = getObjsFromAttrs(rights)
    #if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
    #    objs.extend(potentialNewObjs)
    #    v = potentialNewVerb

    potentialNewVerb, potentialNewObjs = getObjFromAComp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    return v, objs


def find_ADP_for_obj(obj):
    head = obj.head
    if head.pos_ == "ADP" and head.dep_ in ['prep', 'agent']:
        return head
    elif obj.dep_ == "conj" and head.head.pos_ == "ADP" and head.head.dep_ in ['prep', 'agent']:
        return head.head
    else:
        return None


# fell asleep
def find_vert_ADJ(v):
    verb_ADJ = [tok for tok in v.rights if (tok.dep_ == "acomp") and (tok.pos_ in ["ADJ"])]
    return verb_ADJ


def find_verbs(tokens):
    verbs = [tok for tok in tokens if (tok.dep_ != "aux") and (tok.pos_ in ["VERB", "ROOT", "AUX"])]
    return verbs


def findSVOs(tokens):
    svos = []
    verbs = find_verbs(tokens)
    for v in verbs:
        # complete_v(v)
        subs, verbNegated = getAllSubs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            v, objs = getAllObjs(v)
            for sub in subs:
                for obj in objs:
                    objNegated = isNegated(obj)
                    rel_text = "!" + v.lower_ if verbNegated or objNegated else v.lower_
                    relADP = find_ADP_for_obj(obj)
                    verb_adj = find_vert_ADJ(v)
                    if len(verb_adj) > 0:
                        rel_text = rel_text + " " + verb_adj[0].lower_
                    if relADP is not None:
                        rel_text = rel_text + " " + relADP.lower_
                    svos.append((sub.lower_, rel_text, obj.lower_))
    return svos


def getAbuserOntoVictimSVOs(tokens):
    maleAbuser = {'he', 'boyfriend', 'bf', 'father', 'dad', 'husband', 'brother', 'man'}
    femaleAbuser = {'she', 'girlfriend', 'gf', 'mother', 'mom', 'wife', 'sister', 'woman'}
    neutralAbuser = {'pastor', 'abuser', 'offender', 'ex', 'x', 'lover', 'church', 'they'}
    victim = {'me', 'sister', 'brother', 'child', 'kid', 'baby', 'friend', 'her', 'him', 'man', 'woman'}

    svos = findSVOs(tokens)
    wnl = WordNetLemmatizer()
    passed = []
    for s, v, o in svos:
        s = wnl.lemmatize(s)
        v = "!" + wnl.lemmatize(v[1:], 'v') if v[0] == "!" else wnl.lemmatize(v, 'v')
        o = "!" + wnl.lemmatize(o[1:]) if o[0] == "!" else wnl.lemmatize(o)
        if s in maleAbuser.union(femaleAbuser).union(neutralAbuser) and o in victim:
            passed.append((s, v, o))
    return passed

def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])

if __name__ == '__main__':
    # testSVOs()
    # d_l = ['two', 'three', 'four', 'six', 'one', 'two', 'thirty', 'one', 'two']
    # p_l = ['one', 'two']
    # get_phrase_token_indice(d_l, p_l)

    # text = "Bessie Smith was hired by Ken on April 15, 1894."
    ph = ['Bessie Smith', 'April 15, 1894']
    # get_triple(text, [])
    # text = "When sitting in the chair, Bessie Smith fell asleep on April 15, 1894."
    text = "Bessie Smith go to school with Ken on April 15, 1894."
    # text = "Bessie Smith fell in love with Ken on April 15, 1894."
    # printDeps(nlp(text))
    get_triple(text, ph)
