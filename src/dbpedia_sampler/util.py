import re
import validators

def uri_short_extract(uri):
    lastword = uri.split('/')[-1]
    lastword = lastword.split('#')[-1]
    words = wildcase_split(lastword)
    phrases = []
    for w in words:
        ph = camel_case_split(w)
        phrases.append(' '.join([ww for ww in ph]))
    one_phrase = ' '.join(p for p in phrases)
    return one_phrase

def wildcase_split(str):
    p_l = re.findall(r'(?:\d+\.\d+)|(?:\d+)|(?:[a-zA-Z]+)', str)
    return list(filter(None, p_l))


def camel_case_split(str):
    return re.findall(r'(?:\d+\.\d+)|(?:\d+)|(?:^[a-z]+)|(?:[A-Z]+)(?:[a-z]*|[A-Z]*(?=[A-Z]|$))', str)


# "Johnson in 2007"^^<http://www.w3.org/1999/02/22-rdf-syntax-ns#langString>
def property_split(str):
    value_or_type = str.split('^^')
    return value_or_type[0]


def isURI(str):
    return validators.url(str)
