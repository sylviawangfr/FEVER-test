import re
import unicodedata

import regex
from dateutil.parser import parse

from data_util.tokenizers import SpacyTokenizer

tok = SpacyTokenizer()


STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
}


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def filter_word(text):
    """Take out english stopwords, punctuation, and compound endings."""
    text = normalize(text)
    if regex.match(r'^\p{P}+$', text):
        return True
    if text.lower() in STOPWORDS:
        return True
    return False


def filter_ngram(gram, mode='any'):
    """Decide whether to keep or discard an n-gram.

    Args:
        gram: list of tokens (length N)
        mode: Option to throw out ngram if
          'any': any single token passes filter_word
          'all': all tokens pass filter_word
          'ends': book-ended by filterable tokens
    """
    filtered = [filter_word(w) for w in gram]
    if mode == 'any':
        return any(filtered)
    elif mode == 'all':
        return all(filtered)
    elif mode == 'ends':
        return filtered[0] or filtered[-1]
    else:
        raise ValueError('Invalid mode: %s' % mode)


def check_arabic(input_string):
    res = re.findall(
        r'[\U00010E60-\U00010E7F]|[\U0001EE00-\U0001EEFF]|[\u0750-\u077F]|[\u08A0-\u08FF]|[\uFB50-\uFDFF]|[\uFE70-\uFEFF]|[\u0600-\u06FF]', input_string)
    # [\U00010E60-\U00010E7F]+|[\U0001EE00-\U0001EEFF]+
    
    # print(res)
    if len(res) != 0:
        return True
    else:
        return False


def is_date(string, fuzzy=False):
    """
        Return whether the string can be interpreted as a date.

        :param string: str, string to check for date
        :param fuzzy: bool, ignore unknown tokens in string if True
        """
    try:
        parse(string, fuzzy=fuzzy)
        return True
    except ValueError:
        return False
    except TypeError:
        print(f"type error: {string}")
        return False


def easy_tokenize(text):
    if tok is None:
        tok.create_instance()
    return tok.tokenize(normalize(text)).words()


NUMERIC_WORDS = ['zero', 'one', 'two','three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
             'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
             'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
             'first', 'second', 'third', 'fourth', 'fifty', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh'
                 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth',
                 'nineteenth', 'twentieth', 'thirtieth', 'fortieth', 'fiftieth', 'sixtieth', 'seventieth', 'eightieth',
                 'ninetieth', 'hundredth', 'thousandth', 'hundred', 'hundreds', 'thousand', 'thousands']


def is_number(str):
    is_num = str.isnumeric()
    if not is_num:
        tokens = easy_tokenize(str)
        tokens = [i.lower() for i in tokens if re.match(r'(\w)+', i)]
        is_num = True
        for i in tokens:
            if i not in NUMERIC_WORDS and not i.isnumeric():
                is_num = False
                break
    return is_num


def filter_document_id(input_string):
    pid_words = input_string.strip().replace('_', ' ')
    match = re.search('[a-zA-Z]', pid_words)
    if match is None:
        return True
    elif check_arabic(pid_words):
        return True
    else:
        return False


# convert_special
def convert_brc(string):
    string = re.sub('-LRB-', '(', string)
    string = re.sub('-RRB-', ')', string)
    string = re.sub('-LSB-', '[', string)
    string = re.sub('-RSB-', ']', string)
    string = re.sub('-LCB-', '{', string)
    string = re.sub('-RCB-', '}', string)
    string = re.sub('-COLON-', ':', string)
    return string


def reverse_convert_brc(string):
    string = re.sub('\(', '-LRB-', string)
    string = re.sub('\)', '-RRB-', string)
    string = re.sub('\[', '-LSB-', string)
    string = re.sub(']', '-RSB-', string)
    string = re.sub('{', '-LCB-', string)
    string = re.sub('}', '-RCB-', string)
    string = re.sub(':', '-COLON-', string)
    return string


if __name__ == '__main__':
    t = normalize("""["French", "Franco-Seychellois", "E\u0301tienne de Silhouette", "E\u0301tienne de Silhouette", "Louis XV", "Louis XV"]""")
    # is_date('October 18 , 2008')
    is_date('1980-83')
    is_number('1980-83')
    is_number('sixty two')
    pass
    # print(filter_word("what is going on"))

    # print(check_arabic('afadرش'))
    # print(check_arabic('رشيد'))