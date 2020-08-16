import concurrent.futures
import datetime
import json

from tqdm import tqdm

import config
from data_util.tokenizers import spacy_tokenizer
from utils import text_clean, fever_db
from memory_profiler import profile


def thread_exe(func, pieces, thd_num, description):
    with concurrent.futures.ThreadPoolExecutor(thd_num) as executor:
        to_be_done = {executor.submit(func, param): param for param in pieces}
        for t in tqdm(concurrent.futures.as_completed(to_be_done), total=len(list(pieces)), desc=description, position=0):
            to_be_done[t]


def get_current_time_str():
    return str(datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S'))


def wait_delay(d):
    print(d)
    # d_list = []
    # with open(d, encoding='utf-8', mode='r') as in_f:
    #     for line in in_f:
    #         item = json.loads(line.strip())
    #         d_list.append(item)
    # print(len(d_list))



def iter_baskets_contiguous(items, bunch_size):
    item_count = len(items)
    bunch_number = item_count // bunch_size if item_count % bunch_size == 0 else item_count // bunch_size + 1
    for i in range(bunch_number):
        start = i * bunch_size
        stop = (i + 1) * bunch_size
        stop = item_count if stop > item_count else stop
        yield items[start:stop]
    return


class DocIdDict(object):
    def __init__(self):
        self.tokenized_doc_id_dict = None

    def load_dict(self):
        if self.tokenized_doc_id_dict is None:
            self.tokenized_doc_id_dict = json.load(open(config.TOKENIZED_DOC_ID, encoding='utf-8', mode='r'))

    def clean(self):
        self.tokenized_doc_id_dict = None


# global tokenized_doc_id_dict
# tokenized_doc_id_dict = None
global_doc_id_object = DocIdDict()
tokenizer_spacy = spacy_tokenizer.SpacyTokenizer(annotators={'pos', 'lemma'}, model='en_core_web_sm')

def e_tokenize(text, tok):
    return tok.tokenize(text_clean.normalize(text))


def tokenize_doc_id(doc_id, tokenizer):
    # path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
    # print(path_stanford_corenlp_full_2017_06_09)
    #
    # drqa_yixin.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
    # tok = CoreNLPTokenizer(annotators=['pos', 'lemma', 'ner'])

    doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')
    tokenized_doc_id = e_tokenize(doc_id_natural_format, tokenizer)
    t_doc_id_natural_format = tokenized_doc_id.words()
    lemmas = tokenized_doc_id.lemmas()
    return t_doc_id_natural_format, lemmas

@profile
def doc_id_to_tokenized_text(doc_id, including_lemmas=False):
    # global tokenized_doc_id_dict
    # global_doc_id_object.load_dict()
    # tokenized_doc_id_dict = global_doc_id_object.tokenized_doc_id_dict
    #
    # if tokenized_doc_id_dict is None:
    #     tokenized_doc_id_dict = json.load(open(config.TOKENIZED_DOC_ID, encoding='utf-8', mode='r'))
    #
    # if including_lemmas:
    #     return tokenized_doc_id_dict[doc_id]['words'], tokenized_doc_id_dict[doc_id]['lemmas']
    #
    # return ' '.join(tokenized_doc_id_dict[doc_id]['words'])

    if including_lemmas:
        return tokenize_doc_id(doc_id, tokenizer_spacy)
    else:
        return ' '.join(tokenize_doc_id(doc_id, tokenizer_spacy)[0])


