import config
from dbpedia_sampler.dbpedia_triple_linker import find_linked_phrases
from utils import text_clean, file_loader
from utils.triple_extractor import get_triple


def extract_triples_for_claims(data_iter):
    for i in data_iter:
        claim = text_clean.convert_brc(i['claim'])
        phrases = find_linked_phrases(claim)
        print(claim)
        get_triple(claim, phrases)


data = file_loader.read_json_rows(config.FEVER_DEV_JSONL)[0:5]
extract_triples_for_claims(data)
