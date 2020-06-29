import os
from pathlib import Path
import platform

SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PRO_ROOT = SRC_ROOT.parent

UTEST_ROOT = PRO_ROOT / "utest"
DATA_ROOT = PRO_ROOT / "data"
RESULT_PATH = PRO_ROOT / "results"
LOG_PATH = PRO_ROOT / "log"

WIKI_PAGE_PATH = DATA_ROOT / "wiki-pages"
FEVER_DB = DATA_ROOT / "fever.db"

FEVER_TRAIN_JSONL = DATA_ROOT / "fever" / "train.jsonl"
FEVER_DEV_JSONL = DATA_ROOT / "fever" / "shared_task_dev.jsonl"
FEVER_DEV_UNLABELED_JSONL = DATA_ROOT / "fever" / "shared_task_dev_public.jsonl"
FEVER_TEST_JSONL = DATA_ROOT / "fever" / "shared_task_test.jsonl"

T_FEVER_TRAIN_JSONL = DATA_ROOT / "tokenized_fever" / "train.jsonl"
T_FEVER_DEV_JSONL = DATA_ROOT / "tokenized_fever" / "dev.jsonl"

T_FEVER_DT_JSONL = DATA_ROOT / "tokenized_fever" / "dev_train.jsonl"

TOKENIZED_DOC_ID = DATA_ROOT / "tokenized_doc_id.json"

WN_FEATURE_CACHE_PATH = DATA_ROOT / "wn_feature_p"

ES_INIT_LOG_PATH = LOG_PATH / "es_init"

ELASTIC_HOST = "localhost"
ELASTIC_PORT = "9200"
WIKIPAGE_MAPPING = PRO_ROOT / "src" / "ES" / "wikipage_mapping.json"
FEVER_SEN_MAPPING = PRO_ROOT / "src" / "ES" / "fever_sen_mapping.json"
WIKIPAGE_INDEX = "wikipages_tmp"
FEVER_SEN_INDEX = "fever_sentences"

DOC_RETRV_TRAIN = RESULT_PATH / "train_doc_retrieve.jsonl"
S_TFIDF_RETRV_TRAIN = RESULT_PATH / "train_s_tfidf_retrieve.jsonl"
S_TFIDF_RETRV_TEST = RESULT_PATH / "test_s_tfidf_retrieve.jsonl"
DOC_RETRV_DEV = RESULT_PATH / "dev_doc_retrieve.jsonl"
S_TFIDF_RETRV_DEV = RESULT_PATH / "dev_s_tfidf_retrieve.jsonl"
DOC_RETRV_TEST = RESULT_PATH / "test_doc_retrieve.jsonl"

# ------------------------HOSTS-----------------------
DBPEDIA_LOOKUP_PORT = 1111 if platform.system() == 'Linux' else 5001
DBPEDIA_LOOKUP_URL = f"http://localhost:{DBPEDIA_LOOKUP_PORT}/api/search/KeywordSearch?QueryString="

DBPEDIA_LOOKUP_APP_PORT = 9274 if platform.system() == 'Linux' else 5005
DBPEDIA_LOOKUP_APP_URL = f"http://localhost:{DBPEDIA_LOOKUP_APP_PORT}/lookup-application/api/search?query="

DBPEDIA_SPOTLIGHT_PORT = 2222 if platform.system() == 'Linux' else 5000
DBPEDIA_SPOTLIGHT_URL = f"http://localhost:{DBPEDIA_SPOTLIGHT_PORT}/rest/annotate"

DBPEDIA_GRAPH_PORT = 8890 if platform.system() == 'Linux' else 5002
DBPEDIA_GRAPH_URL = f"http://localhost:{DBPEDIA_GRAPH_PORT}/sparql"

BERT_SERVICE_PORT = 5555 if platform.system() == 'Linux' else 5003
BERT_SERVICE_PORT_OUT = 5556 if platform.system() == 'Linux' else 5004


if __name__ == '__main__':
    print(DBPEDIA_LOOKUP_URL)
    print(DBPEDIA_SPOTLIGHT_URL)
    print("PRO_ROOT", PRO_ROOT)
    print("SRC_ROOT", SRC_ROOT)
    print("UTEST_ROOT", UTEST_ROOT)
    print("DATA_ROOT", DATA_ROOT)
    print("TOKENIZED_DOC_ID", TOKENIZED_DOC_ID)