import torch
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
import config
from utils.fever_db import *
from bert_serving.client import BertClient


class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}
    def __call__(self):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls()
        return self._instance[self._cls]


@Singleton
class SpacyModel(object):
    def __init__(self):
        pass


@Singleton
class BERTSSModel(object):
    def __init__(self):
        model_path = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_3s_full2019_07_17_04:00:55"
        tokenizer_path = config.PRO_ROOT / "saved_models/bert_finetuning/ss_ss_3s_full2019_07_17_04:00:55"
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self):
        return self.device


@Singleton
class FeverDBResource(object):
    def __init__(self):
        self.cursor, self.conn = get_cursor()

    def get_cursor(self):
        return self.cursor

    def __del__(self):
        self.cursor.close()
        self.conn.close()


@Singleton
class BERTClientResource(object):
    def __init__(self):
        self.bc = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=60000)

    def get_client(self):
        return self.bc

    def __del__(self):
        self.bc.close()


