import math
import threading

import dgl
import numpy as np
import torch
from bert_serving.client import BertClient
from dbpedia_sampler import bert_similarity
from dbpedia_sampler.uri_util import uri_short_extract
from utils.file_loader import *

__all__ = ['DBpediaGATReader']


class DBpediaGATReader(object):
    def __init__(self, dbpedia_sampled_data_path):
        super(DBpediaGATReader, self).__init__()
        self.graph_instances = []
        self.labels = []
        self._load(dbpedia_sampled_data_path)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
            The graph pair and its label.
        """
        return self.graph_instances[idx], self.labels[idx]

    @property
    def num_classes(self):
        """Number of classes."""
        return 2

    def _load(self, data_path):
        for entry in tqdm(os.listdir(data_path)):
            one_file_data = read_json_rows(data_path / entry)
            self._load_from_list(one_file_data)
            del one_file_data
        print(f"total count: {self.__len__()}")


    def _load_from_list(self, list_data):
        description = "reading dbpedia sample data:"
        tmp_lables = []
        tmp_graph_instance = []
        with tqdm(total=len(list_data), desc=description) as pbar:
            for idx, item in enumerate(list_data):
                pbar.set_postfix_str(f"reading dbpedia sample file")
                claim_graph = item['claim_links']
                pbar.update(1)
                if len(claim_graph) < 1:
                    continue
                candidates = item['examples']
                for c in candidates:
                    c_graph = c['graph']
                    if c_graph is None or len(c_graph) < 1:
                        continue
                    c_label = 1 if c['selection_label'] == 'true' else 0
                    one_example = dict()
                    one_example['graph1'] = claim_graph
                    one_example['graph2'] = c_graph
                    self.labels.append(c_label)
                    self.graph_instances.append(one_example)


if __name__ == '__main__':
    data = config.RESULT_PATH / "sample_ss_graph_train"
    sample = DBpediaGATReader(data)
