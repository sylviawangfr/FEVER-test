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
        self._load(dbpedia_sampled_data_path)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_instances)

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
        return self.graph_instances[idx]

    @property
    def num_classes(self):
        """Number of classes."""
        return 2

    def _load(self, data_path):
        for entry in tqdm(os.listdir(data_path)):
            one_file_data = read_json_rows(data_path / entry)
            self.graph_instances.extend(one_file_data)
        print(f"total count: {self.__len__()}")


if __name__ == '__main__':
    data = config.RESULT_PATH / "sample_ss_graph_train"
    sample = DBpediaGATReader(data)
