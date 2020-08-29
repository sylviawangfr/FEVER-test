import math
import threading

import dgl
import numpy as np
import torch
from bert_serving.client import BertClient
from dbpedia_sampler import bert_similarity
from dbpedia_sampler.uri_util import uri_short_extract
from pathlib import PosixPath
from utils.file_loader import *

__all__ = ['DBpediaGATReader']


class DBpediaGATReader(object):
    def __init__(self, dbpedia_sampled_data, parallel=True, num_worker=4):
        super(DBpediaGATReader, self).__init__()
        self.graph_instances = []
        self.labels = []
        self.parallel = parallel
        self.num_worker = num_worker
        self.lock = threading.Lock()
        self._load(dbpedia_sampled_data)

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
        return self.graph_instances[idx], self.labels[idx]

    @property
    def num_classes(self):
        """Number of classes."""
        return 2

    def _load(self, dbpedia_sampled_data):
        if isinstance(dbpedia_sampled_data, PosixPath):
            dbpedia_sampled_data = read_files_one_by_one(dbpedia_sampled_data)
        if self.parallel:
            self._load_from_dbpedia_sample_multithread(dbpedia_sampled_data)
        else:
            self._load_from_dbpedia_sample_file(dbpedia_sampled_data)

    def _pair_existed(self, src, dst, pairs):
        if len(list(filter(lambda x: (src == x[0] and dst == x[1]), pairs))) < 1:
            return False
        else:
            return True

    def _load_from_list(self, list_data):
        description = "converting data to graph type:"
        if self.parallel:
            description += threading.current_thread().getName()
        tmp_lables = []
        tmp_graph_instance = []
        with tqdm(total=len(list_data), desc=description) as pbar:
            for idx, item in enumerate(list_data):
                g_claim = item['claim_links']
                pbar.update(1)
                if g_claim is None or len(g_claim) < 1:
                    continue

                candidates = item['examples']
                for c in candidates:
                    g_c = c['graph']
                    if g_c is None or len(g_c) < 1:
                        continue
                    c_label = 1 if c['selection_label'] == 'true' else 0
                    one_example = dict()
                    one_example['graph1'] = g_claim
                    one_example['graph2'] = g_c
                    tmp_lables.append(c_label)
                    tmp_graph_instance.append(one_example)
        return tmp_graph_instance, tmp_lables

    # @profile
    def _load_from_dbpedia_sample_file(self, data_or_path):
        if isinstance(data_or_path, list):
            graphs, labels = self._load_from_list(data_or_path)
            print(f"finished sampling one batch of data; count of examples: {len(labels)}")
            if self.parallel:
                self.lock.acquire()
            self.labels.extend(labels)
            self.graph_instances.extend(graphs)
            if self.parallel:
                self.lock.release()
        else:
            for idx, items in enumerate(data_or_path):
                graphs, labels = self._load_from_list(items)
                print(f"finished sampling one batch of data; count of examples: {len(labels)}")
                if self.parallel:
                    self.lock.acquire()
                self.labels.extend(labels)
                self.graph_instances.extend(graphs)
                if self.parallel:
                    self.lock.release()

    # @profile
    def _load_from_dbpedia_sample_multithread(self, dbpedia_sampled_data):
        if isinstance(dbpedia_sampled_data, list):
            batch_size = math.ceil(len(dbpedia_sampled_data) / self.num_worker)
            data_iter = iter_baskets_contiguous(dbpedia_sampled_data, batch_size)
            thread_exe(self._load_from_dbpedia_sample_file, data_iter, self.num_worker, "Multi_thread_gat_sampler\n")
        else:
            for data in dbpedia_sampled_data:
                batch_size = math.ceil(len(data) / self.num_worker)
                data_iter = iter_baskets_contiguous(data, batch_size)
                thread_exe(self._load_from_dbpedia_sample_file, data_iter, self.num_worker,
                           "Multi_thread_gat_sampler\n")
                # thread_exe(list, data_iter, num_worker, "Multi_thread_gat_sampler\n")
                del data_iter
                del data


if __name__ == '__main__':
    pass
