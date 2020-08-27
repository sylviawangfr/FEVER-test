import math
import threading

import dgl
import numpy as np
import torch
from bert_serving.client import BertClient
from dbpedia_sampler import bert_similarity
from dbpedia_sampler.uri_util import uri_short_extract
from utils.file_loader import *

__all__ = ['DBpediaGATSampleConverter']


class DBpediaGATSampleConverter(object):
    def __init__(self):
        super(DBpediaGATSampleConverter, self).__init__()
        self.graph_instances = []
        self.labels = []
        self.parallel = True
        self.num_worker = 3
        self.lock = threading.Lock()

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

    def _pair_existed(self, src, dst, pairs):
        if len(list(filter(lambda x: (src == x[0] and dst == x[1]), pairs))) < 1:
            return False
        else:
            return True

    def _convert_rel_to_efeature(self, triple_l, bc:BertClient):
        cleaned_tris = []
        for tri in triple_l:
            cleaned_t = []
            s = uri_short_extract(tri['subject']).lower()
            r = uri_short_extract(tri['relation']).lower()
            o = uri_short_extract(tri['object']).replace('Category', '').lower()
            if len(list(filter(lambda x: (s == x[0] and r == x[1] and o == x[2]), cleaned_tris))) < 1:
                cleaned_t.extend([s, r, o])
                cleaned_tris.append(cleaned_t)

        all_nodes = set()
        all_nodes.update(set([i[0] for i in cleaned_tris]))
        all_nodes.update(set([i[2] for i in cleaned_tris if i[2] != '']))
        all_nodes = list(all_nodes)

        # text -> num dict
        dict_nodes = dict()
        for idx, n in enumerate(all_nodes):
            dict_nodes[n] = idx

        start_nums = []
        end_nums = []
        edges_text = []
        for tri in cleaned_tris:
            if tri[2] != '':
                start_nums.append(dict_nodes[tri[0]])
                end_nums.append(dict_nodes[tri[2]])
                edges_text.append(tri[1])
            # add self loop
        for i in dict_nodes.values():
            start_nums.append(i)
            end_nums.append(i)

        all_node_embeddings = bert_similarity.get_phrase_embedding(all_nodes, bc)
        edge_embeddings = bert_similarity.get_phrase_embedding(edges_text, bc)
        all_edge_embeddings = []
        if len(edge_embeddings) > 0:
            edge_embeddings = edge_embeddings.tolist()
            all_edge_embeddings_l = []
            feature_dim = len(edge_embeddings[0])
            for idx, p in enumerate(zip(start_nums, end_nums)):
                if p[0] == p[1]:
                    all_edge_embeddings_l.append([0] * feature_dim)
                else:
                    all_edge_embeddings_l.append(edge_embeddings.pop(0))
            all_edge_embeddings = np.array(all_edge_embeddings_l, dtype=np.float32)
            if not len(start_nums) == len(all_edge_embeddings):
                print('error')

        if len(all_node_embeddings) > 0 and len(all_edge_embeddings) > 0:
            g = dgl.DGLGraph()
            g.add_nodes(len(all_nodes), {'nbd': torch.Tensor(np.copy(all_node_embeddings))})
            g.add_edges(start_nums, end_nums, {'ebd': torch.Tensor(np.copy(all_edge_embeddings))})
            return g
        else:
            return None

    def _load_from_list(self, list_data):
        description = "converting data to graph type:"
        if self.parallel:
            description += threading.current_thread().getName()
        bc = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=60000)
        tmp_lables = []
        tmp_graph_instance = []
        with tqdm(total=len(list_data), desc=description) as pbar:
            for idx, item in enumerate(list_data):
                claim_graph = item['claim_links']
                g_claim = self._convert_rel_to_efeature(claim_graph, bc)
                pbar.update(1)
                if g_claim is None:
                    continue

                candidates = item['examples']
                for c in candidates:
                    c_graph = c['graph']
                    if c_graph is None or len(c_graph) < 1:
                        continue
                    g_c = self._convert_rel_to_efeature(c_graph, bc)
                    if g_c is None:
                        continue
                    c_label = 1 if c['selection_label'] == 'true' else 0
                    one_example = dict()
                    one_example['graph1'] = g_claim
                    one_example['graph2'] = g_c
                    tmp_lables.append(c_label)
                    tmp_graph_instance.append(one_example)
        bc.close()
        return tmp_graph_instance, tmp_lables

    # @profile
    def _load_from_dbpedia_sample_file(self, dbpedia_sampled_data):
        if isinstance(dbpedia_sampled_data, list):
            graphs, labels = self._load_from_list(dbpedia_sampled_data)
            print(f"finished sampling one batch of files; count of examples: {len(labels)}")
            if self.parallel:
                self.lock.acquire()
            self.labels.extend(labels)
            self.graph_instances.extend(graphs)
            if self.parallel:
                self.lock.release()
        else:
            for idx, items in enumerate(dbpedia_sampled_data):
                graphs, labels = self._load_from_list(items)
                print(f"finished sampling one batch of files; count of examples: {len(labels)}")
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
                thread_exe(self._load_from_dbpedia_sample_file, data_iter, self.num_worker, "Multi_thread_gat_sampler\n")
                # thread_exe(list, data_iter, num_worker, "Multi_thread_gat_sampler\n")
                del data_iter
                del data

    def convert_dbpedia_to_dgl(self, dbpedia_sampled_data, parallel=True, num_worker=3):
        self.parallel = parallel
        self.num_worker = num_worker
        if self.parallel:
            self._load_from_dbpedia_sample_multithread(dbpedia_sampled_data)
        else:
            self._load_from_dbpedia_sample_file(dbpedia_sampled_data)
        return self.graph_instances, self.labels


if __name__ == '__main__':
    data = read_json_rows(config.RESULT_PATH / "sample_ss_graph.jsonl")[0:12]
    converter = DBpediaGATSampleConverter()
    g, l = converter.convert_dbpedia_to_dgl(data, parallel=True, num_worker=3)
