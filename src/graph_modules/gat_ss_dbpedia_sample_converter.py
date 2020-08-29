import math
import threading

import dgl
import numpy as np
import torch
from bert_serving.client import BertClient
from dbpedia_sampler import bert_similarity
from dbpedia_sampler.uri_util import uri_short_extract
from graph_modules.gat_ss_dbpedia_reader import DBpediaGATReader
from utils.file_loader import *

__all__ = ['DBpediaGATSampleConverter']


class DBpediaGATSampleConverter(object):
    def __init__(self):
        super(DBpediaGATSampleConverter, self).__init__()
        self.graph_instances = []
        self.labels = []
        self.num_worker = False
        self.parallel = 1

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

    def __pair_existed(self, src, dst, pairs):
        if len(list(filter(lambda x: (src == x[0] and dst == x[1]), pairs))) < 1:
            return False
        else:
            return True

    def __convert_rel_to_efeature(self, triple_l, bc:BertClient):
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

    def __convert_from_list(self, graphs_and_labels):
        original_graphs, original_labels = map(list, zip(*graphs_and_labels))
        description = "converting data to graph type:"
        if self.parallel:
            description += threading.current_thread().getName()
        bc = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=60000)
        tmp_graph_instance = []
        tmp_labels = []
        for idx, item in enumerate(original_graphs):
            graph1 = item['graph1']
            graph2 = item['graph2']
            g1 = self.__convert_rel_to_efeature(graph1, bc)
            if g1 is None:
                continue
            g2 = self.__convert_rel_to_efeature(graph2, bc)
            if g2 is None:
                continue
            one_example = dict()
            one_example['graph1'] = g1
            one_example['graph2'] = g2
            tmp_graph_instance.append(one_example)
            tmp_labels.append(original_labels[idx])
        bc.close()
        return tmp_graph_instance, tmp_labels

    # @profile
    def __convert(self, graphs_and_labels):
        graphs, labels = self.__convert_from_list(graphs_and_labels)
        if self.parallel:
            self.lock.acquire()
        self.graph_instances.extend(graphs)
        self.labels.extend(labels)
        if self.parallel:
            self.lock.release()

    # @profile
    def __convert_multithread(self, graphs_and_labels):
        batch_size = math.ceil(len(graphs_and_labels) / self.num_worker)
        data_iter = iter_baskets_contiguous(graphs_and_labels, batch_size)
        thread_exe_local(self.__convert, data_iter, self.num_worker)

    def convert_dbpedia_to_dgl(self, graphs_and_labels, parallel=False, num_worker=1):
        self.parallel = parallel
        self.num_worker = num_worker
        if self.parallel:
            self.__convert_multithread(graphs_and_labels)
        else:
            self.__convert(graphs_and_labels)
        return self.graph_instances, self.labels


def thread_exe_local(func, pieces, thd_num):
    with concurrent.futures.ThreadPoolExecutor(thd_num) as executor:
        to_be_done = {executor.submit(func, param): param for param in pieces}
        for t in concurrent.futures.as_completed(to_be_done):
            to_be_done[t]


if __name__ == '__main__':
    data = read_json_rows(config.RESULT_PATH / "sample_ss_graph.jsonl")[0:12]
    datareader = DBpediaGATReader(data)
    converter = DBpediaGATSampleConverter()
    # g, l = converter.convert_dbpedia_to_dgl(data, parallel=False, num_worker=3)
    # print(len(l))
