import math
import threading

import dgl
import numpy as np
import torch
from bert_serving.client import BertClient
from dbpedia_sampler import bert_similarity
from dbpedia_sampler.uri_util import uri_short_extract
from utils.file_loader import *

__all__ = ['DBpediaGATSampler']


class DBpediaGATSampler(object):
    def __init__(self, dbpedia_sampled_data, parallel=False):
        super(DBpediaGATSampler, self).__init__()
        self.graph_instances = []
        self.labels = []
        self.parallel = parallel
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
        if self.parallel:
            self._load_from_dbpedia_sample_multithread(dbpedia_sampled_data)
        else:
            self._load_from_dbpedia_sample_file(dbpedia_sampled_data)


    def _pair_existed(self, src, dst, pairs):
        if len(list(filter(lambda x: (src == x[0] and dst == x[1]), pairs))) < 1:
            return False
        else:
            return True

    def _convert_rel_to_node(self, triple_l):
        cleaned_pairs = []
        single_nodes = []
        for p in triple_l:
            cleaned_p = []
            s = uri_short_extract(p['subject']).lower()
            r = uri_short_extract(p['relation']).lower()
            o = uri_short_extract(p['object']).replace('Category', '').lower()
            if o == '':
                single_nodes.append(s)
                continue
            if (r == 'subject' or r == '') and not self._pair_existed(s, o, cleaned_pairs):
                cleaned_p.append([s, o])
                cleaned_pairs.extend(cleaned_p)
            else:
                if not self._pair_existed(s, r, cleaned_pairs):
                    cleaned_p.append([s, r])
                if not self._pair_existed(r, o, cleaned_pairs):
                    cleaned_p.append([r, o])
                cleaned_pairs.extend(cleaned_p)

        all_nodes = set()
        all_nodes.update(set([i[0] for i in cleaned_pairs]))
        all_nodes.update(set([i[1] for i in cleaned_pairs]))
        single_nodes = list(set(single_nodes))
        all_nodes.update(set(single_nodes))
        all_nodes = list(all_nodes)

        # text -> num dict
        dict_nodes = dict()
        for idx, n in enumerate(all_nodes):
            dict_nodes[n] = idx

        start_nums = []
        end_nums = []
        for p in cleaned_pairs:
            start_nums.append(dict_nodes[p[0]])
            end_nums.append(dict_nodes[p[1]])
        for n in single_nodes:
            start_nums.append(dict_nodes[n])
            end_nums.append(dict_nodes[n])
        all_node_embeddings = bert_similarity.get_phrase_embedding(all_nodes)

        if len(all_node_embeddings) > 0:
            g = dgl.DGLGraph()
            g.add_nodes(len(all_nodes), {'nbd': torch.Tensor(all_node_embeddings.clone())})
            g.add_edges(start_nums, end_nums)
            return g
        else:
            return None

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

    def _load_from_dbpedia_sample_file(self, dbpedia_sampled_data):
        description = "converting data to graph type:"
        bc = BertClient(port=config.BERT_SERVICE_PORT, port_out=config.BERT_SERVICE_PORT_OUT, timeout=60000)
        tmp_lables = []
        tmp_graph_instance = []
        if self.parallel:
            description += threading.current_thread().getName()
        with tqdm(total=len(dbpedia_sampled_data), desc=description) as pbar:
            for idx, item in enumerate(dbpedia_sampled_data):
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
        if self.parallel:
            self.lock.acquire()
        self.labels.extend(tmp_lables)
        self.graph_instances.extend(tmp_graph_instance)
        if self.parallel:
            self.lock.release()
        bc.close()

    def _load_from_dbpedia_sample_multithread(self, dbpedia_sampled_data):
        num_worker = 3
        if isinstance(dbpedia_sampled_data, list):
            batch_size = math.ceil(len(dbpedia_sampled_data) / num_worker)
            data_iter = iter_baskets_contiguous(dbpedia_sampled_data, batch_size)
            thread_exe(self._load_from_dbpedia_sample_file, data_iter, num_worker, "Multi_thread_gat_sampler\n")
        else:
            for data in dbpedia_sampled_data:
                batch_size = math.ceil(len(data) / 3)
                data_iter = iter_baskets_contiguous(data, batch_size)
                # thread_exe(self._load_from_dbpedia_sample_file, data_iter, num_worker, "Multi_thread_gat_sampler\n")
                thread_exe(len, data_iter, num_worker, "Multi_thread_gat_sampler\n")
                del data_iter
                del data


if __name__ == '__main__':
    data = read_json_rows(config.RESULT_PATH / "sample_ss_graph.jsonl")[0:12]
    sample = DBpediaGATSampler(data, parallel=True)
