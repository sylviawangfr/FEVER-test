from dbpedia_sampler import bert_similarity
import dgl
import torch
from dbpedia_sampler.util import uri_short_extract
import numpy as np
from tqdm import tqdm

__all__ = ['DBpediaGATSampler']


class DBpediaGATSampler(object):
    def __init__(self, dbpedia_sampled_data):
        super(DBpediaGATSampler, self).__init__()
        self.graph_instances = []
        self.labels = []
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
            g.add_nodes(len(all_nodes), {'nbd': torch.Tensor(all_node_embeddings)})
            g.add_edges(start_nums, end_nums)
            dict_nodes_inverse = dict(zip(dict_nodes.values(), dict_nodes.keys()))
            return g, dict_nodes_inverse
        else:
            return None, None

    def _convert_rel_to_efeature(self, triple_l):
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

        all_node_embeddings = bert_similarity.get_phrase_embedding(all_nodes)
        edge_embeddings = bert_similarity.get_phrase_embedding(edges_text)
        all_edge_embeddings = []
        if len(edge_embeddings) > 0:
            edge_embeddings = edge_embeddings.tolist()
            all_edge_embeddings_l = []
            for idx, p in enumerate(zip(start_nums, end_nums)):
                if p[0] == p[1]:
                    all_edge_embeddings_l.append([0] * 768)
                else:
                    all_edge_embeddings_l.append(edge_embeddings.pop(0))
            all_edge_embeddings = np.array(all_edge_embeddings_l, dtype=np.float32)
            if not len(start_nums) == len(all_edge_embeddings):
                print('error')

        if len(all_node_embeddings) > 0 and len(all_edge_embeddings) > 0:
            g = dgl.DGLGraph()
            g.add_nodes(len(all_nodes), {'nbd': torch.Tensor(all_node_embeddings)})
            g.add_edges(start_nums, end_nums, {'ebd': torch.Tensor(all_edge_embeddings)})
            dict_nodes_inversed = dict(zip(dict_nodes.values(), dict_nodes.keys()))
            return g, dict_nodes_inversed
        else:
            return None, None

    def _load(self, dbpedia_sampled_data):
        for item in tqdm(dbpedia_sampled_data):
            claim_graph = item['claim_links']
            g_claim, g_claim_dict = self._convert_rel_to_efeature(claim_graph)
            if g_claim is None:
                continue

            candidates = item['examples']
            for c in candidates:
                c_graph = c['graph']
                if c_graph is None or len(c_graph) < 1:
                    continue
                g_c, g_c_dict = self._convert_rel_to_efeature(c_graph)
                if g_c is None:
                    continue
                one_example = dict()
                one_example['selection_label'] = c['selection_label']
                one_example['selection_id'] = c['selection_id']
                one_example['graph1'] = g_claim
                one_example['graph1_dict'] = g_claim_dict
                one_example['graph2'] = g_c
                one_example['graph2_dict'] = g_c_dict
                self.labels.append(1 if c['selection_label'] == 'true' else 0)
                self.graph_instances.append(one_example)




