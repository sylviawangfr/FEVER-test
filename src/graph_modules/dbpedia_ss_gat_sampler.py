from dbpedia_sampler import bert_similarity
import dgl
import torch
from dbpedia_sampler.util import uri_short_extract
import numpy as np
from tqdm import tqdm
from utils.file_loader import *


__all__ = ['DBpediaGATSampler']


class DBpediaGATSampler(object):
    def __init__(self, dbpedia_sampled_data, from_gat=False, save=False):
        super(DBpediaGATSampler, self).__init__()
        self.graph_instances = []
        self.labels = []
        self._load(dbpedia_sampled_data, from_gat=from_gat, save=save)

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

    def _load(self, data, from_gat=False, save=False):
        self._load_from_dbpedia_sample_file(data, save)
        # if from_gat:
        #     self._load_from_gat_sample_file(data)
        # else:
        #     self._load_from_dbpedia_sample_file(data, save)

    def _pair_existed(self, src, dst, pairs):
        if len(list(filter(lambda x: (src == x[0] and dst == x[1]), pairs))) < 1:
            return False
        else:
            return True

    # def _convert_rel_to_node(self, triple_l):
    #     cleaned_pairs = []
    #     single_nodes = []
    #     for p in triple_l:
    #         cleaned_p = []
    #         s = uri_short_extract(p['subject']).lower()
    #         r = uri_short_extract(p['relation']).lower()
    #         o = uri_short_extract(p['object']).replace('Category', '').lower()
    #         if o == '':
    #             single_nodes.append(s)
    #             continue
    #         if (r == 'subject' or r == '') and not self._pair_existed(s, o, cleaned_pairs):
    #             cleaned_p.append([s, o])
    #             cleaned_pairs.extend(cleaned_p)
    #         else:
    #             if not self._pair_existed(s, r, cleaned_pairs):
    #                 cleaned_p.append([s, r])
    #             if not self._pair_existed(r, o, cleaned_pairs):
    #                 cleaned_p.append([r, o])
    #             cleaned_pairs.extend(cleaned_p)
    #
    #     all_nodes = set()
    #     all_nodes.update(set([i[0] for i in cleaned_pairs]))
    #     all_nodes.update(set([i[1] for i in cleaned_pairs]))
    #     single_nodes = list(set(single_nodes))
    #     all_nodes.update(set(single_nodes))
    #     all_nodes = list(all_nodes)
    #
    #     # text -> num dict
    #     dict_nodes = dict()
    #     for idx, n in enumerate(all_nodes):
    #         dict_nodes[n] = idx
    #
    #     start_nums = []
    #     end_nums = []
    #     for p in cleaned_pairs:
    #         start_nums.append(dict_nodes[p[0]])
    #         end_nums.append(dict_nodes[p[1]])
    #     for n in single_nodes:
    #         start_nums.append(dict_nodes[n])
    #         end_nums.append(dict_nodes[n])
    #     all_node_embeddings = bert_similarity.get_phrase_embedding(all_nodes)
    #
    #     if len(all_node_embeddings) > 0:
    #         g = dgl.DGLGraph()
    #         g.add_nodes(len(all_nodes), {'nbd': torch.Tensor(all_node_embeddings.clone())})
    #         g.add_edges(start_nums, end_nums)
    #         dict_nodes_inverse = dict(zip(dict_nodes.values(), dict_nodes.keys()))
    #         return g, dict_nodes_inverse
    #     else:
    #         return None, None

    def _convert_rel_to_efeature(self, triple_l, return_with_data=False):
        one_example_data = dict()
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
            g.add_nodes(len(all_nodes), {'nbd': torch.Tensor(np.copy(all_node_embeddings))})
            g.add_edges(start_nums, end_nums, {'ebd': torch.Tensor(np.copy(all_edge_embeddings))})
            if return_with_data:
                one_example_data['nodes'] = len(all_nodes)
                one_example_data['edges'] = {'src': start_nums, 'dst': end_nums}
                one_example_data['node_embeddings'] = all_edge_embeddings.tolist()
                one_example_data['edge_embeddings'] = all_edge_embeddings.tolist()
                return g, one_example_data
            else:
                return g, None
        else:
            return None, None

    def _load_from_dbpedia_sample_file(self, dbpedia_sampled_data, save=False):
        dt = get_current_time_str()
        with tqdm(total=len(dbpedia_sampled_data), desc=f"converting data to graph type:") as pbar:
            for idx, item in enumerate(dbpedia_sampled_data):
                claim_graph = item['claim_links']
                g_claim, g_claim_data = self._convert_rel_to_efeature(claim_graph, save)
                pbar.update(1)
                if g_claim is None:
                    continue

                # save_c_data_l = []
                candidates = item['examples']
                for c in candidates:
                    c_graph = c['graph']
                    if c_graph is None or len(c_graph) < 1:
                        continue
                    g_c, g_c_data = self._convert_rel_to_efeature(c_graph, save)
                    if g_c is None:
                        continue
                    c_label = 1 if c['selection_label'] == 'true' else 0
                    # if save:
                    #     g_c_data.update({'label': c_label})
                    #     save_c_data_l.append(g_c_data)
                    one_example = dict()
                    # one_example['selection_label'] = c['selection_label']
                    # one_example['selection_id'] = c['selection_id']
                    one_example['graph1'] = g_claim
                    # one_example['graph1_dict'] = g_claim_dict
                    one_example['graph2'] = g_c
                    # one_example['graph2_dict'] = g_c_dict
                    self.labels.append(c_label)
                    self.graph_instances.append(one_example)
                # if save:
                #     save_batch = []
                #     save_example_data = dict()
                #     save_example_data.update({'claim': g_claim_data})
                #     save_example_data.update({'candidates': save_c_data_l})
                #     if len(save_batch) < 5 or idx < len(dbpedia_sampled_data) - 1:
                #         save_batch.append(save_example_data)
                #     else:
                #         append_results(save_batch, config.RESULT_PATH / f"gat_ss_{dt}.jsonl")
                #         save_batch.clear()

    # def _create_graph(self, data_dict):
    #     nodes = data_dict['nodes']
    #     edges_start = data_dict['edges']['src']
    #     edge_end = data_dict['edges']['dst']
    #     node_embeddings = np.array(data_dict['node_embeddings'], dtype=np.float32)
    #     edge_embeddings = np.array(data_dict['edge_embeddings'], dtype=np.float32)
    #     g = dgl.DGLGraph()
    #     g.add_nodes(nodes, {'nbd': torch.Tensor(node_embeddings)})
    #     g.add_edges(edges_start, edge_end, {'ebd': torch.Tensor(edge_embeddings)})
    #     return g

    # def _load_from_gat_sample_file(self, gat_sampled_data):
    #     with tqdm(total=len(gat_sampled_data), desc=f"loading from file and converting data to graph type:\n") as pbar:
    #         for idx, item in enumerate(gat_sampled_data):
    #             claim = item['claim']
    #             claim_graph = self._create_graph(claim)
    #             candidates = item['candidates']
    #             for c in candidates:
    #                 c_graph = self._create_graph(c)
    #                 one_example = dict()
    #                 # one_example['selection_label'] = c['selection_label']
    #                 # one_example['selection_id'] = c['selection_id']
    #                 one_example['graph1'] = claim_graph
    #                 # one_example['graph1_dict'] = g_claim_dict
    #                 one_example['graph2'] = c_graph
    #                 # one_example['graph2_dict'] = g_c_dict
    #                 self.labels.append(c['lable'])
    #                 self.graph_instances.append(one_example)


if __name__ == '__main__':
    data = read_json_rows(config.RESULT_PATH / "sample_ss_graph.jsonl")[0:3]
    sample = DBpediaGATSampler(data)