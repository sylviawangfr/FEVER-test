"""A mini synthetic dataset for graph classification benchmark."""
import math
import networkx as nx
import numpy as np
from dgl import DGLGraph
from typing import Dict
import random

__all__ = ['MiniGPCDataset']

class MiniGPCDataset(object):
    """The dataset class.

    The datset contains 8 different types of graphs.

    * class 0 : cycle graph
    * class 1 : star graph
    * class 2 : wheel graph
    * class 3 : lollipop graph
    * class 4 : hypercube graph
    * class 5 : grid graph
    * class 6 : clique graph
    * class 7 : circular ladder graph

    .. note::
        This dataset class is compatible with pytorch's :class:`Dataset` class.

    Parameters
    ----------
    num_graph_pairs: int
        Number of graph pairs in this dataset.
    min_num_v: int
        Minimum number of nodes for graphs
    max_num_v: int
        Maximum number of nodes for graphs
    """
    def __init__(self, num_graph_pairs, min_num_v, max_num_v):
        super(MiniGPCDataset, self).__init__()
        self.num_graph_pairs = num_graph_pairs
        self.min_num_v = min_num_v
        self.max_num_v = max_num_v
        self.graph_pairs = []
        self.labels = []
        self._generate()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_pairs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graph_pairs[idx], self.labels[idx]

    @property
    def num_classes(self):
        """Number of classes."""
        return 2

    def _generate(self):
        gs = []
        gs.append(self._gen_cycle(self.num_graph_pairs // 8))
        gs.append(self._gen_star(self.num_graph_pairs // 8))
        gs.append(self._gen_wheel(self.num_graph_pairs // 8))
        gs.append(self._gen_lollipop(self.num_graph_pairs // 8))
        gs.append(self._gen_hypercube(self.num_graph_pairs // 8))
        gs.append(self._gen_grid(self.num_graph_pairs // 8))
        gs.append(self._gen_clique(self.num_graph_pairs // 8))
        gs.append(self._gen_circular_ladder(self.num_graph_pairs - len(self.graph_pairs)))
        # preprocess
        gs_c = []
        for gl in gs:
            gl_c = []
            for g in gl:
                g_c = DGLGraph(g)
                nodes = g_c.nodes()
                g_c.add_edges(nodes, nodes)
                gl_c.append(g_c)
            gs_c.append(gl_c)

        i = 0
        while i < self.num_graph_pairs:
            g_indx = random.sample(range(0, self.num_graph_pairs // 8), 2)
            g_class = np.random.randint(0, 8)
            self.graph_pairs.append({'g1': gs_c[g_class][g_indx[0]], 'g2': gs_c[g_class][g_indx[1]]})
            self.labels.append(1)
            i = i+1
            diff_class = random.sample(range(0, 8), 2)
            self.graph_pairs.append({'g1': gs_c[diff_class[0]][g_indx[0]], 'g2': gs_c[diff_class[1]][g_indx[1]]})
            self.labels.append(0)
            i = i + 1


    def _gen_cycle(self, n):
        cycle = []
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.cycle_graph(num_v)
            cycle.append(g)
        return cycle


    def _gen_star(self, n):
        gs = []
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            # nx.star_graph(N) gives a star graph with N+1 nodes
            g = nx.star_graph(num_v - 1)
            gs.append(g)
        return gs

    def _gen_wheel(self, n):
        gs = []
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.wheel_graph(num_v)
            gs.append(g)
        return gs

    def _gen_lollipop(self, n):
        gs = []
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            path_len = np.random.randint(2, num_v // 2)
            g = nx.lollipop_graph(m=num_v - path_len, n=path_len)
            gs.append(g)
        return gs

    def _gen_hypercube(self, n):
        gs = []
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.hypercube_graph(int(math.log(num_v, 2)))
            g = nx.convert_node_labels_to_integers(g)
            gs.append(g)
        return gs

    def _gen_grid(self, n):
        gs = []
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            assert num_v >= 4, 'We require a grid graph to contain at least two ' \
                                   'rows and two columns, thus 4 nodes, got {:d} ' \
                                   'nodes'.format(num_v)
            n_rows = np.random.randint(2, num_v // 2)
            n_cols = num_v // n_rows
            g = nx.grid_graph([n_rows, n_cols])
            g = nx.convert_node_labels_to_integers(g)
            gs.append(g)
        return gs

    def _gen_clique(self, n):
        gs = []
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.complete_graph(num_v)
            gs.append(g)
        return gs

    def _gen_circular_ladder(self, n):
        gs = []
        for _ in range(n):
            num_v = np.random.randint(self.min_num_v, self.max_num_v)
            g = nx.circular_ladder_graph(num_v // 2)
            gs.append(g)
        return gs
