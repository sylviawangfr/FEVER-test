import dgl
import dgl.function as fn
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch import *
from torch import nn



class Node_Alignment(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Node_Alignment, self).__init__()
        self.projection = nn.Linear(in_dim * 4, out_dim, bias=False)

    def soft_attention_align(self, x1, x2):
        '''
        x1: batch_size * node_len * dim
        x2: batch_size * node_len * dim
        '''
        # attention: batch_size * node_len * node_len
        attention = torch.matmul(x1, x2.transpose(0, 1))

        # weight: batch_size * node_len * node_len
        weight1 = F.softmax(attention, dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(0, 1), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * node_len * hidden_size
        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * node_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, nodes_embeddings1, nodes_embeddings2):
        # g1, g2
        # batch_size * node_len
        o1 = nodes_embeddings1
        o2 = nodes_embeddings2

        # Attention
        # batch_size * seq_len * hidden_size
        # batch_size * num_nodes * dim
        q1_align, q2_align = self.soft_attention_align(o1, o2)

        # Compose
        # batch_size * num_nodes * (4 * dim)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)
        q1_projection = F.relu(self.projection(q1_combined))
        q2_projection = F.relu(self.projection(q2_combined))
        return q1_projection, q2_projection


class GATLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 alpha=0.2,
                 agg_activation=F.elu):
        super(GATLayer, self).__init__()

        self.num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc = nn.Linear(in_dim, num_heads * hidden_dim, bias=False)
        self.attn_l = nn.Parameter(torch.randn(size=(num_heads, hidden_dim, 1)))
        self.attn_r = nn.Parameter(torch.randn(size=(num_heads, hidden_dim, 1)))
        self.attn_e = nn.Parameter(torch.randn(size=(num_heads, hidden_dim, 1)))
        self.attn_drop = nn.Dropout(attn_drop)
        self.activation = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        self.agg_activation=agg_activation

    def clean_data(self):
        ndata_names = ['ft', 'a1', 'a2']
        edata_names = ['a_drop', 'w_n_ft', 'e_ft', 'f_cat', 'a3']
        for name in ndata_names:
            self.g.ndata.pop(name)
        for name in edata_names:
            self.g.edata.pop(name)

    def forward(self, node_feature, edge_feature, bg):
        # prepare, inputs are of shape V x F, V the number of nodes, F the dim of input features
        self.g = bg
        h = self.feat_drop(node_feature)
        e_h = self.feat_drop(edge_feature)
        # V x K x F', K number of heads, F' dim of transformed features
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))
        e_ft = self.fc(e_h).reshape((e_h.shape[0], self.num_heads, -1))
        head_ft = ft.transpose(0, 1)                              # K x V x F'
        e_head_ft = e_ft.transpose(0, 1)
        a1 = torch.bmm(head_ft, self.attn_l).transpose(0, 1)      # V x K x 1
        a2 = torch.bmm(head_ft, self.attn_r).transpose(0, 1)      # V x K x 1
        a3 = torch.bmm(e_head_ft, self.attn_e).transpose(0, 1)      # V x K x 1
        self.g.ndata.update({'ft': ft, 'a1': a1, 'a2': a2})
        self.g.edata.update({'e_ft': e_ft, 'a3': a3})
        # 1. compute edge attention
        self.g.apply_edges(self.edge_attention)
        # 2. compute softmax in two parts: exp(x - max(x)) and sum(exp(x - max(x)))
        self.edge_softmax()
        self.g.apply_edges(fn.src_mul_edge('ft', 'a_drop', 'w_n_ft'))
        self.edge_concat()
        # compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        # self.g.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
        self.g.update_all(fn.copy_edge('f_cat', 'tmp_ft'), fn.sum('tmp_ft', 'ft'))
        # 3. apply normalizer
        ret1 = self.g.ndata['ft']                                  # V x K x F'
        ret2 = self.g.edata['f_cat']
        ret1 = ret1.flatten(1)
        ret2 = ret2.flatten(1)

        if self.agg_activation is not None:
            ret1 = self.agg_activation(ret1)
            ret2 = self.agg_activation(ret2)

        # Clean ndata and edata
        self.clean_data()

        return ret1, ret2

    def edge_attention(self, edges):
        # an edge UDF to compute un-normalized attention values from src and dst
        a = self.activation(edges.src['a1'] + edges.data['a3'] + edges.dst['a2'])
        return {'a': a}

    def edge_softmax(self):
        attention = self.softmax(self.g, self.g.edata.pop('a'))
        # Dropout attention scores and save them
        self.g.edata['a_drop'] = self.attn_drop(attention)

    def edge_concat(self):
        weighted_ebd = torch.mul(self.g.edata['e_ft'], self.g.edata['a_drop']) + self.g.edata['w_n_ft']
        self.g.edata['f_cat'] = weighted_ebd


class GATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes):
        super(GATClassifier, self).__init__()
        self.node_alignment = Node_Alignment(in_dim, hidden_dim)

        self.gat_layers = nn.ModuleList([
            GATLayer(in_dim, hidden_dim, num_heads),    # aligned node embedding = in_dim
            GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
        ])

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * num_heads * 2),
            nn.Dropout(0),
            nn.Linear(hidden_dim * num_heads * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0),
            nn.Linear(hidden_dim, n_classes))

    def forward(self, graph1_batched, graph2_batched):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        bg1 = graph1_batched
        bg2 = graph2_batched
        bg1_align, bg2_align = self.node_alignment(bg1.ndata['nbd'], bg2.ndata['nbd'])  # 768
        bg1.ndata.update({'nbd_align': bg1_align})
        bg2.ndata.update({'nbd_align': bg2_align})
        g1 = dgl.unbatch(bg1)
        g2 = dgl.unbatch(bg2)
        len1 = len(g1)
        len2 = len(g2)
        g1.extend(g2)
        bgp = dgl.batch(g1)
        h_n = bgp.ndata['nbd_align']    # aligned node embeddings: 768
        h_e = bgp.edata['ebd']  # 768
        # h = bgp.in_degrees().view(-1, 1).float()
        for i, gnn in enumerate(self.gat_layers):
            h_n, h_e = gnn(h_n, h_e, bgp)

        bgp.ndata['h'] = h_n
        hg_all = dgl.mean_nodes(bgp, 'h')
        hg1 = hg_all[0:len1]
        hg2 = hg_all[len1:len1+len2]
        x = torch.cat([hg1, hg2], dim=1)
        return self.classifier(x)




