import dgl
import dgl.function as fn
from graph_modules.dbpedia_ss_gat_sampler import DBpediaGATSampler
from dgl.nn.pytorch import *
from torch.utils.data import DataLoader
import random
from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.file_loader import *


class Node_Alignment(nn.Module):
    def __init__(self):
        super(Node_Alignment, self).__init__()

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
        return q1_combined, q2_combined


class GATLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 alpha=0.2,
                 agg_activation=F.elu):
        super(GATLayer, self).__init__()

        self.num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        # self.attn_l = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        # self.attn_r = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_l = nn.Parameter(torch.randn(size=(num_heads, out_dim, 1)))
        self.attn_r = nn.Parameter(torch.randn(size=(num_heads, out_dim, 1)))
        self.attn_drop = nn.Dropout(attn_drop)
        self.activation = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        self.agg_activation=agg_activation

    def clean_data(self):
        ndata_names = ['ft', 'a1', 'a2']
        edata_names = ['a_drop']
        for name in ndata_names:
            self.g.ndata.pop(name)
        for name in edata_names:
            self.g.edata.pop(name)

    def forward(self, feat, bg):
        # prepare, inputs are of shape V x F, V the number of nodes, F the dim of input features
        self.g = bg
        h = self.feat_drop(feat)
        # V x K x F', K number of heads, F' dim of transformed features
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))
        head_ft = ft.transpose(0, 1)                              # K x V x F'
        a1 = torch.bmm(head_ft, self.attn_l).transpose(0, 1)      # V x K x 1
        a2 = torch.bmm(head_ft, self.attn_r).transpose(0, 1)      # V x K x 1
        self.g.ndata.update({'ft' : ft, 'a1' : a1, 'a2' : a2})
        # 1. compute edge attention
        self.g.apply_edges(self.edge_attention)
        # 2. compute softmax in two parts: exp(x - max(x)) and sum(exp(x - max(x)))
        self.edge_softmax()
        # 2. compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        self.g.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
        # 3. apply normalizer
        ret = self.g.ndata['ft']                                  # V x K x F'
        ret = ret.flatten(1)

        if self.agg_activation is not None:
            ret = self.agg_activation(ret)

        # Clean ndata and edata
        self.clean_data()

        return ret

    def edge_attention(self, edges):
        # an edge UDF to compute un-normalized attention values from src and dst
        a = self.activation(edges.src['a1'] + edges.dst['a2'])
        return {'a': a}

    def edge_softmax(self):
        attention = self.softmax(self.g, self.g.edata.pop('a'))
        # Dropout attention scores and save them
        self.g.edata['a_drop'] = self.attn_drop(attention)


class GATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes):
        super(GATClassifier, self).__init__()
        self.node_alignment = Node_Alignment()

        self.gat_layers = nn.ModuleList([
            GATLayer(in_dim * 4, hidden_dim, num_heads),    # aligned node embedding = in_dim * 4
            GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
        ])

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * (num_heads + 1) * 2),
            nn.Dropout(0),
            nn.Linear(hidden_dim * (num_heads + 1) * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0),
            nn.Linear(hidden_dim, n_classes))

    def forward(self, gp):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        g1, g2 = gp[0], gp[1]
        bg1 = dgl.batch(g1)
        bg2 = dgl.batch(g2)
        bg1_align, bg2_align = self.node_alignment(bg1.ndata['nbd'], bg2.ndata['nbd'])  # 768 * 4
        bg1.ndata.update({'nbd_align': bg1_align})
        bg2.ndata.update({'nbd_align': bg2_align})
        g1 = dgl.unbatch(bg1)
        g2 = dgl.unbatch(bg2)
        len1 = len(g1)
        len2 = len(g2)
        g1.extend(g2)
        bgp = dgl.batch(g1)
        h = bgp.ndata['nbd_align']    # aligned node embeddings: 768 * 4
        # h = bgp.in_degrees().view(-1, 1).float()
        for i, gnn in enumerate(self.gat_layers):
            h = gnn(h, bgp)

        edge_embeddings = bgp.edata['ebd']
        bgp.ndata['h'] = torch.cat([h, edge_embeddings], dim=1)    # 768 * 5
        hg_all = dgl.mean_nodes(bgp, 'h')
        hg1 = hg_all[0:len1]
        hg2 = hg_all[len1:len1+len2]
        x = torch.cat([hg1, hg2], dim=1)
        return self.classifier(x)


def collate(samples):
    # The input `samples` is a list of pairs
    # random.shuffle(samples)
    graph_pairs, labels = map(list, zip(*samples))
    g1_l = [i['graph1'] for i in graph_pairs]
    g2_l = [i['graph2'] for i in graph_pairs]
    return (g1_l, g2_l), torch.tensor(labels)


def train():
    # Create training and test sets.
    data_train = read_json_rows(config.RESULT_PATH / "sample_ss_graph.jsonl")
    data_dev = read_json_rows(config.RESULT_PATH / "sample_ss_graph.jsonl")
    trainset = DBpediaGATSampler(data_train)
    testset = DBpediaGATSampler(data_dev)
    # Use PyTorch's DataLoader and the collate function
    # defined before.
    data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                             collate_fn=collate)

    # Create model
    # in_dim, hidden_dim, num_heads, n_classes
    model = GATClassifier(768, 768, 4, trainset.num_classes)   # out: (4 heads + 1 edge feature) * 2 graphs
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    model.train()

    epoch_losses = []
    for epoch in range(80):
        epoch_loss = 0
        for ite, (bgp, label) in enumerate(data_loader):
            prediction = model(bgp)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (ite + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

    model.eval()
    # Convert a list of tuples to two lists
    test_data_loader = DataLoader(testset, batch_size=80, shuffle=True, collate_fn=collate)
    for it, (test_bg, test_Y) in enumerate(test_data_loader):
        test_Y = torch.tensor(test_Y).float().view(-1, 1)
        probs_Y = torch.softmax(model(test_bg), 1)
        sampled_Y = torch.multinomial(probs_Y, 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
        print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
            (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
        print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
            (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))


if __name__ == '__main__':
    train()