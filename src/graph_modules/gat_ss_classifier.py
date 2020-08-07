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
        q1_projection = self.projection(q1_combined)
        q2_projection = self.projection(q2_combined)
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
        self.node_alignment = Node_Alignment(768, 768)

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


def collate(samples):
    # The input `samples` is a list of pairs
    # random.shuffle(samples)
    graph_pairs, labels = map(list, zip(*samples))
    g1_l = [i['graph1'] for i in graph_pairs]
    g2_l = [i['graph2'] for i in graph_pairs]
    return dgl.batch(g1_l), dgl.batch(g2_l), torch.tensor(labels)


def train():
    # Create training and test sets.
    data_train = read_json_rows(config.RESULT_PATH / "sample_ss_graph.jsonl")[0:1500]
    data_dev = read_json_rows(config.RESULT_PATH / "sample_ss_graph.jsonl")[1500:1700]
    trainset = DBpediaGATSampler(data_train)
    testset = DBpediaGATSampler(data_dev)
    # Use PyTorch's DataLoader and the collate function
    # defined before.


    # Create model
    # in_dim, hidden_dim, num_heads, n_classes
    model = GATClassifier(768, 768, 4, trainset.num_classes)   # out: (4 heads + 1 edge feature) * 2 graphs
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {} n_gpu: {}".format(device, n_gpu))
    if device == "cuda":
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)

    train_data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                             collate_fn=collate)

    model.train()
    epoch_losses = []
    for epoch in range(20):
        epoch_loss = 0
        with tqdm(total=len(train_data_loader), desc=f"Epoch {epoch}") as pbar:
            for batch, graphs_and_labels in enumerate(train_data_loader):
                graph1_batched, graph2_batched, label = graphs_and_labels
                if device == "cuda":
                    for t in graphs_and_labels:
                        t.to(device)
                prediction = model(graph1_batched, graph2_batched)
                loss = loss_func(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
        epoch_loss /= (batch + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

    model.eval()
    # Convert a list of tuples to two lists
    test_data_loader = DataLoader(testset, batch_size=80, shuffle=True, collate_fn=collate)
    all_sampled_y_t = 0
    all_argmax_y_t = 0
    test_len = 0
    for graphs_and_labels in tqdm(test_data_loader):
        if device == "cuda":
            for t in graphs_and_labels:
                t.to(device)
        test_bg1, test_bg2, test_y = graphs_and_labels
        test_y = torch.tensor(test_y).float().view(-1, 1)
        probs_Y = torch.softmax(model(test_bg1, test_bg2), 1)
        sampled_Y = torch.multinomial(probs_Y, 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
        all_sampled_y_t = all_sampled_y_t + ((test_y == sampled_Y.float()).sum().item())
        all_argmax_y_t = all_argmax_y_t + ((test_y == argmax_Y.float()).sum().item())
        test_len = test_len + len(test_y)
    accuracy_sampled = all_sampled_y_t / test_len * 100
    accuracy_argmax = all_argmax_y_t / test_len * 100
    print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(all_sampled_y_t / test_len * 100))
    print('Accuracy of argmax predictions on the test set: {:4f}%'.format(all_argmax_y_t / test_len * 100))

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = config.SAVED_MODELS_PATH / f"gat_ss_{get_current_time_str()}_{accuracy_sampled}_{accuracy_argmax}"
    torch.save(model_to_save.state_dict(), output_model_file)

if __name__ == '__main__':
    train()