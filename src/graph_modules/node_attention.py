from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim


class Node_Alignment(nn.Module):
    def __init__(self, args):
        super(Node_Alignment, self).__init__()
        self.args = args
        self.dropout = 0.5
        self.hidden_size = args.hidden_size
        self.embeds_dim = args.embeds_dim
        num_nodes = 20000
        self.embeds = nn.Embedding(num_nodes, self.embeds_dim)
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(args.linear_size, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(args.linear_size, 2),
            nn.Softmax(dim=-1)
        )

    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * node_len * dim
        x2: batch_size * node_len * dim
        '''
        # attention: batch_size * node_len * node_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * node_len * node_len
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
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


    def forward(self, *input):
        # g1, g2
        # batch_size * node_len
        nodes1_embeddings, nodes2_embeddings = input[0], input[1]
        mask1, mask2 = nodes1_embeddings.eq(0), nodes2_embeddings.eq(0)

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.bn_embeds(self.embeds(nodes1_embeddings).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds(nodes2_embeddings).transpose(1, 2).contiguous()).transpose(1, 2)

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        # o1, _ = self.lstm1(x1)
        # o2, _ = self.lstm1(x2)

        # batch_size * num_nodes * dim => batch_size * num_nodes * dim
        o1 = x1
        o2 = x2


        # Attention
        # batch_size * seq_len * hidden_size
        # batch_size * num_nodes * dim
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        # batch_size * num_nodes * (8 * dim)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)


        # batch_size * seq_len * (2 * hidden_size)
        # q1_compose, _ = self.lstm2(q1_combined)
        # q2_compose, _ = self.lstm2(q2_combined)

        # batch_size * num_nodes * (2 * dim)
        q1_compose, _ = self.GAT(q1_combined)
        q2_compose, _ = self.GAT(q2_combined)

        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity


def val_model(val_iter, net3):
    confusion_matrix = torch.zeros(2, 2)
    for labels, batch in enumerate(val_iter):
        predicted = net3(batch.query1, batch.query2)
        prediction = torch.max(predicted, 1)[1].data.numpy()
        label = batch.label
        for t, p in zip(prediction, label):
            confusion_matrix[t, p] += 1
    a_p = (confusion_matrix.diag() / confusion_matrix.sum(1))[0]
    print("被所有预测为负的样本中实际为负样本的概率", a_p)
    b_p = (confusion_matrix.diag() / confusion_matrix.sum(1))[1]
    print("被所有预测为正的样本中实际为正样本的概率", b_p)
    a_r = (confusion_matrix.diag() / confusion_matrix.sum(0))[0]
    print("实际的负样本中预测为负样本的概率，召回率", a_r)
    b_r = (confusion_matrix.diag() / confusion_matrix.sum(0))[1]
    print("实际的正样本中预测为正样本的概率，召回率", b_r)
    f1 = 2 * a_p * a_r / (a_p + a_r)
    f2 = 2 * b_p * b_r / (b_p + b_r)
    return f1, f2

#
# def main():
#     model = Node_Alignment()
#     model.train()
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
#     crition = F.cross_entropy
#     min_f = 0
#     for epoch in range(30):
#         epoch_loss = 0
#         for epoch, batch in enumerate(train_iter):
#             optimizer.zero_grad()
#             predicted = model(batch.query1, batch.query2)
#             loss = crition(predicted, batch.label)
#             loss.backward()
#             optimizer.step()
#             epoch_loss = epoch_loss + loss.data
#         # 计算每一个epoch的loss
#         # 计算验证集的准确度来确认是否存储这个model
#         print("epoch_loss", epoch_loss)
#         f1, f2 = val_model(val_iter, model)
#         if (f1 + f2) / 2 > min_f:
#             min_f = (f1 + f2) / 2
#             print("save model")
#             torch.save(model.state_dict(), '../data/esim_match_data/esim_params_30.pkl')
#
#
# if __name__ == '__main__':
#     main()
