from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import PosixPath
from data_util.toChart import *
import dgl
import torch
from graph_modules.gat_ss_dbpedia_sampler import DBpediaGATSampler
from graph_modules.gat_ss_dbpedia_sample_converter import DBpediaGATSampleConverter
from graph_modules.gat_ss_dbpedia_reader import DBpediaGATReader
from graph_modules.gat_ss_classifier import *
from utils.file_loader import read_json_rows, read_files_one_by_one, read_all_files
from memory_profiler import profile
import gc


def collate_convert_to_dgl(instance_and_labels):
    # paired_g, labels = map(list, zip(*instance_and_labels))
    converter = DBpediaGATSampleConverter()
    dglgraph_pairs, labels = converter.convert_dbpedia_to_dgl(instance_and_labels, parallel=False)
    g1_l = [i['graph1'] for i in dglgraph_pairs]
    g2_l = [i['graph2'] for i in dglgraph_pairs]
    return dgl.batch(g1_l), dgl.batch(g2_l), torch.tensor(labels)


class GAT_para(object):
    data: DBpediaGATReader = None
    lr = 1e-4
    epoches = 400
    dt = get_current_time_str()
    batch_size = 64
    data_num_workers =16



def train(paras:GAT_para):
    lr = paras.lr
    epoches = paras.epoches
    # epoches = 10
    dim = 768
    head = 4
    trainset = paras.data
    dt = paras.dt
    train_data = paras.data

    model = GATClassifier(dim, dim, head, 2)   # out: (4 heads + 1 edge feature) * 2 graphs
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    is_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:1" if is_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"device: {device} n_gpu: {n_gpu}")
    if is_cuda:
        # if n_gpu > 1:
        #     model = torch.nn.DataParallel(model)
        model.to(device)
        loss_func.to(device)

    train_data_loader = DataLoader(train_data, batch_size=paras.batch_size, shuffle=True, collate_fn=collate_convert_to_dgl,
                                   num_workers=paras.data_num_workers, pin_memory=True, drop_last=True)
    model.train()
    epoch_losses = []
    for epoch in range(epoches):
        epoch_loss = 0
        batch = 0
        with tqdm(total=len(train_data_loader), desc=f"Epoch {epoch}") as pbar:
            for batch, graphs_and_labels in enumerate(train_data_loader):
                if is_cuda:
                    graphs_and_labels = tuple(t.to(device) for t in graphs_and_labels)
                    graph1_batched, graph2_batched, label = graphs_and_labels
                else:
                    graph1_batched, graph2_batched, label = graphs_and_labels
                prediction = model(graph1_batched, graph2_batched)
                loss = loss_func(prediction, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
                pbar.update(1)
            epoch_loss /= batch + 1
            pbar.set_postfix_str('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)

    draw_loss_epoches(epoch_losses, f"gat_ss_train_loss_{lr}_epoch{epoches}_{dt}.png")
    return model


def eval(model_or_path, dbpedia_data):
    loss_func = nn.CrossEntropyLoss()
    is_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:1" if is_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    dim = 768
    print(f"device: {device} n_gpu: {n_gpu}")
    if isinstance(model_or_path, PosixPath):
        model = GATClassifier(dim, dim, 4, 2)
        model.load_state_dict(torch.load(model_or_path))
        if is_cuda:
            # if n_gpu > 1:
            #     model = torch.nn.DataParallel(model)
            model.to(device)
            loss_func.to(device)
    else:
        model = model_or_path

    model.eval()
    # Convert a list of tuples to two lists
    test_data_loader = DataLoader(dbpedia_data, batch_size=160, shuffle=True, num_workers=8,
                                  collate_fn=collate_convert_to_dgl, drop_last=True)
    all_sampled_y_t = 0
    all_argmax_y_t = 0
    test_len = 0
    loss_eval_chart = []
    loss_eval = 0
    nb_eval_steps = 0
    for graphs_and_labels in tqdm(test_data_loader):
        if is_cuda:
            batch = tuple(t.to(device) for t in graphs_and_labels)
            test_bg1, test_bg2, test_y = batch
        else:
            test_bg1, test_bg2, test_y = graphs_and_labels

        pred_y = model(test_bg1, test_bg2)

        tmp_eval_loss = loss_func(pred_y, test_y).detach().item()
        loss_eval_chart.append(tmp_eval_loss)
        loss_eval += tmp_eval_loss
        nb_eval_steps += 1

        test_y = test_y.clone().detach().float().view(-1, 1)
        probs_Y = torch.softmax(pred_y, 1)
        sampled_Y = torch.multinomial(probs_Y, 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
        all_sampled_y_t += (test_y == sampled_Y.float()).sum().item()
        all_argmax_y_t += (test_y == argmax_Y.float()).sum().item()
        test_len = test_len + len(test_y)

    accuracy_sampled = all_sampled_y_t / test_len * 100
    accuracy_argmax = all_argmax_y_t / test_len * 100
    loss_eval = loss_eval / nb_eval_steps
    print(f"eval loss: {loss_eval}")
    print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(accuracy_sampled))
    print('Accuracy of argmax predictions on the test set: {:4f}%'.format(accuracy_argmax))
    return loss_eval_chart, accuracy_argmax, accuracy_sampled


def train_and_eval():
    train_data_path = config.RESULT_PATH / "sample_ss_graph_train_test"
    dev_data_path = config.RESULT_PATH / "sample_ss_graph_dev_test"
    paras = GAT_para()
    paras.data = DBpediaGATReader(train_data_path)
    paras.epoches = 40
    paras.batch_size = 64
    paras.data_num_workers = 8
    model = train(paras)
    paras.data = []
    loss_eval_chart, accuracy_argmax, accuracy_sampled = eval(model, DBpediaGATReader(dev_data_path))
    # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = config.SAVED_MODELS_PATH / f"gat_ss_{paras.lr}_epoch{paras.epoches}_{paras.dt}_{accuracy_sampled:.3f}_{accuracy_argmax:.3f}"
    torch.save(model.state_dict(), output_model_file)


def test_load_model():
    model_path = config.SAVED_MODELS_PATH / 'gat_ss_0.0001_epoch10_2020_08_26_20:15:12_50.000_100.000'
    data = read_json_rows(config.RESULT_PATH / 'sample_ss_graph_train' / 'sample_ss_graph_ThreadPoolExecutor-0_0_2020_08_10_09:59:55.jsonl')
    eval(model_path, data)


if __name__ == '__main__':
    # test_load_model()
    # train()
    # concat_tmp_data()
    # test_data()
    train_and_eval()