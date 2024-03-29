from torch.utils.data import DataLoader
from graph_modules.DataLoaderX import DataLoaderX
from tqdm import tqdm
import numpy as np
from pathlib import PosixPath
import torch.nn.functional as F
from datetime import datetime
from data_util.toChart import *
import dgl
import torch
from graph_modules.gat_ss_dbpedia_sampler import DBpediaGATSampler
from graph_modules.gat_ss_dbpedia_sample_converter import DBpediaGATSampleConverter
from graph_modules.gat_ss_dbpedia_reader import DBpediaGATReader
from graph_modules.gat_ss_classifier2 import *
from utils.file_loader import read_json_rows, read_files_one_by_one, save_intermidiate_results, read_all_files
from memory_profiler import profile
import gc
import utils.common_types as model_para
from BERT_test.ss_eval import ss_f1_score_and_save



def collate_with_dgl(dgl_samples):
    # The input `samples` is a list of pairs
    # random.shuffle(samples)
    graph_pairs, labels = map(list, zip(*dgl_samples))
    g1_l = [i['graph1'] for i in graph_pairs]
    g2_l = [i['graph2'] for i in graph_pairs]
    return dgl.batch(g1_l), dgl.batch(g2_l), torch.tensor(labels)


class GAT_para(object):
    data = None
    lr = 1e-4
    epoches = 400
    dt = get_current_time_str()
    batch_size = 64
    data_num_workers = 16
    gpu_num = 0

def train(paras: GAT_para):
    lr = paras.lr
    epoches = paras.epoches
    # epoches = 10
    dim = 768
    head = 4
    trainset = paras.data
    dt = paras.dt
    # Create training and test sets.

    torch.backends.cudnn.benchmark = True
    model = GATClassifier2(dim, dim, head, 2)   # out: (4 heads + 1 edge feature) * 2 graphs
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    is_cuda = True if torch.cuda.is_available() else False
    device = torch.device(f"cuda:{paras.gpu_num}" if is_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"device: {device} n_gpu: {n_gpu}")
    if is_cuda:
        # if n_gpu > 1:
        #     model = torch.nn.DataParallel(model)
        model.to(device)
        loss_func.to(device)

    train_data_loader = DataLoaderX(trainset, batch_size=paras.batch_size, shuffle=True, collate_fn=collate_with_dgl,
                                   pin_memory=True, num_workers=paras.data_num_workers, drop_last=True)

    model.train()
    epoch_losses = []
    for epoch in range(epoches):
        epoch_loss = 0
        with tqdm(total=len(train_data_loader), desc=f"Epoch {epoch}") as pbar:
            for batch, graphs_and_labels in enumerate(train_data_loader):
                # graph1_batched, graph2_batched, label = graphs_and_labels
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
            epoch_loss /= (batch + 1)
            pbar.set_postfix_str('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)
    draw_loss_epoches(epoch_losses, f"gat_ss_train_loss_{lr}_epoch{epoches}_{dt}.png")
    return model


def eval(model_or_path, dbpedia_data, gpu):
    loss_func = nn.CrossEntropyLoss()
    is_cuda = True if torch.cuda.is_available() else False
    device = torch.device(f"cuda:{gpu}" if is_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    dim = 768
    print(f"device: {device} n_gpu: {n_gpu}")
    if isinstance(model_or_path, PosixPath):
        model = GATClassifier2(dim, dim, 4, 2)
        if is_cuda:
            # if n_gpu > 1:
            #     model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(model_or_path, map_location=f"cuda:{gpu}"))
            model.to(device)
            loss_func.to(device)
        else:
            model.load_state_dict(torch.load(model_or_path, map_location="cpu"))
    else:
        model = model_or_path
    testset = DBpediaGATSampler(dbpedia_data, parallel=True, num_worker=8)
    model.eval()
    # Convert a list of tuples to two lists
    test_data_loader = DataLoaderX(testset, batch_size=80, shuffle=True, collate_fn=collate_with_dgl,
                                  pin_memory=True, num_workers=0, drop_last=True)
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


def pred_prob(model_or_path, original_data, dbpedia_data, output_dir, gpu=0, thredhold=0.4, data_from_pred=True, eval=True):
    loss_func = nn.CrossEntropyLoss()
    is_cuda = True if torch.cuda.is_available() else False
    device = torch.device(f"cuda:{gpu}" if is_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    dim = 768
    print(f"device: {device} n_gpu: {n_gpu}")
    if isinstance(model_or_path, PosixPath):
        model = GATClassifier2(dim, dim, 4, 2)
        if is_cuda:
            model.load_state_dict(torch.load(model_or_path, map_location=f"cuda:{gpu}"))
            model.to(device)
            loss_func.to(device)
        else:
            model_or_path = open(model_or_path, 'rb')
            model.load_state_dict(torch.load(model_or_path, map_location=torch.device('cpu')))
    else:
        model = model_or_path
    testset = DBpediaGATSampler(dbpedia_data, parallel=True, num_worker=8, data_from_pred=data_from_pred)
    print(f"done with sampling, data count: {testset.__len__()}, failed count: {testset.failed_count}")
    model.eval()
    # Convert a list of tuples to two lists
    test_data_loader = DataLoader(testset, batch_size=80, shuffle=False, collate_fn=collate_with_dgl,
                                  pin_memory=True, num_workers=0)
    preds = []
    for graphs_and_labels in tqdm(test_data_loader):
        if is_cuda:
            batch = tuple(t.to(device) for t in graphs_and_labels)
            test_bg1, test_bg2, test_y = batch
        else:
            test_bg1, test_bg2, test_y = graphs_and_labels

        pred_y = model(test_bg1, test_bg2)

        # tmp_eval_loss = loss_func(pred_y, test_y).detach().item()
        # loss_eval_chart.append(tmp_eval_loss)
        # loss_eval += tmp_eval_loss
        # nb_eval_steps += 1

        # test_y = test_y.clone().detach().float().view(-1, 1)
        if len(preds) == 0:
            preds = pred_y.detach().cpu()
        else:
            preds = torch.cat((preds, pred_y.detach().cpu()), dim=0)
    probs = torch.softmax(preds, 1)
    probs = probs[:, 1].tolist()
    scores = preds[:, 1].tolist()
    preds = np.argmax(preds, axis=1)
    dbpedia_data_d = dict()
    for example in dbpedia_data:
        for candidate in example['examples']:
            dbpedia_data_d[candidate['selection_id']] = candidate
    graph_instances = testset.graph_instances

    print(f"graph_instance len: {len(graph_instances)}")
    print(f"dbpeida_data len: {len(dbpedia_data_d)}")
    print(f"testset len: {len(testset)}")
    for i in range(len(testset)):
        selection_id = graph_instances[i]['selection_id']
        if selection_id not in dbpedia_data_d:
            print(selection_id)

    for i in range(len(testset)):
        selection_id = graph_instances[i]['selection_id']
        dbpedia_data_d[selection_id]['score'] = scores[i]
        dbpedia_data_d[selection_id]['prob'] = probs[i]
    dict_to_list = list(dbpedia_data_d.values())
    # save_intermidiate_results(dict_to_list, output_file)
    paras = model_para.PipelineParas()
    paras.output_folder = output_dir
    paras.original_data = original_data
    paras.data_from_pred = data_from_pred
    paras.top_n = [10]
    paras.prob_thresholds = thredhold
    if eval:
        paras.mode = 'eval'
    else:
        paras.mode = 'test'
    ss_f1_score_and_save(paras, dict_to_list)
    return dict_to_list


def test_load_model():
    model_path = config.SAVED_MODELS_PATH / 'gat_ss_0.0001_epoch400_65.856_66.430'
    data = read_json_rows(config.RESULT_PATH / 'sample_ss_graph.jsonl')
    eval(model_path, data, gpu=0)


def read_data_in_file_batch():
    data_train = read_files_one_by_one(config.RESULT_PATH / "sample_ss_graph_train_test")
    data_dev = read_files_one_by_one(config.RESULT_PATH / "sample_ss_graph_dev_test")
    # print(f"train data len: {len(data_train)}; eval data len: {len(data_dev)}\n")
    return data_train, data_dev


# @profile
def train_and_eval():
    start = datetime.now()
    data_train, data_dev = read_data_in_file_batch()
    paras = GAT_para()
    paras.data = DBpediaGATSampler(data_train, parallel=True, num_worker=16)
    paras.epoches = 400
    paras.batch_size = 64
    paras.data_num_workers = 8
    paras.gpu_num = 0
    model = train(paras)
    print(f"train time: {datetime.now() - start}")
    paras.data = []
    loss_eval_chart, accuracy_argmax, accuracy_sampled = eval(model, data_dev, paras.gpu_num)
    # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = config.SAVED_MODELS_PATH / f"gat_ss_{paras.lr}_epoch{paras.epoches}_{paras.dt}_{accuracy_sampled:.3f}_{accuracy_argmax:.3f}"
    torch.save(model.state_dict(), output_model_file)
    print(f"total time: {datetime.now() - start}")

# @profile
def test_data():
    t, d = read_data_in_file_batch()
    trainset = DBpediaGATSampler(t, parallel=False)
    devset = DBpediaGATSampler(d, parallel=True)
    dataloader = DataLoader(devset, batch_size=2, shuffle=False, collate_fn=collate_with_dgl,
                                  pin_memory=True, num_workers=2)
    for i in dataloader:
        print(len(i))
    del trainset
    del devset
    return


if __name__ == '__main__':
    # data_dev = read_all_files(config.RESULT_PATH / "sample_ss_graph_train_test")

    # local cpu
    data_dev = read_json_rows(config.RESULT_PATH / 'sample_ss_graph_test_pred' / '1_2020_09_23_09:39:06.jsonl')[0:5]
    original_data = read_json_rows(config.FEVER_TEST_JSONL)
    model_path = config.SAVED_MODELS_PATH / 'gat_ss_0.0001_epoch10_2020_08_26'
    # model_path = config.SAVED_MODELS_PATH / 'gat_ss_0.0001_epoch400_65.856_66.430'
    output_dir = config.RESULT_PATH / 'test000'
    pred_prob(model_path, original_data, data_dev, output_dir, thredhold=0.1, pred=True, gpu=2, eval=False)

    # test set
    # data_dev = read_all_files(config.RESULT_PATH / 'sample_ss_graph_test_pred')
    # original_data = read_json_rows(config.FEVER_TEST_JSONL)
    # model_path = config.SAVED_MODELS_PATH / 'gat_ss_0.0001_epoch400_65.856_66.430'
    # output_dir = config.RESULT_PATH / 'sample_ss_graph_test_pred'
    # pred_prob(model_path, original_data, data_dev, output_dir, thredhold=0.1, pred=True, eval=False, gpu=2)

    # dev set
    # data_dev = read_all_files(config.RESULT_PATH / 'sample_ss_graph_dev_pred')
    # original_data = read_json_rows(config.FEVER_DEV_JSONL)
    # model_path = config.SAVED_MODELS_PATH / 'gat_ss_0.0001_epoch400_65.856_66.430'
    # output_dir = config.RESULT_PATH / 'sample_ss_graph_dev_pred' / 'gat_ss_10.jsonl'
    # pred_prob(model_path, original_data, data_dev, output_dir, thredhold=0.1, pred=True, eval=True, gpu=0)