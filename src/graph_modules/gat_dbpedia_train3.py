from torch.utils.data import DataLoader
# from graph_modules.DataLoaderX import DataLoaderX
from tqdm import tqdm
from pathlib import PosixPath
from datetime import datetime
from data_util.toChart import *
from graph_modules.gat_ss_dbpedia_sampler import DBpediaGATSampler
from graph_modules.gat_ss_dbpedia_sample_converter import DBpediaGATSampleConverter
from graph_modules.gat_ss_dbpedia_reader import DBpediaGATReader
from graph_modules.gat_ss_classifier2 import *
from utils.file_loader import read_json_rows, read_files_one_by_one, read_all_files
from memory_profiler import profile
import gc
from apex import amp
import torch.distributed as dist
from torch import distributed
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import argparse
import os


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
    local_rank = 0

def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.reduce_op.SUM)
    rt /= distributed.get_world_size()
    return rt

def train(paras: GAT_para, args):
    lr = paras.lr
    epoches = paras.epoches
    # epoches = 10
    dim = 768
    head = 4
    trainset = paras.data
    dt = paras.dt
    # Create training and test sets.

    is_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:1" if is_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"device: {device} n_gpu: {n_gpu}")


    model = GATClassifier2(dim, dim, head, 2)   # out: (4 heads + 1 edge feature) * 2 graphs
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if args.distributed:
        # FOR DISTRIBUTED:  Set the device according to local_rank.
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    torch.backends.cudnn.benchmark = True
    if is_cuda:
        # if n_gpu > 1:
        #     model = torch.nn.DataParallel(model)
        model = convert_syncbn_model(model).to(device)
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        model = DistributedDataParallel(model, delay_allreduce=True)
        # model.to(device)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.2)
        loss_func.to(device)


    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_data_loader = DataLoader(trainset, batch_size=paras.batch_size, shuffle=True, collate_fn=collate_with_dgl,
                                   pin_memory=True, num_workers=paras.data_num_workers, drop_last=True, sampler=train_sampler)

    model.train()
    epoch_losses = []
    writer = SummaryWriter(
        log_dir=f'writer_{datetime.datetime.now()}'
    )
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
                scheduler.step()
                reduced_loss = reduce_tensor(loss.data)
                epoch_loss += reduced_loss.item()
                pbar.update(1)
            epoch_loss /= (batch + 1)
            pbar.set_postfix_str('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)


            # TensorBoard
            if paras.local_rank == 0:
                writer.add_scalars('Loss/training', {
                    'train_loss': epoch_losses,
                }, epoch + 1)

                writer.add_scalars('Metrics/validation', {
                    'dice': dice,
                    'iou': iou
                }, epoch + 1)

                writer.add_images('Label/ground_truth', labels[:, :, 0, :, :], epoch + 1)
                if paras.local_rank == 0:
                    torch.save(model, f'unet3d-epoch{epoch + 1}.pth')
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
        model = GATClassifier2(dim, dim, 4, 2)
        model.load_state_dict(torch.load(model_or_path))
        if is_cuda:
            # if n_gpu > 1:
            #     model = torch.nn.DataParallel(model)
            model.to(device)
            loss_func.to(device)
    else:
        model = model_or_path
    testset = DBpediaGATSampler(dbpedia_data, parallel=True, num_worker=8)
    model.eval()
    # Convert a list of tuples to two lists
    test_data_loader = DataLoader(testset, batch_size=80, shuffle=True, collate_fn=collate_with_dgl,
                                  pin_memory=True, num_workers=8, drop_last=True)
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


def test_load_model():
    model_path = config.SAVED_MODELS_PATH / 'gat_ss_0.0001_epoch10_2020_08_27_20:51:50_89.346_94.953'
    data = read_files_one_by_one(config.RESULT_PATH / 'sample_ss_graph_dev_test')
    eval(model_path, data)


def read_data_in_file_batch():
    data_train = read_files_one_by_one(config.RESULT_PATH / "sample_ss_graph_train_test")
    data_dev = read_files_one_by_one(config.RESULT_PATH / "sample_ss_graph_dev_test")
    # print(f"train data len: {len(data_train)}; eval data len: {len(data_dev)}\n")
    return data_train, data_dev


# @profile
def train_and_eval():
    parser = argparse.ArgumentParser()
    # FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
    # automatically by torch.distributed.launch.
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    start = datetime.now()
    data_train, data_dev = read_data_in_file_batch()
    paras = GAT_para()
    paras.data = DBpediaGATSampler(data_train, parallel=True, num_worker=8)
    paras.epoches = 400
    paras.batch_size = 64
    paras.data_num_workers = 8
    model = train(paras, args)
    print(f"train time: {datetime.now() - start}")
    paras.data = []
    loss_eval_chart, accuracy_argmax, accuracy_sampled = eval(model, data_dev)
    # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = config.SAVED_MODELS_PATH / f"gat_ss_{paras.lr}_epoch{paras.epoches}_{paras.dt}_{accuracy_sampled:.3f}_{accuracy_argmax:.3f}"
    torch.save(model.state_dict(), output_model_file)
    print(f"total time: {datetime.now() - start}")

# @profile
def test_data():
    t, d = read_data_in_file_batch()
    trainset = DBpediaGATSampler(t, parallel=False)
    devset = DBpediaGATSampler(d, parallel=True)
    del trainset
    del devset
    return


if __name__ == '__main__':
    # test_load_model()
    train_and_eval()
    # concat_tmp_data()
    # test_data()
