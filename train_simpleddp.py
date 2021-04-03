import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from apex import amp
from apex.parallel import DistributedDataParallel

from feeder.feeders import Feeder
from net.st_gcn import Model

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--world_size', default=2, type=int,
                    help='number of gpus of all nodes')                    
parser.add_argument('--batch_size', default=32, type=int,
                    help='mini batch_size, this is the total')
parser.add_argument('--data', default='data/ucf101_skeleton/train_data.npy',
                    help='train data path')
parser.add_argument('--label', default='data/ucf101_skeleton/train_label.pkl',
                    help='train label path')  
parser.add_argument('--lr', default=0.01,
                    help='set of learning rate')  
parser.add_argument('--epoch', default=10,type=int,
                    help='number of train epochs')                                                                          
args = parser.parse_args()

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels) # 使用type_as(tesnor)将张量转换为给定类型的张量。
    correct = preds.eq(labels).double()  # 记录等于preds的label eq:equal
    correct = correct.sum()
    return correct / len(labels)

def train(rank, world_size, args):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    train_dataset = Feeder(args.data, args.label)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=(train_sampler is None),
                                                num_workers=2,
                                                pin_memory=True,
                                                sampler=train_sampler)

    model = Model(
        in_channels = 3,
        num_class = 80,
        # graph = 'graph.Graph',
        graph_args = {"layout":'openpose', "strategy":'spatial'},
        edge_importance_weighting = True
    ).cuda(rank)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    start_time = time.time()
    for epoch in range(args.epoch):
        for i, traindata in enumerate(train_loader):
            data, label = traindata

            # data = data.cuda(non_blocking=True)
            # label = label.cuda(non_blocking=True)
            data = data.cuda(rank)
            label = label.cuda(rank)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()
            optimizer.step()

            acc = accuracy(output, label)
            print("Train Phase, Epoch:{}/{}, Batch iter:{}/{}, loss:{}, acc:{}"
            .format(epoch, args.epoch, i, len(train_loader), loss, acc))

    end_time = time.time()
    print("Train Finished, Total spend time:{}".format(end_time - start_time))

if __name__ == '__main__':
    mp.spawn(
        train,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )