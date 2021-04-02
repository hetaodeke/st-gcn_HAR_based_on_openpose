# from __future__ import division
# from __future__ import print_function
import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
from tqdm import tqdm

import time
import argparse  # argparse 是python自带的命令行参数解析包，可以用来方便地读取命令行参数
import numpy as np
import pickle
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import  torch.distributed as dist

# visualization
import matplotlib.pyplot as plt
import pylab as pl
from torch.utils.tensorboard import SummaryWriter

from net.st_gcn import Model
# from net.agcn import Model
from utils import *
from feeder.feeders import Feeder

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=80,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument("--local_rank", type=int, default=-1)                   
args = parser.parse_args()

if not torch.cuda.is_available() :
    raise EnvironmentError("not find GPU device for training.")

init_distributed_mode(args)

rank = args.rank
device = torch.device(args.device)
batch_size = args.batch_size
num_classes = args.num_classes
weights_path = args.weights
args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增

if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:7500/')
    tb_writer = SummaryWriter(log_dir='runs', comment='ddp-training')
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

# 加载数据
train_data_path = 'data/HMDB_51/train_data.npy'
train_label_path = 'data/HMDB_51/train_label.pkl'
train_data_set = Feeder(train_data_path, train_label_path)
val_data_path = 'data/HMDB_51/val_data.npy'
val_label_path = 'data/HMDB_51/val_label.pkl'
val_data_set = Feeder(val_data_path, val_label_path)



# 给每个rank对应的进程分配训练的样本索引
train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

# 将样本索引每batch_size个元素组成一个list
train_batch_sampler = torch.utils.data.BatchSampler(
train_sampler, batch_size, drop_last=True)

nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
if rank == 0:
    print('Using {} dataloader workers every process'.format(nw))
train_loader = torch.utils.data.DataLoader(train_data_set,
                                            batch_sampler=train_batch_sampler,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=train_data_set.collate_fn)

val_loader = torch.utils.data.DataLoader(val_data_set,
                                        batch_size=batch_size,
                                        sampler=val_sampler,
                                        pin_memory=True,
                                        num_workers=nw,
                                        collate_fn=val_data_set.collate_fn)
# 实例化模型
model = Model(
    in_channels = 3,
    num_class = 51,
    # graph = 'graph.Graph',
    graph_args = {"layout":'openpose', "strategy":'spatial'},
    edge_importance_weighting = True
).to(device)


# 如果存在预训练权重则载入
if os.path.exists(weights_path):
    weights_dict = torch.load(weights_path, map_location=device)
    load_weights_dict = {k: v for k, v in weights_dict.items()
                        if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(load_weights_dict, strict=False)
else:
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
    if rank == 0:
        torch.save(model.state_dict(), checkpoint_path)

    dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
if args.freeze_layers:
    for name, para in model.named_parameters():
        # 除最后的全连接层外，其他权重全部冻结
        if "fc" not in name:
            para.requires_grad_(False)
else:
    # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
    if args.syncBN:
        # 使用SyncBatchNorm后训练会更耗时
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

# 转为DDP模型
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

# if args.cuda:
#     # 为CPU设置种子用于生成随机数，以使得结果是确定的
#     set_seed(args.seed)
#     # load data
#     train_data_path = 'data/HMDB_51/train_data.npy'
#     train_label_path = 'data/HMDB_51/train_label.pkl'
#     train_dataloader = ngpu_load_data(train_data_path, train_label_path)
#     test_data_path = 'data/HMDB_51/val_data.npy'
#     test_label_path = 'data/HMDB_51/val_label.pkl'
#     test_dataloader = ngpu_load_data(test_data_path, test_label_path)
# else:
#     # 为CPU设置种子用于生成随机数，以使得结果是确定的
#     np.random.seed(args.seed)
#     # load data
#     train_data_path = 'data/HMDB_51/train_data.npy'
#     train_label_path = 'data/HMDB_51/train_label.pkl'
#     train_dataloader = cpu_load_data(train_data_path, train_label_path)
#     test_data_path = 'data/HMDB_51/val_data.npy'
#     test_label_path = 'data/HMDB_51/val_label.pkl'
#     test_dataloader = cpu_load_data(test_data_path, test_label_path)


# Model and optimizer
# model = Model(
#     in_channels = 3,
#     num_class = 51,
#     # graph = 'graph.Graph',
#     graph_args = {"layout":'openpose', "strategy":'spatial'},
#     edge_importance_weighting = True
# )
# model.apply(weights_init)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
#
# # 只 master 进程做 logging，否则输出会很乱
# if not os.path.exists('runs'):
#     os.makedirs('runs')
# if args.local_rank == 0:
#     tb_writer = SummaryWriter(log_dir='runs', comment='ddp-training')



t_total = time.time()

# train phase
# loss_history = {'epoch':[], 'loss_train':[]}
# os.chdir('/home/andy/GitHub项目/st-gcn_HAR_based_on_openpose')
# with open('NTU-train.txt', 'w+') as f:
for epoch in range(args.epochs):
    t = time.time()  # 返回当前时间
    model.train()
    loss = []
    result = []
    label = []
    mean_loss = torch.zeros(1).to(device)
    if rank == 0:
        train_loader = tqdm(train_loader)
    #load data
    for i, dataload in enumerate(train_loader):
        train_data, train_label = dataload
        # if args.cuda:
        #     train_data.to(device)
        #     train_label.to(device)
        optimizer.zero_grad()

    # forward

        output = model(train_data.to(device))
        loss_train = F.cross_entropy(output, train_label.long().to(device))

        
        adjust_lr(optimizer, epoch, args.lr)
        lr = optimizer.param_groups[0]['lr']

        if rank == 0:
            mean_loss = (mean_loss * i + loss_train.detach()) / (i + 1)  # update mean losses
    # backward       
        loss_train.backward()  # 反向求导  Back Propagation
        optimizer.step()  # 更新所有的参数  Gradient Descent

        if device != torch.device("cpu"):
            torch.cuda.synchronize()


        loss.append(loss_train.item())
        if i % 100 == 0:
            history = 'Epoch [{}/{}], '.format(epoch + 1, args.epochs) +\
                'Loss: {:.4f}, '.format(loss_train.item()) + \
                'time: {:.4f}s'.format(time.time() - t)
            print(history)   
            with open('work_result/log.txt', 'r+') as f:
                f.write(history + '\n') 

    # eval phase
    model.eval()
    total_acc = []
    with torch.no_grad():
        for i, dataload in enumerate(val_loader):
            test_data, test_label = dataload
            output = model(test_data.to(device))
            loss_test = F.cross_entropy(output, test_label.long())
            predicts = torch.max(output, 1)[1].numpy()
            test_acc = accuracy(output, test_label)
            total_acc.append(test_acc)

    #ecaluation
    if rank == 0:
        torch.writer.add_scalar("mean_loss", scalar_value=mean_loss, global_step=epoch + 1)

    torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    dist.destroy_process_group()
# save model
# torch.save(model.module.cpu().state_dict(), "work_result/model.pt")

print('Finished!')





# test phase
# model.eval()
# total_acc = []
# with torch.no_grad():
#     for i, dataload in enumerate(val_loader):
#         test_data, test_label = dataload
#         output = model(test_data.to(device))
#         loss_test = F.cross_entropy(output, test_label.long())
#         predicts = torch.max(output, 1)[1].numpy()
#         test_acc = accuracy(output, test_label)
#         total_acc.append(test_acc)
#
#     avg_acc = np.array(total_acc).mean()
#     # print(test_acc)
#     # acc_test = accuracy(output, test_label)
#     # test_history = 'Test set results:\n' + 'loss= {:.4f}'.format(loss_test.item()) + 'accuracy= {:.4f}'.format(acc_test.item())
#     # f.write(test_history)
#     print("Optimization Finished!")
#     print(
#         "Total time elapsed: {:.4f}s".format(time.time() - t_total),
#         " Testing average accurancy: {:.4f}".format(avg_acc)
#





