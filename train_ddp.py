import argparse
import os
import random
import shutil
import time
from train_simpleddp import args
import warnings
import logging

import torch
from torch.multiprocessing.spawn import spawn
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter


# from apex import amp
# from apex.parallel import DistributedDataParallel
from feeder.feeders import Feeder
from net.st_gcn import Model


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='data/ucf101_skeleton/train_data.npy', help='path to train dataset')
parser.add_argument('--label', default='data/ucf101_skeleton/train_label.pkl', help='path to train label')
parser.add_argument('--val_data', default='data/ucf101_skeleton/val_data.npy', help='path to val dataset')
parser.add_argument('--val_label', default='data/ucf101_skeleton/val_label.pkl', help='path to val label')
parser.add_argument('--nprocs',
                    default=2,
                    type=int,
                    metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=32,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 6400), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
args = parser.parse_args()


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt



def main_worker(rank, nprocs, args):
    best_acc1 = .0
    torch.cuda.set_device(rank)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=20, format='%(asctime)s - %(message)s')

    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=args.nprocs, rank=rank)
    # create model
    model = Model(
        in_channels = 3,
        num_class = 80,
        # graph = 'graph.Graph',
        graph_args = {"layout":'openpose', "strategy":'spatial'},
        edge_importance_weighting = True
    ).cuda(rank)

    args.batch_size = int(args.batch_size / nprocs)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(rank)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # model, optimizer = amp.initialize(model, optimizer)
    model = DDP(model, device_ids=[rank])

    cudnn.benchmark = True

    # Data loading code

    train_dataset = Feeder(args.data, args.label)
    val_dataset = Feeder(args.val_data, args.val_label)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True)

    # if args.evaluate:
    #     validate(val_loader, model, criterion, rank, args)
    #     return

    # tesorboard writer
    writer = SummaryWriter()

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss = train(train_loader, model, criterion, optimizer, epoch, rank, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, rank, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if rank == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best)

        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('Accuracy', acc1, epoch)

    logging.FileHandler('logs/{}_log.txt'.format(time.strftime(r"%Y-%m-%d-%H_%M_%S", time.localtime())))

def train(train_loader, model, criterion, optimizer, epoch, rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    
    for i, traindata in enumerate(train_loader):
        data, label = traindata
        data = data.cuda(rank)
        label = label.cuda(rank)
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(data)
        loss = criterion(output, label)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, label, topk=(1, 5))

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), data.size(0))
        top1.update(reduced_acc1.item(), data.size(0))
        top5.update(reduced_acc5.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return losses.avg
    


def validate(val_loader, model, criterion, rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, valdata in enumerate(val_loader):
            data, label = valdata
            data = data.cuda(rank)
            label = label.cuda(rank)

            # compute output
            output = model(data)
            loss = criterion(output, label)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, label, topk=(1, 5))

            # torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), data.size(0))
            top1.update(reduced_acc1.item(), data.size(0))
            top5.update(reduced_acc5.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)


        # TODO: this should also be done with the ProgressMeter
        logging.info(' * Val Acc@1 {top1.avg:.3f} Val Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg/100


def save_checkpoint(state, is_best, filename='checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoints/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, label, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = label.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args), join=True)