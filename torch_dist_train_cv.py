# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Benchmark scripts for TensorFlow and PyTorch."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import time
import warnings

try:
    import torch
except ImportError:
    print('To run benchmarks for PyTorch backend, ')
    print('PyTorch should be installed.')
    print('Instructions for install PyTorch are as follow:')
    print('  - https://pytorch.org')
    raise

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.resnet import resnet50
from models.vgg import vgg16

import flags

# Define and parse command line arguments.
flags.define_flags()
params = flags.parser.parse_args()

best_acc1 = 0


def get_optimizer(model, params, learning_rate):
    """Returns the optimizer that should be used based on params."""
    if params.optimizer == 'momentum':
        opt = torch.optim.SGD(
            model.parameters(), lr=learning_rate,
            momentum=params.momentum, nesterov=False,
        )
    elif params.optimizer == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(
            'Optimizer "{}" was not recognized'.
            format(params.optimizer),
        )
    return opt


def _validate_flags(params):
    """Check if command line arguments are valid or not."""
    if params.model is None:
        err_msg = 'The model to benchmark is not specified.\n' + \
                  "Using '--model' to specify one."
        print(err_msg)
        exit()

    if not params.do_train:
        err_msg = "'--do_train' must be specified."
        print(err_msg)
        exit()


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print('Use GPU: {} for training'.format(args.gpu))

    if args.distributed:
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = int(os.environ['RANK'])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group( #需要nccl、url、节点数、gpu的总数
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank,
        )
    # create model
    print(
        "=> rank_id '{}', world_size '{}'"
        .format(args.rank, args.world_size),
    )
    print("=> creating model '{}'".format(args.model))
    if args.model == 'resnet50':
        model = resnet50()
    elif args.model == 'vgg16':
        model = vgg16()
    else:
        raise ValueError("Only support resnet50 and vgg16.")

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # args.batch_size = int(args.batch_size / ngpus_per_node)
            # args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu],
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate
            # batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size
        # to all available GPUs
        if args.model.startswith('alexnet') or args.model.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # optimizer = get_optimizer(model, params, args.lr)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, nesterov=False,
    )
    cudnn.benchmark = True#将cudnn.benchmark设置为true，可显著提升速度

    # Data loading code
    train_dir = os.path.join(args.data_dir, 'train')
    # valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    # if args.evaluate:
    #     validate(val_loader, model, criterion, args)
    #     return

    for epoch in range(args.num_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        # acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)

        # if not args.multiprocessing_distributed or
        #     (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0):
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    speed = AverageMeter('Speed', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, speed, data_time, losses, top1, top5],
        prefix='Epoch: [{}]'.format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        exams_per_sec = args.batch_size / (time.time() - end)
        speed.update(exams_per_sec)

        if i % args.print_freq == 0:
            progress.display(i)
            #speed.reset()
            #batch_time.reset()

        end = time.time()


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
    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by
    10 every 30 epochs
    """
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions
    for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class BenchmarkCNN(object):
    """Class for benchmarking a cnn network."""

    def __init__(self, params):
        """Initialize BenchmarkCNN.

        Args:
          params: Params tuple, created by make_params_from_flags.
        Raises:
          ValueError: Unsupported params settings.
        """
        self.params = params
        self.batch_size_per_device = self.params.batch_size
        self.batch_size = self.params.batch_size

        self.print_info()

    def print_info(self):
        """Print basic information."""
        mode = ''
        if self.params.do_train:
            mode += 'train '

        print()
        print('Model:       %s' % self.params.model)
        print('Mode:        %s' % mode)
        print('Batch size   %s per device' % self.batch_size_per_device)
        print('Num epochs:  %d' % params.num_epochs)
        print('Optimizer:   %s' % params.optimizer)
        print('=' * 30)

    def run(self):
        """Run the benchmark task assigned to this process."""
        if params.seed is not None:
            random.seed(params.seed)
            torch.manual_seed(params.seed)
            cudnn.deterministic = True
            warnings.warn(
                'You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.',
            )

        params.distributed = params.world_size > 1 or \
            params.multiprocessing_distributed

        if params.dist_url == 'env://' and params.world_size == -1:
            params.world_size = int(os.environ['WORLD_SIZE'])

        ngpus_per_node = torch.cuda.device_count()
        if params.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node,
            # the total world_size
            # needs to be adjusted accordingly
            params.world_size = ngpus_per_node * params.world_size
            # Use torch.multiprocessing.spawn to launch
            # distributed processes: the
            # main_worker process function
            mp.spawn(
                main_worker, nprocs=ngpus_per_node,
                args=(ngpus_per_node, params),
            )
        else:
            # Simply call main_worker function
            main_worker(params.gpu, ngpus_per_node, params)


def main(params):
    """Run the benchmark."""
    # check common parameters
    _validate_flags(params)

    print('Running benchmarks for PyTorch.')
    bench = BenchmarkCNN(params)
    bench.run()


if __name__ == '__main__':
    main(params)
