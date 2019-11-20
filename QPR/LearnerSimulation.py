# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import numpy as np

import torch
# import torch.distributed.deprecated as dist
from cjltest.divide_data import partition_dataset, select_dataset
from cjltest.models import MnistCNN, AlexNetForCIFAR,LeNetForMNIST
from cjltest.utils_data import get_data_transform
from cjltest.utils_model import MySGD, test_model
from torch.autograd import Variable
from torch.multiprocessing import Process as TorchProcess
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import ResNetOnCifar10

parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--ps-ip', type=str, default='127.0.0.1')
parser.add_argument('--ps-port', type=str, default='29500')
parser.add_argument('--this-rank', type=int, default=1)
parser.add_argument('--workers', type=int, default=2)

# 模型与数据集
parser.add_argument('--data-dir', type=str, default='~/dataset')
parser.add_argument('--data-name', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='LROnMnist')
parser.add_argument('--save-path', type=str, default='./')

# 参数信息
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0)
parser.add_argument('--train-bsz', type=int, default=200)
parser.add_argument('--stale-threshold', type=int, default=0)
parser.add_argument('--ratio', type=float, default=1) # pulling ratio
parser.add_argument('--bit', type=int, default=1) # number of quantization bit

args = parser.parse_args()

# return a quantized gradient
def quantization(gradient, bit_num):
    sections = 2 ** bit_num
    sections /= 2

    # compute 2-norm of gradient
    gradient_square = 0.0
    for g_layer in gradient:
        gradient_square += torch.sum(g_layer * g_layer)
    gradient_value = torch.sqrt(gradient_square)
    section_value = gradient_value/sections

    gradient_quantized = []
    for g_layer in gradient:
        section_no = torch.round(g_layer / section_value)
        g_layer_new = section_value * section_no
        gradient_quantized.append(g_layer_new)
    return gradient_quantized, bit_num/32

# judge if the worker pulls down the global parameter based on random selection
def pull_judge(workers_num, ratio):
    if ratio==1:
        pull_workers_list = [w+1 for w in range(workers_num)]
        return pull_workers_list
    pull_workers_num = int(workers_num * ratio)

    pull_workers_list = []
    pull_workers_count = 0
    while pull_workers_count < pull_workers_num:
        pull_worker = np.random.randint(1, workers_num+1, 1)
        if pull_worker not in pull_workers_list:
            pull_workers_list.append(pull_worker[0])
            pull_workers_count += 1

    return pull_workers_list


# noinspection PyTypeChecker
def run(workers, models, save_path, train_data_list, test_data, iterations_epoch):
    workers_num = len(workers)
    print('Model recved successfully!')
    optimizers_list = []
    if args.lr == 0.0:
        if args.model in ['MnistCNN', 'AlexNet', 'ResNet18OnCifar10']:
            learning_rate = 0.1
        else:
            learning_rate = 0.01
    else:
        learning_rate = args.lr

    for i in workers:
        optimizer = MySGD(models[i].parameters(), lr=learning_rate)
        optimizers_list.append(optimizer)

    if args.model in ['MnistCNN', 'AlexNet']:
        criterion = torch.nn.NLLLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.model in ['AlexNet', 'ResNet18OnCifar10']:
        decay_period = 50
    else:
        decay_period = 1000

    print('Begin!')

    # store (train loss, energy, iterations)
    trainloss_file = './trainloss' + args.model + '.txt'
    if(os.path.isfile(trainloss_file)):
        os.remove(trainloss_file)
    f_trainloss = open(trainloss_file, 'a')

    log_file = args.model + 'log.txt'
    if(os.path.isfile(log_file)):
        os.remove(log_file)
    f_log = open(log_file, 'a')

    train_data_iter_list = []
    for i in workers:
        train_data_iter_list.append(iter(train_data_list[i-1]))

    epoch_train_loss = 0.0
    total_time = 0.0
    total_pulling_ratio = 0.0
    epoch_avg_pull_ratio = 0.0

    clock_epoch = 0
    test_loss = 0
    test_acc = 0
    for iteration in range(args.epochs * iterations_epoch):
        clock_epoch += 1
        iteration_loss =0.0
        epoch = int((iteration+1)/iterations_epoch)
        for i in workers:
            models[i].train()

        g_list = []
        for i in workers:
            try:
                data, target = next(train_data_iter_list[i-1])
            except StopIteration:
                train_data_iter_list[i-1] = iter(train_data_list[i - 1])
                data, target = next(train_data_iter_list[i-1])
            data, target = Variable(data), Variable(target)
            optimizers_list[i-1].zero_grad()
            output = models[i](data)
            loss = criterion(output, target)
            loss.backward()
            delta_ws = optimizers_list[i-1].get_delta_w()
            g_list.append(delta_ws)
            iteration_loss += loss.data.item()/workers_num
        epoch_train_loss += iteration_loss

        g_q_list = []
        for g in g_list:
            g_quantization, compression_ratio = quantization(g, args.bit)
            g_q_list.append(g_quantization)

        # 同步操作
        g_avg = []
        for p_idx, param in enumerate(models[0].parameters()):
            global_update_layer = torch.zeros_like(param.data)
            for w in workers:
                global_update_layer += g_q_list[w-1][p_idx]
            tensor = global_update_layer / workers_num
            g_avg.append(tensor)
            param.data -= tensor
        pull_workers = 0
        pull_workers_list = pull_judge(workers_num, args.ratio)
        for w in workers:
            isPulling = w in pull_workers_list
            if isPulling:
                pull_workers += 1
            for p_idx, param in enumerate(models[0].parameters()):
                if isPulling:
                    list(models[w].parameters())[p_idx].data = param.data
                else:
                    list(models[w].parameters())[p_idx].data -= g_q_list[w-1][p_idx]

        print('Epoch {}, Loss:{}'.format(epoch, loss.data.item()))
        total_pulling_ratio += pull_workers/workers_num
        epoch_avg_pull_ratio += pull_workers/workers_num
        f_log.write(str(args.this_rank) +
                          "\t" + str(iteration_loss) +
                          "\t" + str(epoch) +
                          "\t" + str(pull_workers/workers_num) +                    # the ratio of pulling workers
                          "\t" + str(iteration) +
                          "\t" + str(pull_workers_list) +
                          '\n')
        f_log.flush()

        # train loss every epoch
        if iteration % iterations_epoch == 0:
            # 训练结束后进行test
            if iteration % (2*iterations_epoch) == 0:
                test_loss, test_acc = test_model(0, model, test_data, criterion=criterion)
            f_trainloss.write(str(args.this_rank) +
                              "\t" + str(epoch_train_loss / float(clock_epoch)) +
                              "\t" + str(test_loss) +
                              "\t" + str(test_acc) +
                              "\t" + str(total_pulling_ratio) +  # accumulated pulling ratio of workers
                              "\t" + str(epoch) +
                              "\t" + str(epoch_avg_pull_ratio/clock_epoch) +  # the avg ratio of pulling workers in an epoch
                              "\t" + str(iteration) +
                              "\t" + str(total_time) +  # time
                              '\n')
            f_trainloss.flush()
            epoch_train_loss = 0.0
            epoch_avg_pull_ratio = 0.0
            clock_epoch = 0
            for i in workers:
                if (epoch + 1) % decay_period == 0:
                    for param_group in optimizers_list[i - 1].param_groups:
                        param_group['lr'] *= 0.1
                        print('LR Decreased! Now: {}'.format(param_group['lr']))


    f_log.close()
    f_trainloss.close()



def init_processes(workers,
                   models, save_path,
                   train_dataset_list, test_dataset,iterations_epoch,
                   fn, backend='tcp'):
    fn(workers, models, save_path, train_dataset_list, test_dataset, iterations_epoch)


if __name__ == '__main__':
    torch.manual_seed(1)

    workers_num = args.workers
    workers = [v+1 for v in range(workers_num)]
    models = []


    for i in range(workers_num + 1):
        if args.model == 'MnistCNN':
            model = MnistCNN()

            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'LeNet':
            model = LeNetForMNIST()

            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'LROnMnist':
            model = ResNetOnCifar10.LROnMnist()
            train_transform, test_transform = get_data_transform('mnist')

            train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'LROnCifar10':
            model = ResNetOnCifar10.LROnCifar10()
            train_transform, test_transform = get_data_transform('cifar')

            train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                           transform=train_transform)
            test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                          transform=test_transform)
        elif args.model == 'AlexNet':

            train_transform, test_transform = get_data_transform('cifar')

            if args.data_name == 'cifar10':
                model = AlexNetForCIFAR()
                train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                                 transform=train_transform)
                test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                                transform=test_transform)
            else:
                model = AlexNetForCIFAR(num_classes=100)
                train_dataset = datasets.CIFAR100(args.data_dir, train=True, download=False,
                                                  transform=train_transform)
                test_dataset = datasets.CIFAR100(args.data_dir, train=False, download=False,
                                                 transform=test_transform)
        elif args.model == 'ResNet18OnCifar10':
            model = ResNetOnCifar10.ResNet18()

            train_transform, test_transform = get_data_transform('cifar')
            train_dataset = datasets.CIFAR10(args.data_dir, train=True, download=False,
                                             transform=train_transform)
            test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                            transform=test_transform)
        elif args.model == 'ResNet34':
            model = models.resnet34(pretrained=False)

            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            test_transform = train_transform
            train_dataset = datasets.ImageFolder(args.data_dir, train=True, download=False,
                                             transform=train_transform)
            test_dataset = datasets.ImageFolder(args.data_dir, train=False, download=False,
                                            transform=test_transform)
        else:
            print('Model must be {} or {}!'.format('MnistCNN', 'AlexNet'))
            sys.exit(-1)
        models.append(model)
    train_bsz = args.train_bsz
    train_bsz /= len(workers)
    train_bsz = int(train_bsz)

    train_data = partition_dataset(train_dataset, workers)
    train_data_list = []
    for i in workers:
        train_data_sub = select_dataset(workers, i, train_data, batch_size=train_bsz)
        train_data_list.append(train_data_sub)

    test_bsz = 400
    # 用所有的测试数据测试
    test_data = DataLoader(test_dataset, batch_size=test_bsz, shuffle = False)

    iterations_epoch = int(len(train_dataset) / args.train_bsz)

    save_path = str(args.save_path)
    save_path = save_path.rstrip('/')

    p = TorchProcess(target=init_processes, args=(workers,
                                                  models, save_path,
                                                  train_data_list, test_data,iterations_epoch,
                                                  run))
    p.start()
    p.join()
