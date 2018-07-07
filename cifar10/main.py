"""Train CIFAR10 with PyTorch."""

from __future__ import print_function
from models import *

import time
import os
import copy
import argparse

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


best_accuracy = 0.0

transform = {
    'train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
}

train_sets = torchvision.datasets.CIFAR10('./data', train=True, transform=transform['train'], download=True)
test_sets = torchvision.datasets.CIFAR10('./data', train=False, transform=transform['test'], download=True)

train_loader = torch.utils.data.DataLoader(train_sets, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_sets, batch_size=100, shuffle=False, num_workers=2)

nets = {
    'LeNet': LeNet(),
    'VGG11': VGG('VGG11'),
    'VGG13': VGG('VGG13'),
    'VGG16': VGG('VGG16'),
    'VGG19': VGG('VGG19'),
    'GoogLeNet': GoogLeNet(),
    'ResNet18': ResNet18(),
    'ResNet34': ResNet34(),
    'ResNet50': ResNet50(),
    'ResNet101': ResNet101(),
    'ResNet152': ResNet152(),
    'ResNeXt29_2x64d': ResNeXt29_2x64d(),
    'ResNeXt29_4x64d': ResNeXt29_4x64d(),
    'ResNeXt29_8x64d': ResNeXt29_8x64d(),
    'ResNeXt29_32x4d': ResNeXt29_32x4d(),
    'PreActResNet18': PreActResNet18(),
    'PreActResNet34': PreActResNet34(),
    'PreActResNet50': PreActResNet50(),
    'PreActResNet101': PreActResNet101(),
    'PreActResNet152': PreActResNet152(),
    'DenseNet121': DenseNet121(),
    'DenseNet161': DenseNet161(),
    'DenseNet169': DenseNet169(),
    'DenseNet201': DenseNet201(),
    'DPN26': DPN26(),
    'DPN92': DPN92()
}

def params_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 trains with PyTorch')
    parser.add_argument('--net', default='LeNet', type=str, help='choose a network')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=100, type=int, help='epochs')
    parser.add_argument('--device', default='gpu', type=str, help='choose device(gpu or cpu)')
    return parser

def train(net, criterion, device):
    net.train()
    train_loss = 0.0
    total = 0
    corrects = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * targets.size(0)
        _, preds = torch.max(outputs, 1)
        total += targets.size(0)
        corrects += (preds == targets).sum().item()

    loss = train_loss / len(train_sets)
    acc = corrects / float(total)
    print('Train loss: {:.4f}, accuracy: {:.4f}'.format(loss, acc))

def test(net, criterion, epoch, device, net_name):
    global best_accuracy
    net.eval()
    test_loss = 0.0
    total = 0
    corrects = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _, preds = torch.max(outputs, 1)
            total += targets.size(0)
            test_loss += loss.item() * targets.size(0)
            corrects += (preds == targets).sum().item()

    loss = test_loss / len(test_sets)
    acc = corrects / float(total)
    print('Test loss: {:.4f}, accuracy: {:.4f}'.format(loss, acc))
    if acc > best_accuracy:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'accuracy': acc,
            'epoch': epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/%s.t7' % net_name)
        best_accuracy = acc


if __name__ == '__main__':
    parser = params_parser()
    args = parser.parse_args()

    net = nets[args.net]
    device = torch.device('cpu')
    if args.device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)

    start_time = time.time()
    start_epoch = 0
    best_accuracy = 0.0
    for i in xrange(3):
        for epoch in xrange(start_epoch, start_epoch+100):
            print('Epoch: {}'.format(epoch+1))
            train(net, criterion, device)
            test(net, criterion, epoch, device, args.net)
        start_epoch += 100
        # adjust the learning rate
        args.lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
    elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60))
