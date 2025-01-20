# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import numpy as np
import tqdm
import os
import sys
from torch.nn.parallel import DistributedDataParallel as DDP

# Configurations
batch_size = 256
test_batch = 256
nepoch = 5
n_cls = 10
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_gpus = torch.cuda.device_count()
world_size = n_gpus

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(), # image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [0, 1] -> [-1, 1]
])

# Network structure
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 128, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_process(rank, world_size):
    setup(rank, world_size)
    print(f"Running on rank {rank}")
    net = ResNet(10).to(rank)
    ddp_net = DDP(net, device_ids=[rank])

    # Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), # image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [0, 1] -> [-1, 1]
    ])

    # Dataloaders
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size // world_size,
                                            shuffle=False, pin_memory=True,
                                            sampler=DistributedSampler(trainset))

    # Optimization object
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_net.parameters(), lr=4e-2, momentum=0.9)

    avg_loss_train = []
    print(f'Train: {len(trainset)} instances, {len(trainloader)} mini-batches')
    for epoch in range(nepoch):
        trainloader.sampler.set_epoch(epoch)
        dist.barrier()
        if rank == 0:
            print(f'epoch {epoch + 1}')
        total_loss = torch.tensor(0, dtype=torch.float32).to(rank)
        total = torch.tensor(0, dtype=torch.int, device=rank)
        correct = torch.tensor(0, dtype=torch.int, device=rank)
        for data in trainloader:
            inputs, labels = data
            inputs = inputs.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)
            optimizer.zero_grad()
            outputs = ddp_net(inputs)
            total += labels.shape[0]
            correct += torch.sum(torch.argmax(outputs, 1) == labels)
            loss = criterion(outputs, labels)
            total_loss += loss
            loss.backward()
            optimizer.step()
        avg_loss = total_loss.item() / len(trainloader)
        avg_loss_train.append(avg_loss) 
        print(f'average loss : {avg_loss}')
        print(f'acc: {correct.item() / total.item() * 100:.2f}%')
        dist.barrier()
    print("Training finished")
    # Validation, single process
    if rank == 0:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                        transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, pin_memory=True, num_workers=4)
        with torch.no_grad():
            cls_size = torch.zeros(n_cls)
            cls_correct = torch.zeros(n_cls)
            for data in tqdm.tqdm(testloader):
                inputs, labels = data
                inputs = inputs.to(rank)
                predictions = torch.argmax(ddp_net(inputs), 1).to('cpu')
                for label, prediction in zip(labels, predictions):
                    cls_size[label] += 1
                    cls_correct[label] += prediction == label
            rate = (torch.sum(cls_correct) / torch.sum(cls_size)).item() * 100
            print(f"Accuracy:{rate:6.1f}%")
            for i in range(n_cls):
                rate_cls = (cls_correct[i] / cls_size[i]).item() * 100
                print(f"{classes[i]:10}: {rate_cls:6.1f}%")
    cleanup()

if __name__ == "__main__":
    world_size = n_gpus
    mp.spawn(train_process, args=(world_size, ), nprocs=world_size, join=True)
