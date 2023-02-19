import argparse, os, sys, tempfile
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.backends.cudnn as cudnn
from torch.cuda import amp



# import and instantiate tensorboard for monitoring model performance
from torch.utils.tensorboard import SummaryWriter

import time, gc

# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))


torch.manual_seed(43)
cudnn.deterministic = True
cudnn.benchmark = False



def dataloader(gpu,world_size,batch_size,num_workers):
# Prepare training data
    train_transform = transforms.Compose(
    [transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])


    val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    #datadir=os.environ['DATA_DIR']
    datadir='/ibex/ai/home/shaima0d/tiny-imagenet-200'
    
    trainset = torchvision.datasets.ImageFolder(root=os.path.join(datadir,'train'),
                                                transform=train_transform)
    trainSampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
                                                               #num_replicas=world_size,
                                                               #rank=gpu,
                                                               #)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          pin_memory=True,
                                          shuffle=(trainSampler is None),
                                          sampler=trainSampler)
                                         

    valset = torchvision.datasets.ImageFolder(root=os.path.join(datadir,'val'),
                                              transform=val_transform)
    valSampler = torch.utils.data.distributed.DistributedSampler(valset,
                                                                  num_replicas=world_size,
                                                                  rank=gpu,shuffle=False)
    valloader = torch.utils.data.DataLoader(valset, 
                                             batch_size=batch_size,
                                             shuffle=False, 
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             sampler=valSampler)
    return trainloader,valloader





def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def train (net,world_size,rank,args):
    gpu_id=rank
    print(gpu_id)
    
    net.cuda(gpu_id)
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    
    trainloader, valloader = dataloader(gpu_id,world_size,
                                        args.batch_size,
                                        args.num_workers)
    # Wrap model as DDP
    net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[local_rank])
                                                    
    print_interval = args.print_interval
    if gpu_id == 0:
        start_timer()
    if rank == 0:
        print(f'Starting training: on total of {len(trainloader.dataset)} images')
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        train_loss = 0.0
        trainloader.sampler.set_epoch(epoch)
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].cuda(gpu_id), data[1].cuda(gpu_id)
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
        
        if gpu_id ==0:
            train_acc = accuracy(outputs,labels)
            train_loss = train_loss / len(trainloader.dataset)
            print(f'[{epoch + 1}] : Train loss:{train_loss:.3f} | Accuray: {train_acc:.3f}') # | Validation loss:{val_loss:.3f}')
            print(f'Epoch {epoch + 1} finished in {i + 1:5d} steps, with batches {args.batch_size} and total images {len(trainloader.dataset)}')
    
    if gpu_id == 0:
        end_timer_and_print('Finished Training')


if __name__ == '__main__':
    world_size= int(os.environ['WORLD_SIZE'])
    rank      = int(os.environ['RANK'])
    local_rank= int(os.environ['LOCAL_RANK'])
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=4,
                        help="number of dataloaders", type=int)
    parser.add_argument("--batch-size", default=32,
                        help="mini batch size per GPU", type=int)
    parser.add_argument("--epochs", default=5,
                        help="total epochs", type=int)
    parser.add_argument("--lr", default=0.001,
                        help="Learning rate",type=float)
    parser.add_argument("--momentum", default=0.9,
                        help="Momentum", type=float)
    parser.add_argument("--print-interval", default=100,
                        help="Momentum", type=int)
    args = parser.parse_args()
    
    setup(rank, world_size)
    net=torchvision.models.resnet50()
    train(net,world_size,rank,args)




