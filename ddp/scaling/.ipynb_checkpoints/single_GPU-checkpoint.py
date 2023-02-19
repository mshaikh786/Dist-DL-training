#!/usr/bin/env python
# coding: utf-8

# ## Boilerplate code

# In[1]:


# Funcitons for capturing time elapsed
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


# In[2]:


import torch, datetime, os

# Essential packages for training an image classifier in PyTorch
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda import amp

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import ResNet50_Weights

# In[3]:


torch.manual_seed(43)
cudnn.deterministic = True
cudnn.benchmark = False


# In[4]:


# import and instantiate tensorboard for monitoring model performance
from torch.utils.tensorboard import SummaryWriter


# Setting infrastructure for training in a Jupyter notebook.
# In a python script version of the code, this section should be parsed in as arguments.

# In[5]:


nodes = 1
gpus=1
num_workers = 10
batch_size=256
epochs=10
lr=1e-3
momentum=0.9
weight_decay=1e-4
print_interval=100


# ## Miscellaneous utility funtions

# In[6]:


def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return torch.sum(preds == labels).item()


# ## DataLoader
# Add a data management section to load and transform data.
# Here we manage not only the data location but also how it is loaded into memory

# In[7]:


# Prepare training data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
    ])


val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
    ])

datadir=os.environ['DATA_DIR']
trainset = torchvision.datasets.ImageFolder(root=os.path.join(datadir,'train'),
                                                transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size,
                                          shuffle=True, 
                                          num_workers=num_workers,
                                          pin_memory=True,
                                          drop_last=False)                                       

valset = torchvision.datasets.ImageFolder(root=os.path.join(datadir,'val'),
                                              transform=val_transform)
valloader = torch.utils.data.DataLoader(valset, 
                                             batch_size=batch_size,
                                             shuffle=False, 
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             drop_last=False)


# ## Choose a Neural Network architecture

# In[8]:


# Pre-training
#net=torchvision.models.resnet50(weights=None,num_classes=1000)
# Transfer learning for ImageNetDataset num_classes=1000
net=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)


# ## Define a Loss function and optimizer
# Let's use a Classification Cross-Entropy loss and SGD with momentum.
# If trianing on GPUs, we can move the object for loss function to GPU memory as well 
# 
# 

# In[9]:


if torch.cuda.is_available:
    device = 'cuda'
    net.cuda(torch.cuda.current_device());
else:
    device = 'cpu'
    
print(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), 
                      lr=lr, 
                      momentum=momentum,
                      weight_decay=weight_decay)


# ## Enable AMP
# Instantiate a wrapper to implement Automatic mixed precission during trianing

# In[10]:


scaler = amp.GradScaler()


# ## Training

# In[11]:

def main():
    print('Starting the training')
    net.to(device)
    start_timer()
    writer = SummaryWriter("logs/experiment_%s" %(os.environ['SLURM_JOBID']))
    for epoch in range(epochs):  # loop over the dataset multiple times
    
        # Train loop
        net.train()
        train_loss = 0.0
        train_acc = 0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs= data[0].cuda(non_blocking=True)
            labels= data[1].cuda(non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            with torch.cuda.amp.autocast(enabled=True):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_acc += accuracy(outputs,labels)

        train_loss = train_loss / len(trainloader.dataset.targets)
        train_acc  = 100 * train_acc / len(trainloader.dataset.targets)
        writer.add_scalar("Loss/train", train_loss , epoch)
        writer.add_scalar("Accuracy/train", train_acc , epoch)
 
        # Validation loop ( we won't backprop and optimize since this step is not training the model)
        net.eval()    
        val_loss = 0.0
        val_acc = 0
        for i, data in enumerate(valloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs= data[0].cuda(non_blocking=True)        
            labels= data[1].cuda(non_blocking=True)
            with torch.no_grad():
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item() 
            val_acc  += accuracy(outputs,labels)
        val_loss = val_loss / len(valloader.dataset.targets)
        val_acc  = 100 * val_acc / len(valloader.dataset.targets)
        writer.add_scalar("Loss/val", val_loss , epoch)
        writer.add_scalar("Accuracy/val", val_acc , epoch)
        print(f'[{epoch + 1}] :Loss (train, val):{train_loss:.3f}, {val_loss:.3f}| Accuracy (train,val):  {train_acc:.3f}, {val_acc:.3f}')
        writer.flush
    
    end_timer_and_print('Finished Training')
    writer.close()


# ## Save a checkpoint

if __name__ =="__main__":
    main()
    PATH = './tiny_imagenet.pth'
    torch.save(net.state_dict(), PATH)
