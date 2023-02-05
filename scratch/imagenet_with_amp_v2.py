
import time, gc
import argparse

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




import torch, datetime, os
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

import torch.backends.cudnn as cudnn


import matplotlib.pyplot as plt
import numpy as np

from torch.cuda import amp


from torch.utils.tensorboard import SummaryWriter



torch.manual_seed(43)
cudnn.deterministic = True
cudnn.benchmark = False


# In[22]:


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
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size,
                                          shuffle=True, 
                                          num_workers=num_workers,
                                          pin_memory=True)
                                         

valset = torchvision.datasets.ImageFolder(root=os.path.join(datadir,'val'),
                                              transform=val_transform)
valloader = torch.utils.data.DataLoader(valset, 
                                             batch_size=batch_size,
                                             shuffle=True, 
                                             num_workers=num_workers,
                                             pin_memory=True)


# In[23]:


net=torchvision.models.resnet50()
if torch.cuda.is_available:
    device = 'cuda'
    net.cuda(torch.cuda.current_device());
else:
    device = 'cpu'

print(device)


# ### 3. Define a Loss function and optimizer
# Let's use a Classification Cross-Entropy loss and SGD with momentum.
# If trianing on GPUs, we can move the object for loss function to GPU memory as well 
# 
# 

# In[24]:


criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[25]:


scaler = amp.GradScaler()


# Let's define a training loop which does the following:
# - Read from training dataset images transformed tensors as batches as **inputs**
# - load **inputs** to device memory if training on a GPU
# - feed **inputs** to CNN and run a forward pass 
# - Apply loss function and run a backward propation of loss on each layer
# - Optimize weights using the optimizer 
# - Print average loss for every 2000 images trained
# We iterate over these step for N epochs. 

# In[26]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# In[ ]:

def train():
    net.to(device)
    start_timer()
    writer = SummaryWriter("logs/min_%d" %(datetime.datetime.now().minute))
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        # Train loop
        net.train()
        train_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs= data[0].cuda()
            labels= data[1].cuda()
    
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
            
        train_acc = accuracy(outputs,labels)
        train_loss = train_loss / len(trainloader.dataset)
        writer.add_scalar("Loss/train", train_loss , epoch)
        writer.add_scalar("Accuracy/train", train_acc , epoch)
     
        # Validation loop ( we won't backprop and optimize since this step is not training the model)
    #    net.eval()    
    #    val_loss = 0.0
    #    for i, data in enumerate(valloader, 0):
    #        # get the inputs; data is a list of [inputs, labels]
    #        inputs= data[0].cuda()
    #        labels= data[1].cuda()
    #        with torch.no_grad():
    #            outputs = net(inputs)
    #            loss = criterion(outputs, labels)
            val_loss += loss.item() #* data[0].size(0)
    
        val_acc = accuracy(outputs,labels)
        val_loss = val_loss / len(valloader.dataset)
        writer.add_scalar("Loss/val", val_loss , epoch)
        writer.add_scalar("Accuracy/val", val_acc , epoch)
        print(f'[{epoch + 1}] : Train loss:{train_loss:.3f} | Validation loss:{val_loss:.3f}')
    
    
        writer.flush
        
    end_timer_and_print('Finished Training')
    writer.close()

if __name__ == '__main__':
    print('world size: ', os.environ['WORLD_SIZE'])
    print('Rank: ', os.environ['RANK'])
    print('Local Rank: ', os.environ['LOCAL_RANK'])
