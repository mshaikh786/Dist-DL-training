import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

import torch, datetime, os, argparse

# Essential packages for training an image classifier in PyTorch
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda import amp

import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torchvision.models import ResNet50_Weights


# Setting the seed
pl.seed_everything(42)


# # Lightning Data module


class MYDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int = 64, num_workers:int = 10, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        
        self.num_workers = 4
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])


        self.val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        
        print('Initing class')
    def prepare_data(self):
        print('Preparing data')
    
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'train' or stage is None:
            print('setup stage fit')
            self.trainset=ImageFolder(root=os.path.join(self.data_dir,'train'),
                                                transform=self.train_transform)

        # Assign test dataset for use in dataloader(s)
        if stage == 'val' or stage is None:
            print('setup stage test')
            self.valset = ImageFolder(root=os.path.join(self.data_dir,'val'),
                                              transform=self.val_transform)
    
    def train_dataloader(self):
        print('Train loader')
        return DataLoader(self.trainset, 
                                          batch_size=self.batch_size,
                                          shuffle=True, 
                                          num_workers=self.num_workers,
                                          pin_memory=True,
                                          drop_last=False)
    def val_dataloader(self):
        print('Validation loader')
        return DataLoader(self.valset, 
                                             batch_size=self.batch_size,
                                             shuffle=False, 
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False)


# # Lightning module for training
# 
# In PyTorch Lightning, we define pl.LightningModule's (inheriting from Module) that organize our code into 5 main sections:
# 
# - Initialization (__init__), where we create all necessary parameters/models
# - Optimizers (configure_optimizers) where we create the optimizers, learning rate scheduler, etc.
# - Training loop (training_step) where we only have to define the loss calculation for a single batch (the loop of optimizer.zero_grad(), loss.backward() and optimizer.step(), as well as any logging/saving operation, is done in the background)
# - Validation loop (validation_step) where similarly to the training, we only have to define what should happen per step
# - Test loop (test_step) which is the same as validation, only on a test set.

# In[5]:


class CLASSIFY_lit_module(pl.LightningModule):
    def __init__(self,num_classes:int =200, weights=None):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use -- SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        if weights is not None:
            self.model = torchvision.models.resnet50(weights=weights)
        else:
            self.model = torchvision.models.resnet50(weights=weights,num_classes=num_classes)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()        

    def forward(self, inputs):
        # Forward function that is run when visualizing the graph
        return self.model(inputs)

    def configure_optimizers(self):
        # We choose SGD as our optimizers.
        optimizer = optim.SGD(self.parameters(), lr=1e-3)
        
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        #scheduler = optim.lr_scheduler.StepLR(optimizer, 
        #                                      step_size=30, gamma=0.1)
        return [optimizer] #, [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss_module(outputs, labels)
        acc = (outputs.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs).argmax(dim=-1)
        acc = (labels == outputs).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc,on_step=False, on_epoch=True)


def main(args):
# # Trainer definition

# Now that the data pipeline and training scheme is defined, we pass them to the Lightning's execution framework to automate the execution of the training workflow:
# - Epoch and batch iteration
# - Calling forward, loss eval, and backward passes
# - Evaluating cross validation
# - Saving and loading weights
# - MultiGPU support
# - Mixed precision training
# 
# And more

# ### Initialize data module


    data = MYDataModule(batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    data_dir=args.data_dir)


# In[7]:


    data.prepare_data()


# In[8]:


    data.setup()


# ### Initialize model

# In[9]:


    net = CLASSIFY_lit_module(args.num_classes,
                              args.weights)


# In[10]:


    CHPKT_PATH=os.path.join(os.environ['PWD'],'chkpt')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=CHPKT_PATH,
                                                   filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
                                                  save_weights_only=True,
                                                  mode="max",
                                                  monitor='train_acc')
    logger = TensorBoardLogger(save_dir="logs",
                           sub_dir=None,
                           name=None,
                           version=None,
                           default_hp_metric=False,
                          )



    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=args.epochs,
                     logger=logger,
                     callbacks=[checkpoint_callback],
                     accelerator="auto", 
                     devices=args.gpus_per_node, 
                     num_nodes=args.num_nodes, 
                     strategy="ddp",
                     plugins=[SLURMEnvironment(auto_requeue=False)],
                     benchmark=False,
                     deterministic=True,
                     precision=args.precision
                     )




    
    # Train the model 
    trainer.fit(net, data)


if __name__=="__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", default=1,
                        help="Number of nodes (machines)", type=int)
    parser.add_argument("--gpus-per-node", default=1,
                        help="Number of nodes (machines)", type=int)
    parser.add_argument("--num-workers", default=4,
                        help="Number of dataloader threads per node", type=int)
    parser.add_argument("--num-classes", default=200,
                        help="number of classes in the model", type=int)
    parser.add_argument("--batch-size", default=256,
                        help="mini batch size per GPU", type=int)
    parser.add_argument("--epochs", default=5,
                        help="total epochs", type=int)
    parser.add_argument("--precision", default=32,
                        help="Automatic mixed precision data type", type=int)
    parser.add_argument("--data-dir", default='/ibex/ai/reference/CV/tinyimagenet',
                        help="Directory with Dataset", type=str)
    parser.add_argument("--weights", default=None,
                        help="Enable transfer learning" , type=str)
    args = parser.parse_args()
    main(args)




