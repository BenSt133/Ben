"""
Simple functions for performing basic data wrangling of the spectrums
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def np2loaders(x, y, train_ratio=0.9, Nbatch = 100,intout=False):
    """
    Produces the training and testing pytorch dataloaders given numpy inputs
    x: input data in numpy format. First dimension is batch dimension
    y: output data in numpy format. First dimension is batch dimension
    train_ratio: The ratio of dataset that will be training data
    Nbatch: batchsize for the training set
    Note the testset has a batchsize of the whole training set
    """
    Ntotal = x.shape[0]
    Ntrain = int(np.floor(Ntotal*train_ratio))
    train_inds = np.arange(Ntrain)
    val_inds = np.arange(Ntrain, Ntotal)

    X_train = torch.tensor(x[train_inds]).float()
    X_val = torch.tensor(x[val_inds]).float()
    
    if intout == False:
        Y_train = torch.tensor(y[train_inds]).float()
        Y_val = torch.tensor(y[val_inds]).float()
    else:
        Y_train = torch.tensor(y[train_inds])
        Y_val = torch.tensor(y[val_inds])
        
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, Nbatch)
    val_loader = DataLoader(val_dataset, val_dataset.tensors[0].shape[0])

    return train_loader, val_loader

def load_loaders(file, Nbatch_train, Nbatch_val=None, Nbatch_test=None):
    """
    Generates dataloaders, given a dataset .p f
    """
    data = torch.load(file)

    train_dataset = data['train_dataset']
    val_dataset = data['val_dataset']
    test_dataset = data['test_dataset']

    x_train, y_train = train_dataset.dataset[train_dataset.indices]
    x_val, y_val = val_dataset.dataset[val_dataset.indices]
    x_test, y_test = test_dataset.dataset[test_dataset.indices]

    Nbatch_train = 30
    Nbatch_val = None
    Nbatch_test = None

    if Nbatch_val is None:
        Nbatch_val = len(val_dataset.indices)

    if Nbatch_test is None:
        Nbatch_test = len(test_dataset.indices)

    train_loader = DataLoader(dataset=train_dataset,batch_size=Nbatch_train)
    val_loader = DataLoader(dataset=val_dataset, batch_size=Nbatch_val)
    test_loader = DataLoader(dataset=test_dataset, batch_size=Nbatch_test)
    return train_loader, val_loader, test_loader