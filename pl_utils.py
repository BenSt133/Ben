"""
Utility functions for pytorch_lightning.
"""
import torch
import numpy as np

import glob
import os
from functools import reduce

def default_scheduler(lr_scheduler):
    return {'scheduler':lr_scheduler, 'monitor':'val_checkpoint_on'}

def ckpt_file(dir):
    """
    Returns a ckpt file given a directory
    """
    return glob.glob(os.path.join(dir,'*.ckpt'))[0]

def get_parameters(modules):
    return reduce(lambda a, b: a + b, [list(x.parameters()) for x in modules])

def get_dirname(): 
    if os.name == "nt":
        current_dir = os.getcwd().replace("C:\\Users\\to232\\Dropbox\\nonlinear_NN_data\\", "")
        current_dir = current_dir.replace("\\", "--")

    if os.name == "posix":
        current_dir = os.getcwd()
        replace_dir = os.path.join(os.environ['HOME'], "Dropbox", "nonlinear_NN_data", "")
        current_dir = current_dir.replace(replace_dir, "")
        current_dir = current_dir.replace("/", "--")

    return current_dir


def test_plmodel(plmodel, data_loader, Nrepeat=4):
    """
    Function that is helpful for evaluating a model multiple times in the presence of noise! 
    """
    accuracies = []
    losses = []
    for i in range(Nrepeat):
        for batch in data_loader:
            batch = [i.to(plmodel.device) for i in batch]
            with torch.no_grad():
                out = plmodel.validation_step(batch, 0)
            accuracies.append(out['val_accu'])
            losses.append(out['val_loss'])
    return torch.stack(accuracies).cpu().numpy(), torch.stack(losses).cpu().numpy()