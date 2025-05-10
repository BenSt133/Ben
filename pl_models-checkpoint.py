import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
# import ipdb

class RegressionModel(pl.LightningModule):
    """
    Employs Mean Square Error loss to perform regression
    Note: Logs the square root of mse as it's easier to interpret
    """
    def __init__(self):
        super().__init__()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.TrainResult(loss)
        result.log('train_loss', torch.sqrt(loss))
        return result
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', torch.sqrt(loss))
        return result
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('test_loss', torch.sqrt(loss))
        return result

class Classification1DModel(pl.LightningModule):
    """
    Classifies by matching the 1D real number output to the closest integer
    The model employs thus employs the mseloss 
    Note: Logs the square root of mse as it's easier to interpret
    """
    def __init__(self):
        super().__init__()
        self.accu_metric = Accuracy()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.float()) 
        y_pred = y_hat.round().int()
        accuracy = self.accu_metric(y_pred, y)

        result = pl.TrainResult(loss)
        result.log('train_loss', torch.sqrt(loss))
        result.log('train_accu', accuracy)
        return result
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.float()) 

        y_pred = y_hat.round().int()
        accuracy = self.accu_metric(y_pred, y)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', torch.sqrt(loss))
        result.log('val_accu', accuracy)
        return result
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.float()) 

        y_pred = y_hat.round().int()
        accuracy = self.accu_metric(y_pred, y)

        result = pl.EvalResult(checkpoint_on=loss)
#         result.log('test_loss', torch.sqrt(loss))
#         result.log('test_accu', accuracy)
        return result