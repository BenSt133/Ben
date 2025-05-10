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

class ClassificationModelFC(pl.LightningModule):
    """
    Has a fully connected output layer, with each node of the output vector representing a probability of a given class
    """
    def __init__(self):
        super().__init__()
        self.accu_metric = Accuracy()
        #self.loss = nn.CrossEntropyLoss
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        #print(y_hat.shape)
        #print(y.shape)
        #loss = self.loss(y_hat.float(), y.float())
        loss = F.nll_loss(y_hat, y)
        
        y_pred = torch.max(y_hat.data,1)[1]
        
        accuracy = self.accu_metric(y_pred, y)

        result = pl.TrainResult(loss)
        result.log('train_loss', torch.sqrt(loss.detach()))
        result.log('train_accu', accuracy.detach())
        del accuracy,y_pred,loss,y_hat,x,y
        return result
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        #print(y_hat.shape)
        #print(y.shape)
        #loss = self.loss(y_hat.float(), y.float())
        loss = F.nll_loss(y_hat, y)

        y_pred = torch.max(y_hat.data,1)[1]
        
        accuracy = self.accu_metric(y_pred, y)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', torch.sqrt(loss.detach()))
        result.log('val_accu', accuracy.detach())
        del accuracy,y_pred,loss,y_hat,x,y
        return result
    

import abc
    
class ClassificationLagPlModel(pl.LightningModule):
    """
    Classifies via conventional ML approach!
    The loss function has both an cross entropy as well as a lagrangian term
    Thus for this particular module, you need to define the lagrangian function in the class!
    """
    def __init__(self):
        super().__init__()
        self.accu_metric = Accuracy()
    @abc.abstractmethod
    def lagrangian(self):
        """
        A lagragian loss term that will be added to the loss function during backprop!
        If this method is not included, you cannot inherent from this class.
        """
    def _innerstep(self, batch):
        """
        Function that runs model, evaluate loss and accuracy
        """
        x, y = batch
        out = self(x)
        cr_loss = F.cross_entropy(out, y) 
        lag_loss = self.lagrangian() 
        loss = cr_loss + lag_loss
        y_pred = torch.max(out, 1)[1]
        accuracy = self.accu_metric(y_pred, y)        
        return cr_loss, lag_loss, loss, accuracy
    def training_step(self, batch, batch_idx):
        cr_loss, lag_loss, loss, accuracy = self._innerstep(batch)
        result = pl.TrainResult(loss)
        result.log('train_cr_loss', cr_loss)
        result.log('train_lag_loss', lag_loss)
        result.log('train_loss', loss)
        result.log('train_accu', accuracy)
        return result
    def validation_step(self, batch, batch_idx):
        cr_loss, lag_loss, loss, accuracy = self._innerstep(batch)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_cr_loss', cr_loss)
        result.log('val_lag_loss', lag_loss)
        result.log('val_loss', loss)
        result.log('val_accu', accuracy)
        return result
    def test_step(self, batch, batch_idx):
        cr_loss, lag_loss, loss, accuracy = self._innerstep(batch)
        result = pl.EvalResult(checkpoint_on=loss)
        return result    