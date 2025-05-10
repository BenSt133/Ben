import os
import numpy as np
import torch.nn.functional as F
import torch

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.metrics import Accuracy

from .torch_loader import *

accu_metric = Accuracy()

class TestCallback(Callback):
    """
    Function that tests the current code and saves the data in such a way that it will
    be saved in the future. 
    This current version assumes that the prediction is from .round, feel free to modify
    it for proper classification.
    # IMPORTANT: transfer parameters from ref_model to test_model 
    """
    def __init__(self, test_epoch, test_loader, test_model, ref_model, savedir,
                     Nrepeat=1, test_name="test", ref_name="ref"):
        self.test_epoch = test_epoch #this is testing every test_epoch epoch
        self.test_model = test_model
        self.test_loader = test_loader
        self.ref_model = ref_model
        self.Nrepeat = Nrepeat
        self.savedir = savedir
        self.test_name = test_name
        self.ref_name = ref_name
        
    def on_train_epoch_start(self, trainer, pl_module,device='cpu'):
        """Called when the train epoch begins."""
        test_epoch = self.test_epoch
        test_loader = self.test_loader
        test_model = self.test_model
        ref_model = self.ref_model
        Nrepeat = self.Nrepeat
        savedir = self.savedir
        
        epoch = trainer.current_epoch
        state_dict = ref_model.state_dict()
        fname = os.path.join(savedir, f"epoch_{epoch}.p")
        
        if epoch % test_epoch == 0:
            # IMPORTANT: transfer parameters from ref_model to test_model 
            test_model.load_state_dict(state_dict) 
            x, y = next(iter(test_loader))
            x = x.to(device) 
            y = y.to(device)
            x = x.repeat(Nrepeat, 1)
            y = y.repeat(Nrepeat)
            
            test_y_hat = test_model(x, save=True)
            test_y_pred = test_y_hat.round().int()
            test_accu = accu_metric(test_y_pred, y)
            test_loss = torch.sqrt(F.mse_loss(test_y_hat, y))

            ref_y_hat = ref_model(x, save=True)
            ref_y_pred = ref_y_hat.round().int()
            ref_accu = accu_metric(ref_y_pred, y)
            ref_loss = torch.sqrt(F.mse_loss(ref_y_hat, y))
            
            metrics = dict(epoch=epoch)
            metrics[self.test_name+"_accu"] = test_accu
            metrics[self.test_name+"_loss"] = test_loss
            metrics[self.ref_name+"_accu"] = ref_accu
            metrics[self.ref_name+"_loss"] = ref_loss
            
            save_dict = dict(state_dict=state_dict, metrics=metrics,
                             x = x, y=y,
                             test_xin=test_model.xin, #replace this line Martin!
                             test_xout=test_model.xout, 
                             test_y_hat=test_y_hat, test_y_pred=test_y_pred,
                             ref_xin=ref_model.xin, 
                             ref_xout=ref_model.xout, 
                             ref_y_hat=ref_y_hat, ref_y_pred=ref_y_pred
                            )
            save_data = torch.save(save_dict, fname)
            trainer.logger.log_metrics(metrics, step=trainer.global_step)

class RetrainCallback(Callback):
    """
    rt stands for retrain in this code!
    in most cases of course this stands for the digital twin
    """
    def __init__(self, rt_epoch, rt_xlist, rt_trainer, rt_model, run_exp, name="dt_error"):
        super().__init__()
        self.rt_epoch = rt_epoch
        self.rt_xlist = rt_xlist
        self.rt_trainer = rt_trainer
        self.rt_model = rt_model
        self.run_exp = run_exp
        self.name = name

    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the train epoch begins."""
        epoch = trainer.current_epoch
        rt_epoch = self.rt_epoch
        rt_xlist = self.rt_xlist
        rt_trainer = self.rt_trainer
        rt_model = self.rt_model
        run_exp = self.run_exp
        
        if epoch % rt_epoch == 0:
            print('Retraining digital twin') 
            rt_xlist=np.array(rt_xlist)
            print(rt_xlist.shape)
            y = run_exp(rt_xlist)
            rt_train_loader, rt_val_loader = np2loaders(rt_xlist, y, 
                                              train_ratio=0.8, Nbatch=100)
            rt_trainer.fit(rt_model, rt_train_loader, rt_val_loader)
            result = rt_trainer.test(rt_model, rt_val_loader)
            dt_error = result[0]['test_loss']
            metrics = dict(epoch=epoch)
            metrics[self.name] = dt_error
            trainer.logger.log_metrics(metrics, step=trainer.global_step)



def MNISTretraindata(xdim):
    #xdim is the parameter dimension, which must be a factor of 2*196
    Nx = 1000
    
    
    from torchvision.datasets import MNIST
    import torch
    from torchvision import transforms
    import numpy.random as random

    reshape_f = lambda x: torch.reshape(x[0, ::compress_factor, ::compress_factor], (-1, ))
    transform = transforms.Compose([transforms.ToTensor(), reshape_f])

    x = MNIST(".", train=True, download=True, transform=transform).train_data.numpy()
    compress_factor = 2 #Use either 1, 2 or 4! 
    x = x[:, ::compress_factor, ::compress_factor]
    x = x.reshape(x.shape[0], x.shape[1]**2)
    y=MNIST(".", train=True, download=True, transform=transform).train_labels.numpy()
    yn=np.array(y)
    xn=np.array(x)
    xn=xn*1.0/255
    
    def get_inputM(dim_parameter,input_dim=2*196):
    #This calculates the matrix that takes the parameter vector to a 196-D block-structured vector to modulate the input to the first layer
    #Notice dim_parameter must be a factor of input_dim!!    
        M = np.zeros([input_dim,dim_parameter])
        idx2=0
        idx3=0
        for idx in range(input_dim):
            M[idx,idx3]=1.
            idx2+=1
            if np.mod(idx2,input_dim//dim_parameter)==0:
                idx3+=1

        return M
    
    
    seed=10
       
    random.seed(seed)
    x_off = random.uniform(-1.0, 1.0, [Nx, xdim])
    x_fact = random.uniform(-1.0, 1.0, [Nx, xdim])

    M=get_inputM(xdim)
    xin=list()
    for idx in range(Nx):
        factors = M@x_fact[idx,:]
        offsets = M@x_off[idx,:]
        xnew=np.multiply(np.hstack((xn[idx,:],xn[idx,:])),factors)+offsets
        xin.append(xnew)

    xin=np.array(xin) #need to do the clamp!
    xin[xin>1]=1.
    xin[xin<-1]=1.
    return xin


            
class TestCallback_Classifier(Callback):
    """
    Same as above, but for a classifier in which softmax is applied at the end to get the result
    """
    def __init__(self, test_epoch, test_loader, test_model, ref_model, savedir,
                     Nrepeat=1, test_name="test", ref_name="ref"):
        self.test_epoch = test_epoch #this is testing every test_epoch epoch
        self.test_model = test_model
        self.test_loader = test_loader
        self.ref_model = ref_model
        self.Nrepeat = Nrepeat
        self.savedir = savedir
        self.test_name = test_name
        self.ref_name = ref_name
        
    def on_train_epoch_start(self, trainer, pl_module,device='cpu'):
        """Called when the train epoch begins."""
        test_epoch = self.test_epoch
        test_loader = self.test_loader
        test_model = self.test_model
        ref_model = self.ref_model
        Nrepeat = self.Nrepeat
        savedir = self.savedir
     
        
        epoch = trainer.current_epoch
        state_dict = ref_model.state_dict()
        fname = os.path.join(savedir, self.test_name + '_'+f"epoch_{epoch}.p")
        
        if epoch % test_epoch == 0:
            # IMPORTANT: transfer parameters from ref_model to test_model
            print('Saving model results')
            test_model.load_state_dict(state_dict) 
            x, y = next(iter(test_loader))
            x = x.to(device) 
            y = y.to(device)
            x = x.repeat(Nrepeat, 1)
            y = y.repeat(Nrepeat)
            
            test_y_hat = test_model(x, save=True)
            test_y_pred = torch.max(test_y_hat.data,1)[1]
            test_accu = accu_metric(test_y_pred, y)
            test_loss = torch.sqrt(F.cross_entropy(test_y_hat, y))

            ref_y_hat = ref_model(x, save=True)
            ref_y_pred = torch.max(ref_y_hat.data,1)[1]
            ref_accu = accu_metric(ref_y_pred, y)
            ref_loss = torch.sqrt(F.cross_entropy(ref_y_hat, y))  
            
            metrics = dict(epoch=epoch)
            metrics[self.test_name+"_accu"] = test_accu
            metrics[self.test_name+"_loss"] = test_loss
            metrics[self.ref_name+"_accu"] = ref_accu
            metrics[self.ref_name+"_loss"] = ref_loss
            
            save_dict = dict(state_dict=state_dict, metrics=metrics,
                             x = x, y=y,
                             test_xin=test_model.xin, #replace this line Martin!
                             test_xout=test_model.xout, 
                             test_y_hat=test_y_hat, test_y_pred=test_y_pred,
                             ref_xin=ref_model.xin, 
                             ref_xout=ref_model.xout, 
                             ref_y_hat=ref_y_hat, ref_y_pred=ref_y_pred
                            )
            save_data = torch.save(save_dict, fname)
            trainer.logger.log_metrics(metrics, step=trainer.global_step)
            