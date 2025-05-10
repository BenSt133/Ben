import numpy as np
import matplotlib.pyplot as plt
from imp import reload

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ana_lib.mlp import *
from ana_lib.pl_models import *
from ana_lib.pl_utils import *
from ana_lib.optuna_utils import *
from ana_lib.digitize import *
from ana_lib.plot_utils import *
from ana_lib.torch_loader import *
import ipdb

import time
import wandb

from scipy.interpolate import interp1d   

import os


gpu_no='0'
device='cuda:0'


def swish(x):
    return x * torch.sigmoid(x)

# The pytorch model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, Nunits,Ncoms,identity_init):
        super().__init__()
        self.layers = []
        Ncoms = int(Ncoms)
        input_dim0=input_dim
        for N in range(Ncoms):
            input_dim=input_dim0
            for Nunit in Nunits:
                self.layers.append(nn.Linear(input_dim, Nunit)) 
                if identity_init==1:
                    nn.init.eye_(self.layers[-1].weight)
                input_dim = Nunit
                
        self.Ncoms=Ncoms
        self.Nlayers=len(Nunits)
        self.layers.append(nn.Linear(input_dim, output_dim))
        # Assigning the layers as class variables (PyTorch requirement). 
        for idx, layer in enumerate(self.layers):
            setattr(self, "fc{}".format(idx), layer)
            
    def forward(self, data):
        data0 = data
        idx=0
        for n in range(self.Ncoms):
            datal=data0
            for l in range(self.Nlayers):
                datal = self.layers[idx](datal)
                datal = swish(datal)
                idx+=1
            if n==0:
                data = datal
            else:
                data = data+ datal                
        data = self.layers[-1](data)
        return data


def train_dig_twinNAS(X,Y,input_dim,output_dim,Ntrials,name,savename,max_epoch,gpu_no=0,meancheck=0.1):
      
    train_loader,val_loader = np2loaders(X,Y)

    #Defining the model!
    class MLP_Reg(RegressionModel):
        def __init__(self, input_dim=None, output_dim=None, Nunits=None, Ncoms=1,identity_init=1,lr=1e-3):
            super().__init__()
            self.save_hyperparameters() #this appends all the inputs into self.hparams
            for (i, Nunit) in enumerate(Nunits): #writing more attributes to hyperparams that will appear on wandb
                self.hparams[f'ldim_{i}'] = Nunit
            self.hparams['Nlayers'] = len(Nunits) #repeat - adding Nlayers
            self.hparams['Ncoms'] = Ncoms
            self.hparams['identity_init'] = identity_init
            self.model = MLP(input_dim, output_dim, Nunits,Ncoms,identity_init) #Multilayer Perceptron pytorch model

        def forward(self, data):
            return self.model(data)

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=self.hparams['lr'])
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=10)
            scheduler = default_scheduler(lr_scheduler) #a function basically to make up for pl_lightnings fault - please use it if you want to use a scheduler
    #         return optimizer
            return [optimizer], [scheduler]

    #Individual training function - returns the best validation loss as well as the path to the best model. 
    def train_digital_twin(name, *args):
        model = MLP_Reg(*args)

        csv_logger = pl.loggers.CSVLogger('logs', project_name, name)
        wandb_logger = pl.loggers.WandbLogger(name=name,project=project_name)
        while True:
            try:
                wandb_logger._experiment = wandb.init(name=wandb_logger._name, project=wandb_logger._project)
                break
            except:
                print("Retrying")
                time.sleep(10)
        logger = [csv_logger, wandb_logger]

        #saves checkpoints in the 
        checkpoint_cb = pl.callbacks.ModelCheckpoint(os.path.join(csv_logger.log_dir, "{epoch}-{val_loss:.5f}"))
        lr_cb = pl.callbacks.LearningRateLogger()
        trainer = pl.Trainer(max_epochs=max_epoch, logger=logger, gpus=gpu_no, weights_summary=None, 
                             progress_bar_refresh_rate=0, checkpoint_callback=checkpoint_cb, callbacks=[lr_cb])
        trainer.fit(model, train_loader, val_loader)

        wandb_logger.experiment.join()
        return checkpoint_cb.best, checkpoint_cb.kth_best_model_path 


    project_name = name+"v19sumSwish" #the name that will appear in wandb

    #value, model_path = train_digital_twin(name+'initial', input_dim, output_dim, [200,400],3,1, 0.0031) #run it once
    #value, model_path = train_digital_twin(name+'initial2', input_dim, output_dim, [], 0.0031) #run it once
    #print('start nas')
    #now perform the NAS loop
    NAS_name = project_name
    sampler = optuna.samplers.RandomSampler() #choose the random sampler
    study = create_study(NAS_name, sampler=sampler) #create the NAS with the chosen sampler

    def objective(trial):
        Nlayers = trial.suggest_categorical("Nlayers", [1, 2, 3, 4, 5])
        
        Ncoms = trial.suggest_categorical("Ncoms", [1, 2, 3, 4, 5])
        identity_init = trial.suggest_categorical("identity_init", [0, 1])
        
        lr = trial.suggest_categorical("lr", [0.0031, 0.000456, 0.01, 0.001, 0.0001, 0.00001])
        
        Nunits = []
            
        if Nlayers == 2:
            Nunits.append(int(trial.suggest_loguniform("Nunits1", 150, 1500)))
            
        if Nlayers == 3:
            Nunits.append(int(trial.suggest_loguniform("Nunits1", 150, 1500)))
            Nunits.append(int(trial.suggest_loguniform("Nunits2", 150, 1500)))
            
        if Nlayers == 4:
            Nunits.append(int(trial.suggest_loguniform("Nunits1", 150, 1500)))
            Nunits.append(int(trial.suggest_loguniform("Nunits2", 150, 1650)))
            Nunits.append(int(trial.suggest_loguniform("Nunits3", 150, 1250)))
        if Nlayers == 5:
            Nunits.append(int(trial.suggest_loguniform("Nunits1", 150, 1500)))
            Nunits.append(int(trial.suggest_loguniform("Nunits2", 150, 1650)))
            Nunits.append(int(trial.suggest_loguniform("Nunits3", 150, 1250)))
            Nunits.append(int(trial.suggest_loguniform("Nunits4", 150, 1250)))

        name = f"v{trial.number}" #create name with trial index
        value, model_path = train_digital_twin(name, input_dim, output_dim, Nunits,Ncoms,identity_init, lr) #do the training!
        trial.set_user_attr('model_path', model_path) #save the checkpoint model path string in NAS study
        return value

    study.optimize(objective, n_trials=Ntrials) #run 10 trials

    best_model_path = study.best_trial.user_attrs['model_path']

    #best_model = MLP_Reg.load_from_checkpoint(best_model_path)
    #print('saving model ' + NAS_name)
    #torch.save(best_model, savename) #SAVES THE MLP_reg!! 

    digital_twin = MLP_Reg.load_from_checkpoint(best_model_path)#torch.load(savename, map_location="cpu")

    x, y = next(iter(val_loader))
    with torch.no_grad():
        y_pred = digital_twin(x)

    plot_grid(y_pred, y)
    plt.savefig(f"monitor/{NAS_name}.png", bbox_inches="tight", dpi=500) 
    plt.close() 
    
    return best_model_path