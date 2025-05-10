"""
Insert all the custom plmodels that are used in this file!
"""
from .pl_models import *
from .pl_utils import *
from .mlp import *
import torch.optim as optim

class MLP_Reg(RegressionModel):
    """Vanilla multilayer perceptron model for regression."""

    def __init__(self, input_dim=None, output_dim=None, Nunits=None, lr=None):
        super().__init__()
        self.save_hyperparameters()  # this appends all the inputs into self.hparams
        for (i, Nunit) in enumerate(Nunits):  # writing more attributes to hyperparams that will appear on wandb
            self.hparams[f'ldim_{i}'] = Nunit
        self.hparams['Nlayers'] = len(Nunits)  # repeat - adding Nlayers
        self.model = MLP(input_dim, output_dim, Nunits)  # Multilayer Perceptron pytorch model

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return optimizer
    
class MLP_TL_Reg(RegressionModel):
    """
    Vanilla multilayer perceptron model for regression,
    with an additional transfer learning (TL) feature for the
    optimization. 
    """

    def __init__(self, input_dim=None, output_dim=None, Nunits=None, lr=None):
        super().__init__()
        self.save_hyperparameters()  # this appends all the inputs into self.hparams
        for (i, Nunit) in enumerate(Nunits):  # writing more attributes to hyperparams that will appear on wandb
            self.hparams[f'ldim_{i}'] = Nunit
        self.hparams['Nlayers'] = len(Nunits)  # repeat - adding Nlayers
        self.model = MLP(input_dim, output_dim, Nunits)  # Multilayer Perceptron pytorch model

    def forward(self, data):
        return self.model(data)

    def configure_optimizers(self):
        optimizer = optim.Adam(get_parameters(self.model.layers[-1::]), lr=self.hparams.lr)
        return optimizer
    

#defining the NN model
class ManifoldModel(nn.Module):
    def __init__(s, flist, dims):
        super().__init__() 
        s.factors = nn.ParameterList()
        s.offsets = nn.ParameterList()
        
        s.dims = dims
        nlayers = len(flist)
        for dim in dims[:-1]:
            s.factors.append(nn.Parameter(0.9*torch.ones(dim))) 
            s.offsets.append(nn.Parameter(torch.zeros(dim)))
    
        s.A = nn.Parameter(torch.randn(dims[-1])) #for the RC
        s.b = nn.Parameter(torch.tensor(0.0)) #for the RC
        s.flist = flist
        s.xPLMs = []
    
    def forward(s, x, save=False):
        x = torch.repeat_interleave(x, dim=1, repeats=int(s.dims[0]/12))
        if save:
            s.xin = []
            s.xout = []
            
        for (l, f) in enumerate(s.flist):
            x = x*s.factors[l]+s.offsets[l] #manifold
            s.xPLMs.append(x)
            if save:
                s.xin.append(x.detach())
            x = x.clamp(0.0, 1.0)
            x = f(x)
            if save:
                s.xout.append(x.detach())
        return torch.sum(s.A*x, dim=1) + s.b
    
    
class ManifoldPlModel(Classification1DModel):
    def __init__(self, flist, dims, lr):
        super().__init__()
        
        self.model = ManifoldModel(flist, dims)
        
        self.save_hyperparameters()
        #delete flist since it is a function which cannot be saved as JSON and will cause a bug...
        del self.hparams["flist"] 
        # writing more attributes to hyperparams that will appear on wandb
        for (i, Nunit) in enumerate(dims):
            self.hparams[f'ldim_{i}'] = Nunit
        self.hparams['Nlayers'] = len(dims)  #adding Nlayers
        
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer