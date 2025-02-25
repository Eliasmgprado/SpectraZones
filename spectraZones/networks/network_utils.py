import torch
from torch import nn
import numpy as np

class Config:
    """
        Models Configuration
    """
    def __init__(self):
        self.input_dim = None
        self.start_filter = 256
        self.depth = 4
        
        self.lr = 1e-3
        self.lr_w_decay = 1e-5
        self.batch_size = 64

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        self.lr_patience = 10
        self.lr_delta = 1e-4
        self.lr_decay_factor = 1e-1
        self.es_scheduler = EarlyStopping
        self.es_patience = 20
        self.es_delta = 1e-4
        
    def merge(self, config_dict=None):
        if config_dict is None:
            pass
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
            
    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        star_args = {}
        
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_strings.append('%s=%r' % (name, value))
            else:
                star_args[name] = value
        if star_args:
            arg_strings.append('**%s' % repr(star_args))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))
    
    def _get_kwargs(self):
        return list(self.__dict__.items())
    
    
class EarlyStopping:
    """
        Early stops the training if validation loss doesn't improve after a given patience.
    
        Code Modified From:
        https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.trace_func = trace_func
        
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.loss_min = np.Inf
        
    def __call__(self, loss, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, val_loss, model)
            self.best_epoch = model.train_epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose and self.counter%10 == 0:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}' +\
                                f' - LR: {model.optimizer.param_groups[0]["lr"]}')
            if self.counter >= self.patience:
                self.trace_func(f'EarlyStopping training.')
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(loss, val_loss, model)
            self.counter = 0
            self.best_epoch = model.train_epoch

    def save_checkpoint(self, loss, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            
        if getattr(model, "doLog", False):
            model.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving checkpoint to {model.save_path}...')
           
        model.saveModel()
        self.val_loss_min = val_loss
        self.loss_min = loss