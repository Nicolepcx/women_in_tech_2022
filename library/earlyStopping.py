import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose_es=False, delta=3e-5, path='checkpoint.pt', trace_func=print):
        
        """
        :param: patience (int): How long to wait after last time validation loss improved.
                            Default: 7
        :param: verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False                     
        :param: delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0     
        :param: path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'     
        :param: trace_func (function): trace print function. Default: print      

        """
        self.patience = patience
        self.verbose_es = verbose_es
        self.counter = 0
        self.prev_score = np.inf
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        if val_loss > self.prev_score + self.delta:
            self.counter += 1
            if self.verbose_es:
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

        self.prev_score = val_loss
        self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        
        '''
        
        if self.verbose_es:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
