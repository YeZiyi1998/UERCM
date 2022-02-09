import torch
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, ndcg_score, roc_auc_score, precision_score, f1_score


class BaseRunner(object):
    def __init__(self, model, train_loader, valid_loader, device, loss_module, optimizer, print_interval=30, batch_size= 8):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.loss_module = loss_module
        self.print_interval = print_interval
        self.batch_size = batch_size
        self.optimizer = optimizer 
    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

class SupervisedRunner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super(SupervisedRunner, self).__init__(*args, **kwargs)

    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        epoch_metrics = {}
        for i, batch in enumerate(self.train_loader):
            X, padding_masks, w_labe, s_label = batch
            predictions = self.model(X, padding_masks)
            # print(X, padding_masks, predictions)
            # input()
            loss = self.loss_module(predictions, s_label)  

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            with torch.no_grad():
                total_samples += self.batch_size
                epoch_loss += loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        epoch_metrics['epoch'] = epoch_num
        epoch_metrics['loss'] = epoch_loss
        print('Epoch {:.1f}, loss {:.5f}'.format(epoch_num, epoch_loss), end = ' ')
        return epoch_metrics

    def evaluate(self, ):
        self.model = self.model.eval()
        total_Y = []
        total_predictions = []
        for i, batch in enumerate(self.valid_loader):
            X, padding_masks, w_labe, s_label = batch
            predictions = self.model(X.to(self.device), padding_masks)
            predictions = predictions.cpu().detach().numpy()
            s_label = s_label.cpu().detach().numpy()   
            total_Y += s_label.tolist()
            total_predictions += [item[1] for item in predictions]
        auc = roc_auc_score(total_Y, total_predictions)
        return auc, total_predictions, total_Y
            
class SupervisedWordRunner(BaseRunner):
    def __init__(self, *args, **kwargs):
        super(SupervisedWordRunner, self).__init__(*args, **kwargs)

    def train_epoch(self, epoch_num=None):
        self.model = self.model.train()
        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        epoch_metrics = {}
        for i, batch in enumerate(self.train_loader):
            X, padding_masks, w_label, s_label = batch
            predictions = self.model(X, padding_masks)
            padding_masks = padding_masks.view((-1,1)).squeeze()
            predictions = predictions.view((-1,predictions.shape[-1]))[padding_masks]
            w_label = w_label.view((-1,1))[padding_masks].squeeze()
            goon_flag = True
            try:
                if len(w_label) == 0:
                    goon_flag = False
            except:
                goon_flag = False
            if goon_flag == False:
                continue
            loss = self.loss_module(predictions, w_label)  
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            with torch.no_grad():
                total_samples += self.batch_size
                epoch_loss += loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        epoch_metrics['epoch'] = epoch_num
        epoch_metrics['loss'] = epoch_loss
        print('Epoch {:.1f}, loss {:.5f}'.format(epoch_num, epoch_loss), end = ' ')
        return epoch_metrics

    def evaluate(self, ):
        self.model = self.model.eval()
        total_Y = []
        total_predictions = []
        for i, batch in enumerate(self.valid_loader):
            X, padding_masks, w_label, s_label = batch
            predictions = self.model(X.to(self.device), padding_masks)
            padding_masks = padding_masks.view((-1,1)).squeeze()
            predictions = predictions.view((-1,predictions.shape[-1]))[padding_masks]
            w_label = w_label.view((-1,1))[padding_masks].squeeze()
            predictions = predictions.cpu().detach().numpy()
            w_label = w_label.cpu().detach().numpy()   
            total_Y += w_label.tolist()
            total_predictions += [item[1] for item in predictions]
        auc = roc_auc_score(total_Y, total_predictions)
        return auc, total_predictions, total_Y


