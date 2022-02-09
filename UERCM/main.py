from dataloader import ReadDataLoader, MyDataset
from torch.utils.data import DataLoader
import torch
import argparse
from model import TSTransformerEncoderClassiregressor, TSTransformerEncoderWordClassiregressor
from running import SupervisedRunner, SupervisedWordRunner
import torch.nn as nn
import os
import numpy as np
import time

def main(args):
    data_class = ReadDataLoader(args)
    if os.path.exists('models/' + args.save_dir) == False:
        os.mkdir('models/' + args.save_dir)
        os.mkdir('results/' + args.save_dir)
    if args.strategy == 'LOPO':
        args.valid_number = 21
    data_class.load_data()
    best_auc_list = []
    for valid_id in range(args.valid_number):
        train, valid = data_class.split_dataset(valid_id, args.strategy, valid_info = {'valid_number':args.valid_number})
        train_dataset = MyDataset(train, torch.device(f'cuda:{args.cuda}'))
        valid_dataset = MyDataset(valid, torch.device(f'cuda:{args.cuda}'))
        train_loader = DataLoader(dataset=train_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,)
        valid_loader = DataLoader(dataset=valid_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,)
        feat_dim = train[0][0].shape[1]
        
        loss_module = nn.CrossEntropyLoss()
        
        if args.target == 's':
            model = TSTransformerEncoderClassiregressor(feat_dim = feat_dim, max_len = 15, d_model = args.dmodel, n_heads = 8, num_layers = 1, dim_feedforward = 128, num_classes = 2,)
            model.to(torch.device(f'cuda:{args.cuda}'))
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, )
            runner = SupervisedRunner(model, train_loader, valid_loader, torch.device(f'cuda:{args.cuda}'), loss_module, optimizer, print_interval=30,batch_size=args.batch_size)
        else:
            model = TSTransformerEncoderWordClassiregressor(feat_dim = feat_dim, max_len = 15, d_model = args.dmodel, n_heads = 8, num_layers = 1, dim_feedforward = 128, num_classes = 2,)
            model.to(torch.device(f'cuda:{args.cuda}'))
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, )
            runner = SupervisedWordRunner(model, train_loader, valid_loader, torch.device(f'cuda:{args.cuda}'), loss_module, optimizer, print_interval=30,batch_size=args.batch_size)
        best_auc = 0
        best_predictions = []
        early_stop = 20
        early_stop_num = 0
        for i in range(args.num_epochs):
            time_start=time.time()
            runner.train_epoch(i)
            auc, total_predictions, total_Y = runner.evaluate()
            time_end = time.time()
            time_cost = time_end - time_start
            print('validation auc: {:.2f}, time cost: {:.3f}'.format(auc,time_cost))
            if auc > best_auc:
                early_stop_num = 0
                torch.save(model, os.path.join('models/' + args.save_dir, f'{valid_id}.pkl'))
                best_auc = auc
                best_predictions = total_predictions
            else:
                early_stop_num += 1
            if early_stop_num > early_stop:
                break
        with open(os.path.join('results/' + args.save_dir, f'{valid_id}.txt'), 'w') as fw:
            fw.write(str(best_predictions))
            fw.write('\n')
            fw.write(str(total_Y))
        best_auc_list.append(best_auc)
    with open(os.path.join('results/' + args.save_dir, 'all.txt'), 'w') as fw:
        fw.write('mean auc: '+ str(np.mean(best_auc_list)))
        fw.write('\n')
        fw.write('auc list: ' + str(best_auc_list))
    print('mean auc:', np.mean(best_auc_list))
    print('auc list:', best_auc_list)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-strategy', type=str, help='Training strategy.', default = 'CVOQ', choices = ['ALL','PALL','LOPO','CVOQ','PCVOQ'])
    parser.add_argument('-target', type=str, default = 's', choices = ['s','w'])
    parser.add_argument('-valid_number', type=int,default = 10, required=False)
    parser.add_argument('-cuda', type=int,default = 0, required=False)
    parser.add_argument('-batch_size', type=int,default = 8, required=False)
    parser.add_argument('-lr', type=float,default = 1e-3, required=False)
    parser.add_argument('-erp_type', type=str, default = 'erp_lowered_False', required=False)
    parser.add_argument('-dmodel', type=int, default = 8, required=False)
    parser.add_argument('-save_dir', type=str,default = 'tmp', required=False)
    parser.add_argument('-num_epochs', type=int,default = 50, required=False)
    args = parser.parse_args() 
    main(args)





