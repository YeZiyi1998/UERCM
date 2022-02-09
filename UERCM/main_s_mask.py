from dataloader_s_mask import ReadDataLoader, MyDataset
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
    data_class.load_data()
    best_auc_list = []
    if args.strategy == 'LOPO':
        args.valid_number = 21
    for valid_id in range(args.valid_number):
        train, valid = data_class.split_dataset(valid_id, args.strategy, valid_info = {'valid_number':args.valid_number})
        with open(f'{args.strategy}_s/{valid_id}.txt', 'w') as f:
            f.write(str([item[2] for item in valid]))
            f.write('\n')
            f.write(str([item[3] for item in valid]))
            f.write('\n')
            f.write(str([item[4] for item in valid]))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-strategy', type=str, help='Training strategy.', default = 'CVOQ', choices = ['ALL','PALL','LOPO','CVOQ','PCVOQ'])
    parser.add_argument('-target', type=str, default = 's', choices = ['s','w'])
    parser.add_argument('-valid_number', type=int,default = 10, required=False)
    parser.add_argument('-cuda', type=int,default = 0, required=False)
    parser.add_argument('-batch_size', type=int,default = 8, required=False)
    parser.add_argument('-lr', type=float,default = 1e-3, required=False)
    parser.add_argument('-erp_type', type=str, default = 'erp_lowered_False', required=False)
    parser.add_argument('-save_dir', type=str,default = 'tmp', required=False)
    parser.add_argument('-num_epochs', type=int,default = 1, required=False)
    args = parser.parse_args() 
    main(args)





