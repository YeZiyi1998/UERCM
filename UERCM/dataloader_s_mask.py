import json
from torch.utils.data import Dataset
import torch
import numpy as np

# Todo: normalize

class ReadDataLoader():
    def __init__(self,args):
        self.user_name_list = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
        self.base_path = '../../cikm/src/classification/answer_extraction/tmp_data/'
        self.args = args
        self.valid_id = 0
        self.data_shape = 0
        self.sentence_length = 15

    def load_data(self,):
        f = open(self.base_path + self.args.erp_type)
        lines = f.readlines()
        self.re_list = json.loads(lines[1])
        self.total_list = json.loads(lines[0])

    def split_dataset(self, valid_id, strategy, valid_info = None):
        train, valid = [], []
        # data mask label1, label2
        if strategy == "LOPO":
            for idx, user_name in enumerate(self.user_name_list):   
                if idx == valid_id:
                    tmp = valid
                elif idx != valid_id:
                    tmp = train
                for sid in self.total_list[user_name].keys():
                    if self.data_shape == 0:
                        self.data_shape = len(self.total_list[user_name][sid][0][2])
                    base_data = np.zeros((self.sentence_length, self.data_shape))
                    base_mask = np.zeros(self.sentence_length)
                    slabel = self.re_list[user_name][sid]
                    wlabel = np.zeros(self.sentence_length)
                    for item in self.total_list[user_name][sid]:
                        poi = item[1]
                        if poi >= 15:
                            continue
                        if item[0] == 3:
                            wlabel[poi] = 1
                        else:
                            wlabel[poi] = 0
                        base_mask[poi] = 1
                        base_data[poi] = self.total_list[user_name][sid][0][2]
                    if np.sum(base_mask) > 0:
                        tmp.append([base_data, base_mask, sid, slabel, user_name])

        elif strategy == "CVOQ":
            valid_number = valid_info['valid_number']
            for idx, user_name in enumerate(self.user_name_list):   
                for sid in self.total_list[user_name].keys():
                    if int(sid) % valid_number == valid_id:
                        tmp = valid
                    elif int(sid) % valid_number != valid_id:
                        tmp = train
                    if self.data_shape == 0:
                        self.data_shape = len(self.total_list[user_name][sid][0][2])
                    base_data = np.zeros((self.sentence_length, self.data_shape))
                    base_mask = np.zeros(self.sentence_length)
                    slabel = self.re_list[user_name][sid]
                    wlabel = np.zeros(self.sentence_length)
                    for item in self.total_list[user_name][sid]:
                        poi = item[1]
                        wlabel[poi] = item[0]
                        base_mask[poi] = 1
                        base_data[poi] = self.total_list[user_name][sid][0][2]
                    if np.sum(base_mask) > 0:
                        tmp.append([base_data, base_mask, sid, slabel, user_name])
        print("len(train), len(valid) ", len(train), len(valid))
        return train, valid

class NonClickDataLoader():
    def __init__(self,args):
        self.user_name_list = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','pilot2',]
        # self.user_name_list = ['0']
        self.base_path = '/work/yeziyi/EEG-classification/dataset/Non-click/features_info/'
        self.args = args
        self.valid_id = 0

    def load_data(self,):
        self.data = []
        for user_name in self.user_name_list:
            q2d2f = json.load(open(self.base_path+user_name+'_5_gamma_high_temporal_type_rank.json'))
            for q in q2d2f.keys():
                for d in q2d2f[q].keys():
                    self.data.append([{'user_name':user_name,'q':q,'d':d}, q2d2f[q][d]])
        self.q_set_list = list(set([self.data[i][0]['q'] for i in range(len(self.data))]))
        self.q_set_list = sorted(self.q_set_list)

    def split_dataset(self, valid_id, strategy, valid_info = None):
        if strategy == "LOPO":
            user_name = self.user_name_list[valid_id]
            train = [self.data[item] for item in range(len(self.data)) if self.data[item][0]['user_name'] != user_name]
            valid = [self.data[item] for item in range(len(self.data)) if self.data[item][0]['user_name'] == user_name]
        elif strategy == "CVOQ":
            valid_number = valid_info['valid_number']
            train = [self.data[item] for item in range(len(self.data)) if self.q_set_list.index(self.data[item][0]['q']) % valid_number != valid_id]
            valid = [self.data[item] for item in range(len(self.data)) if self.q_set_list.index(self.data[item][0]['q']) % valid_number == valid_id]
        elif strategy == "ALL":
            valid_number = valid_info['valid_number']
            train = [self.data[item] for item in range(len(self.data)) if item % valid_number != valid_id]
            valid = [self.data[item] for item in range(len(self.data)) if item % valid_number == valid_id]
        return self.get_eeg_set(train), self.get_eeg_set(valid)

    def get_eeg_set(self, train):
        eeg_XY = []
        for item in train:
            if int(item[-1]['score']) <= 1:
                eeg_XY += [[[channel_data[0:5] for channel_data  in item[-1]['eeg']], 0]]
            elif int(item[-1]['score']) >= 5:
                eeg_XY += [[[channel_data[0:5] for channel_data in item[-1]['eeg']], 1]]
        return eeg_XY

class MyDataset(Dataset):
    def __init__(self, train, device):
        super(MyDataset, self).__init__()
        self.train = train
        self.device = device

    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            mask: (seq_length, feat_dim) boolean tensor: 0s mask and predict, 1s: unaffected input
            ID: ID of sample
        """
        X =  np.array(self.train[ind][0]) 
        mask = np.array(self.train[ind][1], bool)
        w_label = np.array(self.train[ind][2])
        s_label = np.array(self.train[ind][3])
        return torch.from_numpy(X).to(self.device, dtype=torch.float32), torch.from_numpy(mask).to(self.device, dtype=torch.bool), torch.from_numpy(w_label).to(self.device, dtype=torch.long), torch.from_numpy(s_label).to(self.device, dtype=torch.long), 
    
    def __len__(self,):
        return len(self.train)




