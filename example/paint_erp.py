import numpy as np
import mne
import json
import pickle
import ssl
import os
import philistine as phil
from matplotlib import pyplot as plt

def abs_threshold(epochs, threshold,
                  eeg=True, eog=False, misc=False, stim=False):
    data = epochs.pick_types(eeg=eeg, misc=misc, stim=stim).get_data()
    rej = np.any( np.abs(data) > threshold, axis=(-1, -2) )  
    return rej

def get_mask_file(file_index, mode):
    base_file_path = '../dataset/raw_txt/'+str(file_index % 3)+'_process'+mode+'.txt'
    tru_list = []
    with open(base_file_path,'r') as f:
        for line in f.readlines():
            line_list = line.strip().split('\t')
            tru_list.append([line_list[0], json.loads(line_list[1])])
    tru_list = tru_list[0:150]
    return tru_list

change_dic = {'1':'ordinary words','2':'semantic-related words','3':'answer words',}

def key_reflextion(dic1):
    dic2 = {}
    for key in dic1.keys():
        dic2[change_dic[key]] = dic1[key]
    return dic2
channel_choosen = ['Cz','FCz','C3','C4','FC3','FC4']

def epoch_mask(epochs, tru_list):
    word_epoch_list = {}
    for key in change_dic.keys():
        word_epoch_list[key] = []
    
    for i in range(len(tru_list)):
        for j in range(len(tru_list[i][1])):
            id = (i + 1) * 100 + j
            if str(id) not in epochs.event_id:
                continue        
            if str(tru_list[i][1][j]) in word_epoch_list.keys():
                # if str(tru_list[i][1][j]) != '3':
                    # tru_list[i][1][j] = '1'
                word_epoch_list[str(tru_list[i][1][j])].append(str(id))
    return word_epoch_list

global_word = {}

for f_i in range(0,21):
    tru_list = get_mask_file(f_i, '_xl')
    epochs = mne.read_epochs('../dataset/raw_eeg/qa'+str(f_i)+'.fif', preload = True).pick_channels(channel_choosen)
    bad_epoch_mask = phil.mne.abs_threshold(epochs, 50e-6)
    epochs.drop(bad_epoch_mask, reason="absolute threshold")

    word_epoch_list = epoch_mask(epochs, tru_list)
    for cond in word_epoch_list.keys():
        word_epoch_list[cond] = epochs[word_epoch_list[cond]]
        if 0 == f_i:
            global_word[cond] = word_epoch_list[cond].get_data()
        else:
            global_word[cond] = np.concatenate((global_word[cond], word_epoch_list[cond].get_data()), axis =0)

import seaborn as sns

f, ax1 = plt.subplots(figsize=(15,9.5))
figs = mne.viz.plot_compare_evokeds({"answer words":global_word['3'], "semantic-related words":global_word['2'], "ordinary words":global_word['1']}, linestyles = ['solid', 'dotted', 'dashed',],
                styles=key_reflextion({"3": {"linewidth": 4.2},"2": {"linewidth": 4.2},"1": {"linewidth": 4.2},}),colors = ['green',sns.color_palette()[1],'blue'],
                 legend='lower right', 
                 show_sensors=None,
                 combine = 'mean',show=True,axes=ax1)
plt.xticks(fontsize = 32)
plt.xlabel('Time (s)', fontdict={'weight': 'normal', 'size': 28})
plt.yticks(fontsize = 32)
plt.ylabel(r'$\mu$V', fontdict={'weight': 'normal', 'size': 28})
plt.legend(fontsize = 24)
figs[0].savefig('word3'+'.pdf')
