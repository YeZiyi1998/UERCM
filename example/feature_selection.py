import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import json
import jieba

parser = argparse.ArgumentParser()
parser.add_argument("--filter", default = True, type = bool)
parser.add_argument("--feature_selection", default = 'erp_lowered', type = str, choices = ['old', 'old_not_average','erp_lowered_not_average', 'erp_lowered', 'origin', 'origin_not_average', 'select', 'select_not_average'])
parse_result = parser.parse_args()


## feature selection
def my_percentile(base):
    return [np.percentile(base, 0), np.percentile(base, 25), np.median(base), np.percentile(base, 75), np.percentile(base, 100), np.mean(base)]

def erp_lowered(data):
    base = data[0:30]
    base = my_percentile(base)
    n1 = data[30:40]
    n1 = my_percentile(n1)
    # n1 = [(n1[i] + n1[i+1]) / 2 for i in range(len(n1)//2)]
    p2 = data[40:65]
    p2 = my_percentile(p2)
    # p2 = [(p2[i] + p2[i+1]) / 2 for i in range(len(p2)//2)]
    n4 = data[65:95]
    n4 = my_percentile(n4)
    # n4 = [(n4[i] + n4[i+1]) / 2 for i in range(len(n4)//2)]
    p6 = data[95:125]
    p6 = my_percentile(p6)
    # p6 = [(p6[i] + p6[i+1]) / 2 for i in range(len(p6)//2)]
    return p2 + n4 + p6

def select(data):
    p2 = data[40:65]
    n4 = data[65:95]
    p6 = data[95:125]
    return p2 + n4 + p6

def feature_selection(data_dic, method):
    re = []
    for idx, word in enumerate(data_dic):
        re.append(feature_selection_word(word[-1], method))
    if 'not_average' in method:
        return re
    re_mean = np.mean(re, axis = 0).tolist()
    if len(re) > 1:
        re_std = np.std(re, axis = 0).tolist()
    elif len(re) == 1:
        re_std = [1 for i in range(len(re[0]))]
    else:
        re_std = []
    for idx in range(len(re)):
        re[idx] = [(re[idx][i] - re_mean[i]) / re_std[i] for i in range(len(re_mean))]
    return re

def feature_selection_word(data, method):
    band_base_len = 3 * 8
    if len(data) != 375 + band_base_len:
        print('data len error')
        input()
    if 'old' in method:
        return data[band_base_len:]
    if 'origin' in method:
        psd, central, r_temporal, parietal = data[0:band_base_len], data[band_base_len:band_base_len+125], data[band_base_len+125:band_base_len+250], data[band_base_len+250:band_base_len+375]
        central = select(central)
        r_temporal = select(r_temporal)
        parietal = select(parietal)
        return psd + central + r_temporal + parietal
    if 'erp_lowered' in method:
        psd, central, r_temporal, parietal = data[0:band_base_len], data[band_base_len:band_base_len+125], data[band_base_len+125:band_base_len+250], data[band_base_len+250:band_base_len+375]
        central = erp_lowered(central)
        parietal = erp_lowered(parietal)
        return psd + central + parietal
    if 'select' in method:
        psd, central, r_temporal, parietal = data[0:band_base_len], data[band_base_len:band_base_len+125], data[band_base_len+125:band_base_len+250], data[band_base_len+250:band_base_len+375]
        central = select(central)
        parietal = select(parietal)
        return psd + central + parietal

def get_file(f_i, method = 'erp_lowered', filter = False):
    base_path = '../dataset/processed_eeg/' + str(f_i) + '.json'
    re_list = {}
    with open(base_path) as f:
        lines = f.readlines()
        rank_list = json.loads(lines[0])
        semantic_list = json.loads(lines[1])
        for key in rank_list.keys():
            rank_features = feature_selection(rank_list[key], method)
            if semantic_list[key]['dtype'] != 'truth':
                re_list[key] =  0
            else:
                re_list[key] =  1
            for word_idx in range(len(rank_list[key])):
                rank_list[key][word_idx][-1] = rank_features[word_idx]
        
    filter_rank_list = rank_list.copy()
    if filter:
        for key in semantic_list.keys():
            if semantic_list[key]['dtype'] != 'truth':
                if key in filter_rank_list.keys():
                    del filter_rank_list[key]
    return filter_rank_list, re_list

total_rank_list = {}
re_list = {}
for f_i in range(21):
    total_rank_list[str(f_i)], re_list[str(f_i)] = get_file(f_i, parse_result.feature_selection, parse_result.filter)
fw = open('../UERCM/tmp_data/' + parse_result.feature_selection + '_' + str(parse_result.filter),'w')
fw.write(json.dumps(total_rank_list))
fw.write('\n')
fw.write(json.dumps(re_list))
fw.close()
