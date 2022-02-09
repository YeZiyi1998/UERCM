import os
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
import json
import demjson
import numpy as np
import random
from scipy import stats
import os

# CVOQ_lr0.005_ba16

base_path = 'results/'
target_dir_list = os.listdir(base_path)
target_dir = 'LOPO_lr0.005_ba8'
# target_dir = 'CVOQ_lr0.005_ba16'
strategy = 'CVOQ'
valid_number_dict = {'CVOQ':10, 'LOPO':20}
if 'CVOQ' in target_dir:
    strategy = 'CVOQ'
elif 'LOPO' in target_dir:
    strategy = 'LOPO'
valid_number = valid_number_dict[strategy]

total_list = {}

for valid_id in range(valid_number):
    with open(base_path + target_dir + f'/{valid_id}.txt') as f:
        lines = f.readlines()
        y_pred = json.loads(lines[0])
        y_true = json.loads(lines[1])
        f_s = open(f'{strategy}_s/{valid_id}.txt')
        lines_s = f_s.readlines()
        s_info = demjson.decode(lines_s[0])
        uid_list = demjson.decode(lines_s[2])
        for i in range(len(s_info)):
            uid = int(uid_list[i]) % 3
            sid = s_info[i]
            if sid not in total_list.keys():
                total_list[sid] = {}
            if uid not in total_list[sid].keys():
                total_list[sid][uid] = [y_true[i]]
            total_list[sid][uid].append(y_pred[i])

map_list = []
y_true_list = []
y_pred_list = []

for sid in total_list.keys():
    items = total_list[sid].values()
    y_true = [item[0] for item in items]
    y_true_list += y_true
    y_pred = [np.mean(item[1:]) for item in items]
    y_pred_list += y_pred
    map_list.append(average_precision_score(y_true, y_pred))

print("auc: ", roc_auc_score(y_true_list, y_pred_list))
print("map: ", np.mean(map_list))

random_map_list = []
for i in range(5000):
    random_map_list.append(average_precision_score([1,0,0],[random.random(), random.random(), random.random()]))

print("t test", stats.ttest_ind(map_list, random_map_list))
print(map_list)