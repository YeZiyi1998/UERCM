import os
results = os.listdir('results')
for dir in results:
    if 'w' in dir:
        continue
    try:
        with open('results/'+dir+'/all.txt') as f:
            lines = f.readlines()
            auc = float(lines[0].strip().split(':')[1])
            print(dir, auc)
    except:
        continue

