import os

# for batch_size in [8,16,32,64]:
#     for lr in [1e-3, 1e-4, 5e-4, 5e-3]:
#         os.system(f'python3 main.py -cuda 0 -lr {lr} -batch_size {batch_size} -strategy CVOQ -save_dir CVOQ_lr{lr}_ba{batch_size}')

# for batch_size in [8,16,32,64]:
#     for lr in [1e-3, 1e-4, 5e-4, 5e-3]:
#         os.system(f'python3 main.py -cuda 1 -lr {lr} -batch_size {batch_size} -strategy LOPO -save_dir LOPO_lr{lr}_ba{batch_size}')

for batch_size in [32,64]:
    for lr in [1e-3, 1e-2, ]:
        for dmodel in [8,128]:
            if os.path.exists(f'results/w_LOPO_lr{lr}_ba{batch_size}_dmodel{dmodel}') == False: 
                os.system(f'python3 main.py -cuda 1 -lr {lr} -batch_size {batch_size} -strategy LOPO -dmodel {dmodel} -save_dir w_LOPO_lr{lr}_ba{batch_size}_dmodel{dmodel}')

# idx = 0
# for batch_size in [8,32]:
#     for lr in [1e-3, 1e-4, 5e-4, 5e-3]:
#         for dmodel in [8,32,64]:
#             idx += 1
#             if idx >= 3: 
#                 os.system(f'python3 main.py -cuda 1 -lr {lr} -batch_size {batch_size} -strategy CVOQ -dmodel {dmodel} -save_dir w_CVOQ_lr{lr}_ba{batch_size}')

# for batch_size in [8,16,32,64]:
#     for lr in [1e-3, 1e-4, 5e-4, 5e-3]:
#         os.system(f'python3 main.py -cuda 1 -lr {lr} -batch_size {batch_size} -strategy LOPO -save_dir LOPO_lr{lr}_ba{batch_size}')
