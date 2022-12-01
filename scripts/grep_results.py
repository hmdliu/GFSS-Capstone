
"""
Usage:
    python scripts/grep_results.py [date] [ignore_args]
    => Show the exp results on that day
"""

import os
import sys

res_list = []
dmp_list = []
dir_list = os.listdir(f"./results/{sys.argv[1]}")

for dir in sorted(dir_list):

    log_path = os.path.join(f'./results/{sys.argv[1]}', dir, 'output.log')

    with open(log_path, 'r') as f:
        lines = f.read().split('\n')
    exp_id = f'{sys.argv[1]}_{dir}'
    exp_dict = eval(lines[2].replace('exp_dict: ', ''))
    for key in sys.argv[2:]:
        del exp_dict[key]
    res = [l.split(': ')[-1].strip() for l in lines if l.find('pred0') != -1]
    val_num = len(res) // 6

    if val_num == 0:
        dmp_list.append(','.join(['TBD'] * 9 + [str(val_num), exp_id]))
        res_list.append((dir, val_num, 'TBD', exp_dict))
        continue
    else:
        max_mIoU = res[-6].split()
        avg_mIoU = res[-4].split()
        smt_mIoU = res[-2].split()
        dmp_list.append(','.join([
            smt_mIoU[1], smt_mIoU[4], smt_mIoU[7],
            avg_mIoU[1], avg_mIoU[4], avg_mIoU[7],
            max_mIoU[1], max_mIoU[4], max_mIoU[7],
            str(val_num), exp_id
        ]))    
        res_list.append((dir, val_num, res[-2], exp_dict))

print('\n[csv]:')
print('\n'.join(dmp_list))

print('\n[summary]:')
for dir, ep, miou, cfg in res_list:
    print(dir.ljust(10), f'val{ep:02d}'.ljust(10), str(miou).ljust(48), cfg)
