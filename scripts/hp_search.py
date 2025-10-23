# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse

from experiments.hyperparam_search import HPSelection


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--isTrain', action='store_true')

    # train
    parser.add_argument('--hp_dir', type=str, default=r'D:\my_phd\on_git\DANN_Alpha\Results')

    # test
    parser.add_argument('--weight_dir', type=str, default=r'D:\my_phd\on_git\DANN_Alpha\Results\48_0.001_Adam_COS_\3_0.74442')
    parser.add_argument('--test_ds_list', nargs='+', default=['D1'])
    parser.add_argument('--test_txt', type=str, default=r'D:\my_phd\on_git\DANN_Alpha\Results\test_res.txt', help='txt file that records test results')

    args = parser.parse_args()

    return args


args = get_args()
# args.isTrain = True

hp = HPSelection(args)
if args.isTrain:
    hp.hp_search()
else:
    print(f'Current mode: Test')
    hp.test()

















