# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)


import argparse, random, torch, os, datetime
import numpy as np

from experiments.DANN_method import DANN_Trainer


def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--source', nargs='+', default=['D1'])
    parser.add_argument('--target', nargs='+', default=['D2'])

    # train
    parser.add_argument('--monitored_metric', default='loss')
    parser.add_argument('--isTrain', action='store_true')
    parser.add_argument('--seed', default=13)

    # callbacks
    parser.add_argument('--top_k', default=1)
    parser.add_argument('--patience', default=10)

    # test
    parser.add_argument('--test_ds_list', nargs='+', default=None)
    parser.add_argument('--weight_dir', type=str, default='./model')
    parser.add_argument('--test_txt', type=str, default=None, help='txt file that records test results')



    args = parser.parse_args()

    return args


args = get_args()

manual_seed = args.seed
random.seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
os.environ['PYTHONHASHSEED'] = str(manual_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

start_time = datetime.datetime.now()
print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

dann_cls = DANN_Trainer(args)
if args.isTrain:
    dann_cls.train()
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))
else:
    dann_cls.test()













































