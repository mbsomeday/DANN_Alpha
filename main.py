import argparse, random, torch, os, datetime
import numpy as np

from experiments.DANN_method import DANN_Trainer


def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--source', nargs='+', default=['D1'])
    parser.add_argument('--target', nargs='+', default=['D2'])
    parser.add_argument('--batch_size', default=4, type=int)

    # train
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--min_train_epoch', type=int, default=30)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--monitored_metric', default='loss')

    # callbacks
    parser.add_argument('--top_k', default=2)
    parser.add_argument('--patience', default=10)

    # test
    parser.add_argument('--adapt_test_epoch', type=int, default=5)

    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--seed', default=13)


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
dann_cls.train()
dann_cls.test()


end_time = datetime.datetime.now()
duration = end_time - start_time
print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
print("Duration: " + str(duration))




















