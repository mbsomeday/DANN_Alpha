import argparse, random, torch, os, datetime
import numpy as np

from experiments.DANN_method import DANN_Trainer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=['D1'])
    parser.add_argument('--target', default=['D2'])

    parser.add_argument('--batch_size', default=4)

    parser.add_argument('--src_epochs', default=50)
    parser.add_argument('--adapt_epochs ', default=50)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--monitored_metric', default='loss')
    parser.add_argument('--min_train_epoch', default=30)

    parser.add_argument('--model_path', type=str, default='./model')

    args = parser.parse_args()

    return args




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

args = get_args()
dann_cls = DANN_Trainer(args)
dann_cls.dann()


end_time = datetime.datetime.now()
duration = end_time - start_time
print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
print("Duration: " + str(duration))




















