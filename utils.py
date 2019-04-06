import time
from functools import wraps
import numpy as np
import torch
from torch.autograd import Variable

"""
refer to this repo:
    https://github.com/weixu000/DSH-pytorch
"""

def feed_random_seed(seed=np.random.randint(1, 10000)):
    """feed random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def timing(f):
    """print time used for function f"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        time_start = time.time()
        ret = f(*args, **kwargs)
        print(f'total time = {time.time() - time_start:.4f}')
        return ret

    return wrapper


def compute_result(dataloader, net):
    bs, clses = [], []
    net.eval()
    for img, cls, name in dataloader:
        clses.append(cls)
        bs.append(net(Variable(img.cuda(), volatile=True)).data.cpu())
    return torch.sign(torch.cat(bs)), torch.cat(clses)


@timing
def compute_mAP(trn_binary, tst_binary, trn_label, tst_label):
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pytorch_deephash
    """
    for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    AP = []
    Ns = torch.arange(1, trn_binary.size(0) + 1)
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
        correct = (query_label == trn_label[query_result]).float()
        P = torch.cumsum(correct, dim=0) / Ns
        AP.append(torch.sum(P * correct) / torch.sum(correct))
    mAP = torch.mean(torch.Tensor(AP))
    return mAP