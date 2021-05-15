from sacrebleu import corpus_bleu
import random
import torch
import numpy as np

def lmap(f,x):
    """list(map(f, x))"""
    return list(map(f, x))

def calculate_bleu(output_lns, refs_lns):
    """Uses sacrebleu's corpus_bleu implementation."""
    return round(corpus_bleu(output_lns, [refs_lns]).score, 4)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    '''
    将字符转化为bool类型
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
