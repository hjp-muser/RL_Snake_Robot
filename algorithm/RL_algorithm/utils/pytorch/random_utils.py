import numpy as np
import random
import os


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0
    myseed = i  + 1000 * rank if i is not None else None

    try:
        import torch
        torch.manual_seed(myseed)
        torch.cuda.manual_seed(myseed)
        torch.cuda.manual_seed_all(myseed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

    os.environ['PYTHONHASHSEED'] = str(myseed)
    np.random.seed(myseed)
    random.seed(myseed)

