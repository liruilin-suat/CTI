import os
import argparse
import time
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from HDR import HDR
from util.config import ModelConf
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Global seed set to: {seed}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommendation Model with Custom Config")
    parser.add_argument(
        '-c', '--config', 
        type=str, 
        default='HyperCTI_symmap.conf', 
        help='Name or path of the configuration file (default: HyperCTI_tcmsuite.conf)'
    )
    return parser.parse_args()

if __name__ == '__main__':
    seed_everything()
    args = parse_args()
    conf_file = args.config
    print(f"Loading config file: {conf_file}")
    s = time.time()
    #Register your model here and add the conf file into the config directory
    try:
        conf = ModelConf(conf_file)
    except KeyError:
        print('wrong num!')
        exit(-1)
           
    print("config:", conf.config)

    recSys = HDR(conf)
    recSys.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
