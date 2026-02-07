#!/usr/bin/env python
# -*- coding=utf-8 -*-

import argparse
import torch

from runner import Runner

def get_args():
    parser = argparse.ArgumentParser()
    
    ## required
    parser.add_argument("-r", "--run_name", type=str, required=False)
    parser.add_argument("-d", "--run_dir",  type=str, required=False, default='.runs')
    parser.add_argument("-p", "--phase", default ='train', required=False)
    parser.add_argument("-C", "--config",  default='./config.yaml')

    return parser.parse_args()
##onif

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    args = get_args()
    Runner(args).run()