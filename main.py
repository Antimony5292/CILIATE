import torch
import numpy as np
from trainer import Trainer
import sys
import argparse

parser = argparse.ArgumentParser(description='Incremental Learning BIC')
parser.add_argument('--batch_size', default = 128, type = int)
parser.add_argument('--dataname', default = 'cifar100', type = str)
parser.add_argument('--epoch', default = 10, type = int)
parser.add_argument('--lr', default = 0.1, type = float)
parser.add_argument('--max_size', default = 2000, type = int)
parser.add_argument('--total_cls', default = 100, type = int)
parser.add_argument('--method', default = 'icarl', type = str)
parser.add_argument('--ita', default = 0.1, type = float)
# parser.add_argument('--is_FN', default = False, type = bool)
args = parser.parse_args()


if __name__ == "__main__":
    trainer = Trainer(args.total_cls, args.dataname)
    trainer.train(args.batch_size, args.epoch, args.lr, args.max_size, args.method, args.ita)
