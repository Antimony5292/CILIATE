import torch
import numpy as np
from trainer import Trainer
import sys
import argparse

parser = argparse.ArgumentParser(description='CILIATE')
parser.add_argument('--batch_size', default = 32, type = int)
parser.add_argument('--dataname', default = 'cifar100', type = str)
parser.add_argument('--epoch', default = 250, type = int)
parser.add_argument('--lr', default = 0.1, type = float)
parser.add_argument('--total_cls', default = 100, type = int)
parser.add_argument('--inc_num', default = 5, type = int)
parser.add_argument('--method', default = 'FN', type = str)
parser.add_argument('--ita', default = 0.1, type = float)
parser.add_argument('--loss_name', default = 'JS', type = str)
parser.add_argument('--dropout_state', default = 'selective', type = str)
parser.add_argument('--random_select', default = False, type = bool)
args = parser.parse_args()


if __name__ == "__main__":
    trainer = Trainer(args.total_cls, args.dataname, args.inc_num)
    trainer.train(args.batch_size, args.epoch, args.lr, args.method, args.ita, args.loss_name, args.dropout_state,args.random_select)
