# Incremental Learning with Weight Aligning 
pytorch implementation of "Maintaining Discrimination and Fairness in Class Incremental Learning" from https://arxiv.org/abs/1911.07053

# Dataset 
Download Cifar100 dataset from https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

Put meta, train, test into ./cifar-100-python

# Get Started
## Environment
* Python 3.6+
* torch 1.3.1
* torchvision 0.4.2
* CUDA 10.0 & cudnn 7.6.4
* argparse

# Basic Install
```
pip install -r requirements.txt
```

# Usage
```
python main.py
```

# Result
| **#classes**       | **20** | **40** | **60** | **80** | **100** |
| ------------------ | ------ | ------ | ------ | ------ | ------- |
| 原文**(CE+KD)**    | 83.5   | 72.8   | 60.1   | 49.9   | 42.9    |
| 实现**(CE+KD)**    | 82.5   | 70.9   | 59.2   | 48.7   | 44.2    |
| 原文**(CE+KD+WA)** | 83.5   | 75.5   | 68.7   | 63.1   | 59.2    |
| 实现**(CE+KD+WA)** | 83.3   | 71.1   | 67.8   | 64.9   | 58.7    |

# Reference
* https://github.com/sairin1202/BIC

# TODO
- [ ] Add weight clipping