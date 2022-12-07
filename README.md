## TL;DR

CILIATE is a Class-based Incremental Learning (CIL) model automatic repairing tool, which can help identify important samples and train the model using the debiased training method on these samples to fix model fairness bugs.
It does not require modifying model training protocol or architecture.

## Repo Structure

1. `data`: It contains three datasets we used in this paper. 

2. `main.py`: It contains the source codes of this work and a demo. The way to run the demo cases has been shown [here](#setup).

3. misc: The `README.md` shows how to use our demos, the repo structure, the way to reproduce our experiments and our experiment results. And the `requirement.txt` shows all the dependencies of this work.

```
- data/
- cars.py
- cifar100.py
- dataset.py
- flowers.py
- imbalanced.py
- main.py
- model.py
- README.md
- requirements.txt
- trainer.py
```




## <span id="setup">Setup</span>
### (Recommended) Create a virtual environment
CILIATE requires specific versions of some Python packages which may conflict with other projects on your system. A virtual environment is strongly recommended to ensure the requirements may be installed safely.

### Install with `pip`
To install the requirements, run:

`pip install -r ./requirements.txt`

Note: The version of Pytorch in this requirements is CPU version. If you need GPU version, please check your CUDA version and [install Pytorch manually](https://pytorch.org/).

### Run demo
for cifar-100 dataset:
```shell
python main.py 
```

for flowers dataset:
```shell
python main.py --dataset flowers --total_cls 102
```

for cars dataset:
```shell
python main.py --dataset cars --total_cls 196
```

The optional Args are:

| Argument     | Help                                                    | Default                                                      |
| ------------ | ------------------------------------------------------- | ------------------------------------------------------------ |
| --dataset    | (Char) Choose dataset. Include "cifar100", "flowers" and "cars" | cifar100                                                       |
| --epoch      | (Int) Training epoch for network                              | 250                                                           |
| --batch-size | (Int) Batch size of dataloader                                | 32                                                          |
| --lr   | (Float) Learning rate         | 0.1                 |
| --ita   | (Float) Hyperparameter $\eta$ of CILIATE                                 | 0.1     |
| --total_cls       | (Int) Total classes of dataset     | 100     |

