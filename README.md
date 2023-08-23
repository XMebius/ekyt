# Eye Know You Too

- [Eye Know You Too](#eye-know-you-too)
  - [Overview](#overview)
  - [Citation](#citation)
  - [License](#license)
  - [Contact](#contact)
  - [Setting up a compatible Anaconda environment](#setting-up-a-compatible-anaconda-environment)
  - [Training a model](#training-a-model)
  - [Testing a model (i.e., computing embeddings for later evaluation)](#testing-a-model-ie-computing-embeddings-for-later-evaluation)
  - [Evaluating a model](#evaluating-a-model)
  - [Replicating our published results](#replicating-our-published-results)
    - [Table II and Figures 4a, 4b, 5b](#table-ii-and-figures-4a-4b-5b)
    - [Table III](#table-iii)
    - [Table IV](#table-iv)
    - [Table V](#table-v)
    - [Table VI](#table-vi)
    - [Table VII](#table-vii)
    - [Figure 7](#figure-7)

## Overview

This is source code for our eye movement biometrics model, Eye Know You Too.
Everything needed to train and evaluate our model is included.

## Citation

If you use this code and/or our pre-trained model(s), please cite our
associated publication:

D. Lohr and O. V. Komogortsev, "Eye Know You Too: Toward Viable End-to-End Eye
Movement Biometrics for User Authentication," in _IEEE Transactions on
Information Forensics and Security_, 2022, doi: 10.1109/TIFS.2022.3201369.

## License

This work is licensed under a "Creative Commons Attribution-NonCommercial-
ShareAlike 4.0 International License"
(https://creativecommons.org/licenses/by-nc-sa/4.0/).

Property of Texas State University.

## Contact

If you have any questions or difficulties regarding the use of this code,
please feel free to email the author, Dillon Lohr, at <djl70@txstate.edu>.

## Setting up a compatible [Anaconda](https://www.anaconda.com/) environment

```bash
# 先安装mamba加快速度
conda install mamba -n base -c conda-forge

$ mamba create -n ekyt-release python==3.7.11
$ conda activate ekyt-release   # 这里用mamba的话不能切换

# PyTorch (different setups may require a different version of cudatoolkit)
$ mamba install -c pytorch -c conda-forge pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit==11.3.1

# PyTorch Metric Learning (for multi-similarity loss and, optionally, MAP@R)
$ mamba install -c metric-learning -c pytorch pytorch-metric-learning==0.9.99

# PyTorch Lightning and other necessary packages
$ mamba install -c conda-forge pytorch-lightning==1.5.0 pandas==1.3.4 tensorboard==2.6.0 scikit-learn==1.0.1 numpy==1.21.2 scipy==1.7.1 tqdm==4.62.3

# (optional) For computing MAP@R
$ mamba install -c conda-forge faiss-gpu==1.7.1

# (optional) For plotting figures
$ mamba install -c conda-forge matplotlib==3.4.3 umap-learn==0.5.1

# (optional) For formatting source code
$ mamba install -c conda-forge black flake8
```

## Training a model

If you are only interested in using our provided pre-trained models, you can
skip this section.

```bash
# See all possible command-line arguments for training and testing
$ python train_and_test.py --help
# 需要把X:\miniconda3\envs\ekyt\lib\site-packages\torch\utils\tensorboard\__init__.py第4~7行和10行注释掉，否则会报错

# 输入序列长度(seq_len)
# 降采样因子，对应不同采样率(ds)
# 每个小批处理中的类数(batch_classes)
# 每个类每个小批处理中的序列数(batch_samples)
# multi-similarity loss的权重(w_ms)
# cross-entropy loss的权重(w_ce)
# 是否降低空间精度(degrade_precision)
# 使用哪一个fold作为验证集，其他的作为训练集(fold)
$ python train_and_test.py --mode=train

# (optional) Track the model's progress with Tensorboard
$ tensorboard --logdir=lightning_logs

# Train a full ensemble to enable evaluation
$ python train_and_test.py --mode=train --fold=0
$ python train_and_test.py --mode=train --fold=1
$ python train_and_test.py --mode=train --fold=2
$ python train_and_test.py --mode=train --fold=3

# Other configurations used in our manuscript are given below.  Remember
# to train and test one model for each fold to enable evaluation.

# 500 Hz
$ python train_and_test.py --mode=train --ds=2
# 250 Hz
$ python train_and_test.py --mode=train --ds=4
# 125 Hz
$ python train_and_test.py --mode=train --ds=8
# 50 Hz
$ python train_and_test.py --mode=train --ds=20
# 31.25 Hz
$ python train_and_test.py --mode=train --ds=32

# 125 Hz with spatial precision degraded by 0.5 degrees
$ python train_and_test.py --mode=train --ds=8 --degrade_precision

# Different loss weighting schemes
$ python train_and_test.py --mode=train --w_ms=1.0 --w_ce=0.0
# (default)
# $ python train_and_test.py --mode=train --w_ms=1.0 --w_ce=0.1
$ python train_and_test.py --mode=train --w_ms=1.0 --w_ce=0.2
$ python train_and_test.py --mode=train --w_ms=1.0 --w_ce=0.3
# ...
$ python train_and_test.py --mode=train --w_ms=1.0 --w_ce=0.9
$ python train_and_test.py --mode=train --w_ms=1.0 --w_ce=1.0
$ python train_and_test.py --mode=train --w_ms=0.9 --w_ce=1.0
# ...
$ python train_and_test.py --mode=train --w_ms=0.1 --w_ce=1.0
$ python train_and_test.py --mode=train --w_ms=0.0 --w_ce=1.0
```

## Testing a model (i.e., computing embeddings for later evaluation)

```bash
# See all possible command-line arguments for training and testing
$ python train_and_test.py --help

# Test a model that was trained with default settings.  The first time
# this script is run for a given `--ds` and/or with the flag
# `--degrade_precision`, the dataset will need to be prepared.
$ python train_and_test.py --mode=test

# Test a full ensemble to enable evaluation
$ python train_and_test.py --mode=test --fold=0
$ python train_and_test.py --mode=test --fold=1
$ python train_and_test.py --mode=test --fold=2
$ python train_and_test.py --mode=test --fold=3

# If you have less than 16 GB of VRAM and are working with our pre-
# trained models on 1000 Hz data, you might need to reduce the batch
# size at test time
$ python train_and_test.py --mode=test --batch_size_for_testing=64
```

## Evaluating a model

```bash
# See all possible command-line arguments for evaluation.  Note that we
# evaluate on one task (--task), one round (--round), and one duration
# (--n_seq) at a time.
$ python evaluate.py --help

# Evaluate the ensemble model that was trained with default settings.
# We evaluate under the primary evaluation scenario by default (the
# first 5 seconds during R1 TEX).
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce01_normal

# Evaluate using 60 seconds (12 windows of 5 seconds each) of data
$ python evaluate.py --model=... --n_seq=12

# Create plots like Figures 4a, 4b, 5b
$ python evaluate.py --plot

# Compute bootstrapped results (including FRR @ FAR)
$ python evaluate.py --bootstrap

# Compute results associated with the validation set (from Table V)
$ python evaluate.py --val

# Compute results associated with JuDo1000 (from Table VII)
$ python evaluate.py --judo
```

## Replicating our published results

We provide here the sequence of commands to use if you want to replicate our
published results using our pre-trained models. Running all of these commands
will require approximately 90 GB of storage (approx. 70 GB for datasets and
approx. 20 GB for embeddings).

### Table II and Figures 4a, 4b, 5b

```bash
# Compute subsequence embeddings
$ python train_and_test.py --mode=test --fold=0
$ python train_and_test.py --mode=test --fold=1
$ python train_and_test.py --mode=test --fold=2
$ python train_and_test.py --mode=test --fold=3

# Primary result (also creates the 3 figures, saved to ./figures)
$ python evaluate.py --plot

# Task group
$ python evaluate.py --task=HSS
$ python evaluate.py --task=RAN
$ python evaluate.py --task=FXS
$ python evaluate.py --task=VD1
$ python evaluate.py --task=VD2
$ python evaluate.py --task=BLG

# Test-retest interval group
$ python evaluate.py --round=2
$ python evaluate.py --round=3
$ python evaluate.py --round=4
$ python evaluate.py --round=5
$ python evaluate.py --round=6
$ python evaluate.py --round=7
$ python evaluate.py --round=8
$ python evaluate.py --round=9

# Duration group
$ python evaluate.py --n_seq=2
$ python evaluate.py --n_seq=3
$ python evaluate.py --n_seq=4
$ python evaluate.py --n_seq=5
$ python evaluate.py --n_seq=6
$ python evaluate.py --n_seq=7
$ python evaluate.py --n_seq=8
$ python evaluate.py --n_seq=9
$ python evaluate.py --n_seq=10
$ python evaluate.py --n_seq=11
$ python evaluate.py --n_seq=12
```

### Table III

```bash
# 500 Hz
$ python train_and_test.py --mode=test --fold=0 --ds=2
$ python train_and_test.py --mode=test --fold=1 --ds=2
$ python train_and_test.py --mode=test --fold=2 --ds=2
$ python train_and_test.py --mode=test --fold=3 --ds=2
$ python evaluate.py --model=ekyt_t5000_ds2_bc16_bs16_wms10_wce01_normal

# 250 Hz
$ python train_and_test.py --mode=test --fold=0 --ds=4
$ python train_and_test.py --mode=test --fold=1 --ds=4
$ python train_and_test.py --mode=test --fold=2 --ds=4
$ python train_and_test.py --mode=test --fold=3 --ds=4
$ python evaluate.py --model=ekyt_t5000_ds4_bc16_bs16_wms10_wce01_normal

# 125 Hz
$ python train_and_test.py --mode=test --fold=0 --ds=8
$ python train_and_test.py --mode=test --fold=1 --ds=8
$ python train_and_test.py --mode=test --fold=2 --ds=8
$ python train_and_test.py --mode=test --fold=3 --ds=8
$ python evaluate.py --model=ekyt_t5000_ds8_bc16_bs16_wms10_wce01_normal

# 50 Hz
$ python train_and_test.py --mode=test --fold=0 --ds=20
$ python train_and_test.py --mode=test --fold=1 --ds=20
$ python train_and_test.py --mode=test --fold=2 --ds=20
$ python train_and_test.py --mode=test --fold=3 --ds=20
$ python evaluate.py --model=ekyt_t5000_ds20_bc16_bs16_wms10_wce01_normal

# 31.25 Hz
$ python train_and_test.py --mode=test --fold=0 --ds=32
$ python train_and_test.py --mode=test --fold=1 --ds=32
$ python train_and_test.py --mode=test --fold=2 --ds=32
$ python train_and_test.py --mode=test --fold=3 --ds=32
$ python evaluate.py --model=ekyt_t5000_ds32_bc16_bs16_wms10_wce01_normal
```

### Table IV

```bash
# Requires embeddings for ekyt_t5000_ds1_bc16_bs16_wms10_wce01_normal
# Requires embeddings for ekyt_t5000_ds2_bc16_bs16_wms10_wce01_normal
# Requires embeddings for ekyt_t5000_ds4_bc16_bs16_wms10_wce01_normal
# Requires embeddings for ekyt_t5000_ds8_bc16_bs16_wms10_wce01_normal
# Requires embeddings for ekyt_t5000_ds20_bc16_bs16_wms10_wce01_normal
# Requires embeddings for ekyt_t5000_ds32_bc16_bs16_wms10_wce01_normal

# 1000 Hz, 5 seconds (look at "bootstrapped" results)
$ python evaluate.py --bootstrap

# 1000 Hz, 10 seconds
$ python evaluate.py --bootstrap --n_seq=2

# 1000 Hz, 20 seconds
$ python evaluate.py --bootstrap --n_seq=4

# 1000 Hz, 30 seconds
$ python evaluate.py --bootstrap --n_seq=6

# 1000 Hz, 60 seconds
$ python evaluate.py --bootstrap --n_seq=12

# 500 Hz, 60 seconds
$ python evaluate.py --bootstrap --n_seq=12 --model=ekyt_t5000_ds2_bc16_bs16_wms10_wce01_normal

# 250 Hz, 60 seconds
$ python evaluate.py --bootstrap --n_seq=12 --model=ekyt_t5000_ds4_bc16_bs16_wms10_wce01_normal

# 125 Hz, 60 seconds
$ python evaluate.py --bootstrap --n_seq=12 --model=ekyt_t5000_ds8_bc16_bs16_wms10_wce01_normal

# 50 Hz, 60 seconds
$ python evaluate.py --bootstrap --n_seq=12 --model=ekyt_t5000_ds20_bc16_bs16_wms10_wce01_normal

# 31.25 Hz, 60 seconds
$ python evaluate.py --bootstrap --n_seq=12 --model=ekyt_t5000_ds32_bc16_bs16_wms10_wce01_normal
```

### Table V

```bash
# Requires embeddings for ekyt_t5000_ds1_bc16_bs16_wms10_wce01_normal

# Look for results under the headers that include "no ensembling, val fold"
$ python evaluate.py --val
```

### Table VI

Because this analysis involves random additive noise, results are expected to
vary for this analysis much more than for the other analyses. But general
trends should still persist.

```bash
# Compute subsequence embeddings
$ python train_and_test.py --mode=test --fold=0 --ds=8 --degrade_precision
$ python train_and_test.py --mode=test --fold=1 --ds=8 --degrade_precision
$ python train_and_test.py --mode=test --fold=2 --ds=8 --degrade_precision
$ python train_and_test.py --mode=test --fold=3 --ds=8 --degrade_precision

# Look at "bootstrapped" results
$ python evaluate.py --bootstrap --model=ekyt_t5000_ds8_bc16_bs16_wms10_wce01_degraded
$ python evaluate.py --bootstrap --model=ekyt_t5000_ds8_bc16_bs16_wms10_wce01_degraded --n_seq=2
$ python evaluate.py --bootstrap --model=ekyt_t5000_ds8_bc16_bs16_wms10_wce01_degraded --n_seq=12
```

### Table VII

```bash
# Requires embeddings for ekyt_t5000_ds1_bc16_bs16_wms10_wce01_normal
# Requires embeddings for ekyt_t5000_ds8_bc16_bs16_wms10_wce01_normal
# Requires embeddings for ekyt_t5000_ds8_bc16_bs16_wms10_wce01_degraded

# First group
$ python evaluate.py --bootstrap --n_seq=12

# Second group
$ python evaluate.py --bootstrap --round=5

# Third group
$ python evaluate.py --bootstrap --task=RAN --model=ekyt_t5000_ds8_bc16_bs16_wms10_wce01_normal

# Fourth group (due to additive random noise, results may vary)
$ python evaluate.py --bootstrap --task=RAN --model=ekyt_t5000_ds8_bc16_bs16_wms10_wce01_degraded

# Fifth group (look at results under "JuDo1000" header)
$ python evaluate.py --judo

# Sixth group (look at results under "JuDo1000" header)
$ python evaluate.py --judo --n_seq=2

# Seventh group (look at results under "JuDo1000" header)
$ python evaluate.py --judo --n_seq=12
```

### Figure 7

Although code is not provided for recreating Figure 7, we can still replicate
the performance measures presented in that figure.

```bash
# w_MS = 1.0, w_CE = 0.0
$ python train_and_test.py --mode=test --fold=0 --w_ms=1.0 --w_ce=0.0
$ python train_and_test.py --mode=test --fold=1 --w_ms=1.0 --w_ce=0.0
$ python train_and_test.py --mode=test --fold=2 --w_ms=1.0 --w_ce=0.0
$ python train_and_test.py --mode=test --fold=3 --w_ms=1.0 --w_ce=0.0
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce00_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce00_normal --n_seq=12

# w_MS = 1.0, w_CE = 0.1 (default settings)
$ python train_and_test.py --mode=test --fold=0
$ python train_and_test.py --mode=test --fold=1
$ python train_and_test.py --mode=test --fold=2
$ python train_and_test.py --mode=test --fold=3
$ python evaluate.py
$ python evaluate.py --n_seq=12

# w_MS = 1.0, w_CE = 0.2
$ python train_and_test.py --mode=test --fold=0 --w_ms=1.0 --w_ce=0.2
$ python train_and_test.py --mode=test --fold=1 --w_ms=1.0 --w_ce=0.2
$ python train_and_test.py --mode=test --fold=2 --w_ms=1.0 --w_ce=0.2
$ python train_and_test.py --mode=test --fold=3 --w_ms=1.0 --w_ce=0.2
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce02_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce02_normal --n_seq=12

# w_MS = 1.0, w_CE = 0.3
$ python train_and_test.py --mode=test --fold=0 --w_ms=1.0 --w_ce=0.3
$ python train_and_test.py --mode=test --fold=1 --w_ms=1.0 --w_ce=0.3
$ python train_and_test.py --mode=test --fold=2 --w_ms=1.0 --w_ce=0.3
$ python train_and_test.py --mode=test --fold=3 --w_ms=1.0 --w_ce=0.3
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce03_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce03_normal --n_seq=12

# w_MS = 1.0, w_CE = 0.4
$ python train_and_test.py --mode=test --fold=0 --w_ms=1.0 --w_ce=0.4
$ python train_and_test.py --mode=test --fold=1 --w_ms=1.0 --w_ce=0.4
$ python train_and_test.py --mode=test --fold=2 --w_ms=1.0 --w_ce=0.4
$ python train_and_test.py --mode=test --fold=3 --w_ms=1.0 --w_ce=0.4
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce04_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce04_normal --n_seq=12

# w_MS = 1.0, w_CE = 0.5
$ python train_and_test.py --mode=test --fold=0 --w_ms=1.0 --w_ce=0.5
$ python train_and_test.py --mode=test --fold=1 --w_ms=1.0 --w_ce=0.5
$ python train_and_test.py --mode=test --fold=2 --w_ms=1.0 --w_ce=0.5
$ python train_and_test.py --mode=test --fold=3 --w_ms=1.0 --w_ce=0.5
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce05_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce05_normal --n_seq=12

# w_MS = 1.0, w_CE = 0.6
$ python train_and_test.py --mode=test --fold=0 --w_ms=1.0 --w_ce=0.6
$ python train_and_test.py --mode=test --fold=1 --w_ms=1.0 --w_ce=0.6
$ python train_and_test.py --mode=test --fold=2 --w_ms=1.0 --w_ce=0.6
$ python train_and_test.py --mode=test --fold=3 --w_ms=1.0 --w_ce=0.6
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce06_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce06_normal --n_seq=12

# w_MS = 1.0, w_CE = 0.7
$ python train_and_test.py --mode=test --fold=0 --w_ms=1.0 --w_ce=0.7
$ python train_and_test.py --mode=test --fold=1 --w_ms=1.0 --w_ce=0.7
$ python train_and_test.py --mode=test --fold=2 --w_ms=1.0 --w_ce=0.7
$ python train_and_test.py --mode=test --fold=3 --w_ms=1.0 --w_ce=0.7
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce07_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce07_normal --n_seq=12

# w_MS = 1.0, w_CE = 0.8
$ python train_and_test.py --mode=test --fold=0 --w_ms=1.0 --w_ce=0.8
$ python train_and_test.py --mode=test --fold=1 --w_ms=1.0 --w_ce=0.8
$ python train_and_test.py --mode=test --fold=2 --w_ms=1.0 --w_ce=0.8
$ python train_and_test.py --mode=test --fold=3 --w_ms=1.0 --w_ce=0.8
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce08_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce08_normal --n_seq=12

# w_MS = 1.0, w_CE = 0.9
$ python train_and_test.py --mode=test --fold=0 --w_ms=1.0 --w_ce=0.9
$ python train_and_test.py --mode=test --fold=1 --w_ms=1.0 --w_ce=0.9
$ python train_and_test.py --mode=test --fold=2 --w_ms=1.0 --w_ce=0.9
$ python train_and_test.py --mode=test --fold=3 --w_ms=1.0 --w_ce=0.9
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce09_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce09_normal --n_seq=12

# w_MS = 1.0, w_CE = 1.0
$ python train_and_test.py --mode=test --fold=0 --w_ms=1.0 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=1 --w_ms=1.0 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=2 --w_ms=1.0 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=3 --w_ms=1.0 --w_ce=1.0
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce10_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms10_wce10_normal --n_seq=12

# w_MS = 0.9, w_CE = 1.0
$ python train_and_test.py --mode=test --fold=0 --w_ms=0.9 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=1 --w_ms=0.9 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=2 --w_ms=0.9 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=3 --w_ms=0.9 --w_ce=1.0
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms09_wce10_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms09_wce10_normal --n_seq=12

# w_MS = 0.8, w_CE = 1.0
$ python train_and_test.py --mode=test --fold=0 --w_ms=0.8 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=1 --w_ms=0.8 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=2 --w_ms=0.8 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=3 --w_ms=0.8 --w_ce=1.0
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms08_wce10_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms08_wce10_normal --n_seq=12

# w_MS = 0.7, w_CE = 1.0
$ python train_and_test.py --mode=test --fold=0 --w_ms=0.7 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=1 --w_ms=0.7 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=2 --w_ms=0.7 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=3 --w_ms=0.7 --w_ce=1.0
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms07_wce10_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms07_wce10_normal --n_seq=12

# w_MS = 0.6, w_CE = 1.0
$ python train_and_test.py --mode=test --fold=0 --w_ms=0.6 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=1 --w_ms=0.6 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=2 --w_ms=0.6 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=3 --w_ms=0.6 --w_ce=1.0
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms06_wce10_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms06_wce10_normal --n_seq=12

# w_MS = 0.5, w_CE = 1.0
$ python train_and_test.py --mode=test --fold=0 --w_ms=0.5 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=1 --w_ms=0.5 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=2 --w_ms=0.5 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=3 --w_ms=0.5 --w_ce=1.0
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms05_wce10_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms05_wce10_normal --n_seq=12

# w_MS = 0.4, w_CE = 1.0
$ python train_and_test.py --mode=test --fold=0 --w_ms=0.4 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=1 --w_ms=0.4 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=2 --w_ms=0.4 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=3 --w_ms=0.4 --w_ce=1.0
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms04_wce10_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms04_wce10_normal --n_seq=12

# w_MS = 0.3, w_CE = 1.0
$ python train_and_test.py --mode=test --fold=0 --w_ms=0.3 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=1 --w_ms=0.3 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=2 --w_ms=0.3 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=3 --w_ms=0.3 --w_ce=1.0
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms03_wce10_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms03_wce10_normal --n_seq=12

# w_MS = 0.2, w_CE = 1.0
$ python train_and_test.py --mode=test --fold=0 --w_ms=0.2 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=1 --w_ms=0.2 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=2 --w_ms=0.2 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=3 --w_ms=0.2 --w_ce=1.0
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms02_wce10_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms02_wce10_normal --n_seq=12

# w_MS = 0.1, w_CE = 1.0
$ python train_and_test.py --mode=test --fold=0 --w_ms=0.1 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=1 --w_ms=0.1 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=2 --w_ms=0.1 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=3 --w_ms=0.1 --w_ce=1.0
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms01_wce10_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms01_wce10_normal --n_seq=12

# w_MS = 0.0, w_CE = 1.0
$ python train_and_test.py --mode=test --fold=0 --w_ms=0.0 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=1 --w_ms=0.0 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=2 --w_ms=0.0 --w_ce=1.0
$ python train_and_test.py --mode=test --fold=3 --w_ms=0.0 --w_ce=1.0
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms00_wce10_normal
$ python evaluate.py --model=ekyt_t5000_ds1_bc16_bs16_wms00_wce10_normal --n_seq=12
```
