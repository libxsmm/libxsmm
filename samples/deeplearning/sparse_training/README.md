# DNN Training with Incremental Sparsification + Sparse JIT Kernels

## This project contains code for the following DNN models

1. Resnet - ported from [link](https://pytorch.org/vision/stable/models.html)
2. Transformer - ported from [link](https://github.com/pytorch/fairseq)
3. DLRM - ported from [link](https://github.com/facebookresearch/dlrm)
4. PCL_MLP - A python extension of the `torch.nn.Linear` module that uses efficient sparse JIT kernels for matrix multiplication (supports forward, backward and update pass) - ported from [link](https://github.com/hfp/libxsmm/tree/master/samples/deeplearning/sparse_weight_mult)

## Features

1. Training scripts for all three models located at the root of each directory in a form of a shell file
2. By specifying each of the four parameters, the pruning criteria (magnitude-based or random-based), the pruning start time and end time and target sparsity you can apply incremental sparsity to model weights for training
3. Additionally, by specifying a tensorboard log directory, one can examine training logs and metrics using tensorboard.

## Data preparation

Each model requires an extensive amount of data to be properly stress-tested against incremental sparsity. According to [The State of Sparsity](https://arxiv.org/abs/1902.09574) and by extensive experimentation, using a relatively small dataset or an overparameterized model may lead to false performance implications. For instance, when a ResNet-50 model is trained with the CIFAR-10 dataset or if the base Transformer is trained with a limited sentence pair dataset (i.e., EN-VI) it may seem as if the model isn't impacted even with extremely high sparsity since the model was overdetermined to begin with.

- For Resnet
- For Resnet training, a smaller subset of ImageNet was used, called ImageNette due to its massiveness in size. Download from [here](https://github.com/fastai/imagenette). 
- For Transformer
- As a neural machine translation task, the transformer model requires the WMT2014 EN_DE dataset. Preprocessing steps are described [here](https://fairseq.readthedocs.io/en/latest/getting_started.html#data-pre-processing)
- For DLRM
- Training the DLRM requires the terabyte dataset [link](https://labs.criteo.com/2013/12/download-terabyte-click-logs/)

## Running scripts

Each project consists of two scripts: a script that launches `sbatch` scripts for experimenting various target sparsities (usually named as `launch_pruning_runs.sh`)and a script that runs a single experiment. Use accordingly.

1. ResNet model
`./launch_pruning_jobs.sh ${TARGET_SPARSITY}` or
`python train.py ${TARGET_SPARSITY}`
2.  Transformer(FAIRSEQ) model
`./launch_pruning_runs.sh` or `./prune_en_de.sh ${TARGET_SPARSITY} ${PRUNE_TYPE} ${EMB}`
where PRUNE_TYPE is either `magnitude` or `random` and EMB indicates whether the embedding portion is pruned alongside the weights
3. DLRM model
`./launch_pruning_runs.sh` or `./run_terabyte.sh ${TARGET_SPARSITY} ${PRUNE_TYPE}`
where PRUNE_TYPE is either `magnitude` or `random` 
