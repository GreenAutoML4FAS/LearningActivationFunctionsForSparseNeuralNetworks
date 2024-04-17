# Learning Activation Functions for Sparse Neural Networks

This is the code to the paper 
[**"Learning Activation Functions for Sparse Neural Networks"**](https://arxiv.org/abs/2305.10964) 
by Mohammad Loni, Aditya Mohan, Mehdi Asadi, and Marius Lindauer
which was published at AutoML 2023.

If you have questions or want to contribute, please refer to the official repository.

---

Abstract:
>Sparse Neural Networks (SNNs) can potentially demonstrate similar performance to their dense counterparts while saving significant energy and memory at inference. However, the accuracy drop incurred by SNNs, especially at high pruning ratios, can be an issue in critical deployment conditions. While recent works mitigate this issue through sophisticated pruning techniques, we shift our focus to an overlooked factor: hyperparameters and activation functions. Our analyses have shown that the accuracy drop can additionally be attributed to (i) Using ReLU as the default choice for activation functions unanimously, and (ii) Fine-tuning SNNs with the same hyperparameters as dense counterparts. Thus, we focus on learning a novel way to tune activation functions for sparse networks and combining these with a separate hyperparameter optimization (HPO) regime for sparse networks. By conducting experiments on popular DNN models (LeNet-5, VGG-16, ResNet-18, and EfficientNet-B0) trained on MNIST, CIFAR-10, and ImageNet-16 datasets, we show that the novel combination of these two approaches, dubbed Sparse Activation Function Search, short: SAFS, results in up to 15.53%, 8.88%, and 6.33% absolute improvement in the accuracy for LeNet-5, VGG-16, and ResNet-18 over the default training protocols, especially at high pruning ratios.

We invite the reader to have a look at our paper and the code to reproduce our results. All details necessary to understand the code and hyperparameter are given in the paper.

---
## Install

Checkout the repository:

```bash
git clone https://github.com/GreenAutoML4FAS/LearningActivationFunctionsForSparseNeuralNetworks
cd LearningActivationFunctionsForSparseNeuralNetworks
```

Create a new environment and set it up:

```bash
conda create -n safs python=3.10
conda activate safs
pip install -r requirements.txt
```

## Run

We will execute `main.py` for all our experiments on the MNIST, CIFAR-10 and Imagenet16 datasets with the following parameters:

- `--d`: dataset selection (0:MNIST, 1:CIFAR10, 2:Imagenet16)
- `--model_arch`: model selection(1:Lenet5, 2:VGG-16, 3:ResNet-18, 4:EfficientNet_B0)
- `--optim-method`: stage1 search strategy(0:LAHC, 1:RS, 2:GA, 3:SA)
- `--first_train`: enable first_train(True, False)
- `--Train_after_prune'`: enable Train_after_prune(True, False)
- `--first_stage'`: enable first_stage(True, False)
- `--Second_stage`: enable Second_stage(True, False)
- `--gpus`: num of available gpus
- `--pruning_method`: LWM
- `--set_device`: select a gpu(0, 1, 2, ...)


## Example
First, train VGG-16 networks `--first_train=1` then pruned the model  with `--pruning_rate=0.99`. Next, retrain the pruned model `--Train_after_prune=1`, followed by the design of new activation functions for each layer of the network using the LAHC algorithm `--first_stage=1`. Finally, execute the second stage HPO `--Second_stage=1`.  

```bash
python main.py --model_arch=2 --runing_mode="metafire" --pruning_rate=0.99 --first_train=1  --Train_after_prune=1 --first_stage=1 --Second_stage=1
```

### Summery of results
![Results](./docs/images/results_table.png?raw=true)

## Citation

If you find this work useful, please include the following citation:

```latex
@inproceedings{loni2023learning,
  title={Learning Activation Functions for Sparse Neural Networks},
  author={Loni, Mohammad and Mohan, Aditya and Asadi, Mehdi and Lindauer, Marius},
  booktitle={International Conference on Automated Machine Learning (AutoMLConf)},
  pages={16--1},
  year={2023},
  organization={PMLR}
}
```

---

## License Notice

The **content of this repository** is licensed under the Apache.

## Acknowledgement
This work was partially supported by the Federal Ministry of the Environment, Nature Conservation, Nuclear Safety and Consumer Protection, Germany under the project **GreenAutoML4FAS** (grant no. 67KI32007A).

The work was done at the Leibniz University Hannover and published at the *International Conference on Automated Machine Learning* 2023.

<p align="center">
    <img width="100" height="100" src="fig/AutoML4FAS_Logo.jpeg"> 
    <img width="300" height="100" src="fig/Bund.png">
    <img width="300" height="100" src="fig/LUH.png"> 
</p>