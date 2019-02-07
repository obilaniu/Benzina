ImageNet training in PyTorch
============================

This implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset using Benzina to load the data. It is based on the `implementation example <https://github.com/pytorch/examples/tree/master/imagenet>` found in PyTorch

Training
--------

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

.. code-block:: bash
    python main.py -a resnet18 [imagenet-folder in Benzina format]

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. This is appropriate for ResNet and models with batch normalization, but too high for AlexNet and VGG. Use 0.01 as the initial learning rate for AlexNet or VGG:

.. code-block:: bash
    python main.py -a alexnet --lr 0.01 [imagenet-folder in Benzina format]

Usage
-----

.. code-block:: bash
    usage: main.py [-h] [-a ARCH] [--epochs N] [--start-epoch N] [-b N] [--lr LR]
                   [--momentum M] [--wd W] [-p N] [--resume PATH] [-e]
                   [--pretrained] [--seed SEED] [--gpu GPU]
                   DIR

    PyTorch ImageNet Training

    positional arguments:
      DIR                   path to dataset

    optional arguments:
      -h, --help            show this help message and exit
      -a ARCH, --arch ARCH  model architecture: alexnet | densenet121 |
                            densenet161 | densenet169 | densenet201 | inception_v3
                            | resnet101 | resnet152 | resnet18 | resnet34 |
                            resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                            vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                            | vgg19_bn (default: resnet18)
      --epochs N            number of total epochs to run
      --start-epoch N       manual epoch number (useful on restarts)
      -b N, --batch-size N  mini-batch size (default: 256), this is the total
                            batch size of all GPUs on the current node
      --lr LR, --learning-rate LR
                            initial learning rate
      --momentum M          momentum
      --wd W, --weight-decay W
                            weight decay (default: 1e-4)
      -p N, --print-freq N  print frequency (default: 10)
      --resume PATH         path to latest checkpoint (default: none)
      -e, --evaluate        evaluate model on validation set
      --pretrained          use pre-trained model
      --seed SEED           seed for initializing training.
      --gpu GPU             GPU id to use.
