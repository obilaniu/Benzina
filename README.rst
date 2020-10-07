.. |pypi| image:: https://badge.fury.io/py/benzina.svg
   :scale: 100%
   :target: https://pypi.python.org/pypi/benzina

.. |docs| image:: https://readthedocs.org/projects/docs/badge/?version=latest
   :scale: 100%
   :target: https://benzina.readthedocs.io/en/latest

|pypi| |docs|


Бензина / Benzina
=================


Description of the project
==========================

Benzina is an image loading library that accelerates image loading and
preprocessing by making use of the hardware decoder in NVIDIA's GPUs.

Since it minimize the use of the CPU and of the GPU computing units, it's
easier to reach saturation of GPU computing power / CPU. In our tests using
ResNet18 models in PyTorch on the ImageNet 2012 dataset, we could observe an
increase by 1.8x the amount of images loaded, preprocessed then processed by
the model when using a single CPU and GPU:

===================   ================   ===========   =========   ===========   ==========   ==============
Data Loader           CPU                CPU Workers   CPU Usage   GPU           Batch Size   Pipeline Speed
===================   ================   ===========   =========   ===========   ==========   ==============
Benzina               Intel Xeon 2698*   1             33%         Tesla V100*   256          525 img/s
PyTorch ImageFolder   Intel Xeon 2698*   2             100%        Tesla V100*   256          290 img/s
PyTorch ImageFolder   Intel Xeon 2698*   4             100%        Tesla V100*   256          395 img/s
PyTorch ImageFolder   Intel Xeon 2698*   6             100%        Tesla V100*   256          425 img/s
DALI                  Intel Xeon 2698*   1             100%        Tesla V100*   256          575 img/s
===================   ================   ===========   =========   ===========   ==========   ==============

.. Note::
   * Intel Xeon 2698 is the Intel Xeon E5-2698 v4 @ 2.20GHz version
   * Tesla V100 is the Tesla V100 SXM2 16GB version

While DALI currently outperforms Benzina, the speedup can only be seen on JPEGs
through the `nvJPEG <https://developer.nvidia.com/nvjpeg>`_ decoder. Benzina
requires to transcode the input dataset to H.265 but then the gain can be seen
on all type of images as well as providing the dataset in a format that is
easier to distribute.

The name "Benzina" is a phonetic transliteration of the Ukrainian word
"Бензина", meaning "gasoline" (or "petrol").


ImageNet loading in PyTorch
===========================

As long as your dataset is converted into Benzina's data format, you can load
it to train a PyTorch model in a few lines of code. Here is an example
demonstrating how this can be done with an ImageNet dataset. It is based on the
`ImageNet example from PyTorch
<https://github.com/pytorch/examples/tree/master/imagenet>`_

.. code-block:: python

    import torch
    import benzina.torch as bz
    import benzina.torch.operations as ops

    seed = 1234
    torch.manual_seed(seed)

    # Dataset
    train_dataset = bz.dataset.ImageNet("path/to/dataset", split="train")
    val_dataset = bz.dataset.ImageNet("path/to/dataset", split="val")

    # Dataloaders
    bias = ops.ConstantBiasTransform(bias=(0.485 * 255, 0.456 * 255, 0.406 * 255))
    std = ops.ConstantNormTransform(norm=(0.229 * 255, 0.224 * 255, 0.225 * 255))

    train_loader = bz.DataLoader(
        train_dataset,
        shape=(224, 224),
        batch_size=256,
        shuffle=True,
        seed=seed,
        bias_transform=bias,
        norm_transform=std,
        warp_transform=ops.SimilarityTransform(scale=(0.08, 1.0),
                                               ratio=(3./4., 4./3.),
                                               flip_h=0.5,
                                               random_crop=True))
    val_loader = bz.DataLoader(
        val_dataset,
        shape=(224, 224),
        batch_size=256,
        shuffle=False,
        seed=seed,
        bias_transform=bias,
        norm_transform=std,
        warp_transform=ops.CenterResizedCrop(224/256)))

    for epoch in range(1, 10):
        # train for one epoch
        train(train_dataloader, ...)

        # evaluate on validation set
        accuracy = validate(valid_dataloader, ...)


`Objectives <https://benzina.readthedocs.io/en/latest/objectives.html>`_
========================================================================


`Known limitations and important notes <https://benzina.readthedocs.io/en/latest/limits.html>`_
===========================================================================


`Roadmap <https://benzina.readthedocs.io/en/latest/roadmap.html>`_
==================================================================


`How to Contribute <https://benzina.readthedocs.io/en/latest/contribution/_index.html>`_
========================================================================================


`Submitting bugs <https://benzina.readthedocs.io/en/latest/contribution/_index.html#submitting-bugs>`_
======================================================================================================


`Contributing changes <https://benzina.readthedocs.io/en/latest/contribution/_index.html#contributing-changes>`_
================================================================================================================
