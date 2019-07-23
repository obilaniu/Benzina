.. |pypi| image:: https://badge.fury.io/py/benzina.svg
   :scale: 100%
   :target: https://pypi.python.org/pypi/benzina

.. |docs| image:: https://readthedocs.org/projects/docs/badge/?version=latest
   :scale: 100%
   :target: https://benzina.readthedocs.io/en/latest

|pypi| |docs|


=================
Бензина / Benzina
=================


Description of the project
==========================

Benzina is an image loading library that accelerates image loading and preprocessing
by making use of the hardware decoder in NVIDIA's GPUs.

Since it minimize the use of the CPU and of the GPU computing units, it's easier
to reach saturation of GPU computing power / CPU. In our tests using ResNet18 models
in PyTorch on the ImageNet 2012 dataset, we could observe an increase by 2.4x the
amount of images loaded, preprocessed then processed by the model when using a
single CPU and GPU:

===================   ===================   ===========   ===========   =================   ========================
Data loader           CPU                   CPU Workers   GPU           GPU compute speed   Pipeline effective speed
===================   ===================   ===========   ===========   =================   ========================
PyTorch ImageFolder   Intel Xeon E5-2623*   2             Tesla V100*   1050 img/s          400 img/s
Benzina               Intel Xeon E5-2623*   1             Tesla V100*   1050 img/s          960 img/s
===================   ===================   ===========   ===========   =================   ========================

.. Note::
   * Intel Xeon E5-2623 is the Xeon E5-2623 v3 @ 3.00 GHz version
   * Tesla V100 is the Tesla V100 PCIE 16GB version

The name "Benzina" is a phonetic transliteration of the Ukrainian word "Бензина", meaning "gasoline" (or "petrol").


ImageNet loading in PyTorch
===========================

As long as your dataset is converted into Benzina's data format, you can load it
to train a PyTorch model in a few lines of code. Here is an example demonstrating
how this can be done with an ImageNet dataset. It is based on the
`ImageNet example from PyTorch <https://github.com/pytorch/examples/tree/master/imagenet>`_

.. code-block:: python

    import torch
    import benzina.torch as bz
    import benzina.torch.operations as ops

    seed = 1234
    torch.manual_seed(seed)

    # Dataset
    dataset = bz.ImageNet("path/to/data")

    indices = list(range(len(dataset)))
    n_valid = 50000
    n_test = 100000
    n_train = len(dataset) - n_valid - n_test
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[:n_train])
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[n_train:-n_test])

    # Dataloaders
    bias = ops.ConstantBiasTransform(bias=(123.675, 116.28 , 103.53))
    std = ops.ConstantNormTransform(norm=(58.395, 57.12 , 57.375))

    train_dataloader = bz.DataLoader(
        dataset,
        batch_size=256,
        sampler=train_sampler,
        seed=seed,
        shape=(224,224),
        bias_transform=bias,
        norm_transform=std,
        warp_transform=ops.SimilarityTransform(flip_h=0.5))
    valid_dataloader = bz.DataLoader(
        dataset,
        batch_size=512,
        sampler=valid_sampler,
        seed=seed,
        shape=(224,224),
        bias_transform=bias,
        norm_transform=std,
        warp_transform=ops.SimilarityTransform())

    for epoch in range(1, 10):
        # train for one epoch
        train(train_dataloader, ...)

        # evaluate on validation set
        accuracy = validate(valid_dataloader, ...)


==========================================
`Objectives <https://benzina.readthedocs.io/en/latest/objectives.html>`_
==========================================


=============================================
`Known limitations <https://benzina.readthedocs.io/en/latest/limits.html>`_
=============================================


====================================
`Roadmap <https://benzina.readthedocs.io/en/latest/roadmap.html>`_
====================================


==========================================================
`How to Contribute <https://benzina.readthedocs.io/en/latest/contribution/_index.html>`_
==========================================================


`Submitting bugs <https://benzina.readthedocs.io/en/latest/contribution/_index.html#submitting-bugs>`_
========================================================================


`Contributing changes <https://benzina.readthedocs.io/en/latest/contribution/_index.html#contributing-changes>`_
==================================================================================
