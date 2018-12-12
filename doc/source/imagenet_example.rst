ImageNet loading in PyTorch
===================================

As long as your dataset is converted into Benzina's data format, you can load it
to train a PyTorch model in a few lines of code. Here is an example demontrating
how this can be done with an ImageNet dataset. It is heavily based on the
`ImageNet example from PyTorch <https://github.com/pytorch/examples/tree/master/imagenet>`_

::

    import numpy as np
    import torch
    import benzina.torch as bz
    import benzina.torch.operations as ops

    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Data loading
    dataset = bz.ImageNet("path/to/data")

    indices = list(range(len(dataset)))
    n_valid = len(dataset) * 1 / 5
    train_subset = torch.utils.data.Subset(dataset, indices[:-n_valid])
    valid_subset = torch.utils.data.Subset(dataset, indices[-n_valid:])

    bias = ops.ConstantBiasTransform(bias=(123.675, 116.28 , 103.53))
    std = ops.ConstantNormTransform(norm=(58.395, 57.12 , 57.375))

    train_dataloader = bz.DataLoader(
        train_subset,
        batch_size=256,
        shuffle=True,
        seed=seed,
        shape=(224,224),
        bias_transform=bias,
        norm_transform=std,
        warp_transform=ops.SimilarityTransform(reflecth=0.5))
    valid_dataloader = bz.DataLoader(
        valid_subset,
        batch_size=512,
        shuffle=False,
        seed=seed,
        shape=(224,224),
        bias_transform=bias,
        norm_transform=std,
        warp_transform=ops.SimilarityTransform())

    for epoch in range(1, 10):
        # train for one epoch
        train(train_loader, ...)

        # evaluate on validation set
        accuracy = validate(valid_dataloader, ...)

In the example above, we first create a ``benzina.torch.ImageNet`` and specify
the location of the dataset.

.. note::
   To be able to quickly load your dataset, Benzina needs it to be converted in
   its own format.

::

    dataset = bz.ImageNet("path/to/data")

Then we define the training and validation subsets of the dataset. In this case,
the training data and the validation data are in the same dataset with the training
data being at the beginning and the validation data at the end.

::

    indices = list(range(len(dataset)))
    n_valid = len(dataset) * 1 / 5
    train_subset = torch.utils.data.Subset(dataset, indices[:-n_valid])
    valid_subset = torch.utils.data.Subset(dataset, indices[-n_valid:])

The last steps are to define the dataloaders and the transformations to apply to
the dataset during the loading of the images. It is usually a good idea to normalize
the data based on its statistical bias and standard deviation which can be done with
Benzina by respectively using its ``benzina.torch.operations.ConstantBiasTransform``
and ``benzina.torch.operations.ConstantNormTransform``.

.. note::
   - ``benzina.torch.operations.ConstantBiasTransform`` will substract its bias
     from the images RGB channels
   - ``benzina.torch.operations.ConstantNormTransform`` will multiply its norm
     with the images RGB channels

::

    bias = ops.ConstantBiasTransform(bias=(123.675, 116.28 , 103.53))
    std = ops.ConstantNormTransform(norm=(58.395, 57.12 , 57.375))

The loaders are now ready to be instantiated. In this example, the dataset's images
are all of size 256 x 256. The resulting images we want to feed in our model are
the center crop of size 224 x 224 with an horizontal flip being randomly applied.
In Benzina, you would do this by first defining the size of the output image,
with the ``shape`` argument, then using Benzina's similarity transform which can
randomly apply the horizontal flip among other transformations.

.. note::
   It's useful to know that ``benzina.torch.operations.SimilarityTransform`` will
   automatically center the output frame on the input image. This means that even
   if there is no wish to apply a random transformation to the input image, like
   a scale, rotation or a translation, ``benzina.torch.operations.SimilarityTransform``
   can be still used to apply a center crop in the case the output size is not the
   same as the input size.

::

    train_dataloader = bz.DataLoader(
        train_subset,
        batch_size=256,
        shuffle=True,
        seed=seed,
        shape=(224,224),
        bias_transform=bias,
        norm_transform=std,
        warp_transform=ops.SimilarityTransform(reflecth=0.5))
    valid_dataloader = bz.DataLoader(
        valid_subset,
        batch_size=512,
        shuffle=False,
        seed=seed,
        shape=(224,224),
        bias_transform=bias,
        norm_transform=std,
        warp_transform=ops.SimilarityTransform())
