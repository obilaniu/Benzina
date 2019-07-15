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

In the example above, we first create a ``benzina.torch.ImageNet`` and specify
the location of the dataset.

.. note::
   To be able to quickly load your dataset with the hardware decoder of a GPU,
   Benzina needs it to be converted in its own format embedding h.264 containers.

.. code-block:: python

    dataset = bz.ImageNet("path/to/data")

Then we define the training and validation samplers for the dataset. In this case,
the training, validation and test data are in the same dataset with the training
data being at the beginning followed by the validation then the test data.

.. code-block:: python

    indices = list(range(len(dataset)))
    n_valid = 50000
    n_test = 100000
    n_train = len(dataset) - n_valid - n_test
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[:n_train])
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[n_train:-n_test])

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

.. code-block:: python

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

.. code-block:: python

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

As demonstrated in the `full example loading ImageNet to feed a PyTorch module <https://github.com/obilaniu/Benzina/blob/master/Users/satya/travail/examples/python/imagenet>`_, code change between a pure PyTorch implementation and an implementation using Benzina holds in only a few lines

.. code-block:: bash

    $ diff -ty --suppress-common-lines examples/python/imagenet/main.py examples/python/imagenet/imagenet_pytorch.py

.. code-block:: none

                                                                    >  import torchvision.transforms as transforms
                                                                    >  import torchvision.datasets as datasets
    ## Benzina        ###                                           <
    # Dependancies                                                  <
    import benzina.torch as bz                                      <
    import benzina.torch.operations as ops                          <
    ### Benzina - end ###                                           <
                                                                    <
                                                                    >  parser.add_argument('-j', '--workers', default=4, type=int, met
                                                                    >                      help='number of data loading workers (defau
                                                                    |      traindir = os.path.join(args.data, 'train')
        ### Benzina       ###                                       |      valdir = os.path.join(args.data, 'val')
        # Dataset                                                   |      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406]
        dataset = bz.ImageNet(args.data)                            |                                       std=[0.229, 0.224, 0.225])
                                                                    |
        indices = list(range(len(dataset)))                         |      train_dataset = datasets.ImageFolder(
        n_valid = 50000                                             |          traindir,
        n_test = 100000                                             |          transforms.Compose([
        n_train = len(dataset) - n_valid - n_test                   |              transforms.RandomResizedCrop(224),
        train_sampler = torch.utils.data.SubsetRandomSampler(indice |              transforms.RandomHorizontalFlip(),
        valid_sampler = torch.utils.data.SubsetRandomSampler(indice |              transforms.ToTensor(),
                                                                    |              normalize,
        # Dataloaders                                               |          ]))
        bias = ops.ConstantBiasTransform(bias=(123.675, 116.28 , 10 |
        std = ops.ConstantNormTransform(norm=(58.395, 57.12 , 57.37 |      train_loader = torch.utils.data.DataLoader(
                                                                    |          train_dataset, batch_size=args.batch_size, shuffle=True
        train_loader = bz.DataLoader(dataset, batch_size=args.batch |          num_workers=args.workers, pin_memory=True)
            sampler=train_sampler, seed=args.seed, shape=(224,224), |
            norm_transform=std, warp_transform=ops.SimilarityTransf |      val_loader = torch.utils.data.DataLoader(
        val_loader = bz.DataLoader(dataset, batch_size=args.batch_s |          datasets.ImageFolder(valdir, transforms.Compose([
            sampler=valid_sampler, seed=args.seed, shape=(224,224), |              transforms.Resize(256),
            norm_transform=std, warp_transform=ops.SimilarityTransf |              transforms.CenterCrop(224),
        ### Benzina - end ###                                       |              transforms.ToTensor(),
                                                                    >              normalize,
                                                                    >          ])),
                                                                    >          batch_size=args.batch_size, shuffle=False,
                                                                    >          num_workers=args.workers, pin_memory=True)
