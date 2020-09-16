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

In the example above, two ``benzina.torch.dataset.ImageNet`` are first created
with the location of the dataset and the desired split specified.

.. note::
   To be able to quickly load your dataset with the hardware decoder of a GPU,
   Benzina needs the data to be converted in its own format embedding H.265
   images.

.. code-block:: python

    train_dataset = bz.dataset.ImageNet("path/to/dataset", split="train")
    val_dataset = bz.dataset.ImageNet("path/to/dataset", split="val")

Then the transformations to apply to the dataset are defined. It is usually a
good idea to normalize the data based on its statistical bias and standard
deviation which can be done with Benzina by using its
``benzina.torch.operations.ConstantBiasTransform`` and
``benzina.torch.operations.ConstantNormTransform`` respectively.

.. note::
   - ``benzina.torch.operations.ConstantBiasTransform`` will substract the bias
     from the images' RGB channels
   - ``benzina.torch.operations.ConstantNormTransform`` will multiply the norm
     with the images' RGB channels

.. code-block:: python

    bias = ops.ConstantBiasTransform(bias=(123.675, 116.28 , 103.53))
    std = ops.ConstantNormTransform(norm=(58.395, 57.12 , 57.375))

The dataloaders are now ready to be instantiated. In this example, the
dataset's images are all of size 512 x 512 by the dataset specifications. A
random crop resized to 224 x 224 and a random horizontal flip will be applied
to the images prior feeding them to the model. In Benzina, this is done by
defining the size of the output tensor with the dataloader's ``shape`` argument
and using Benzina's similarity transform.

In the case of the validation transform, an alias to a specific similarity
transform, which applies a center crop of edges scale 224 / 256, resize the
cropped section to have its smaller edge matched to 224 then center a crop of
224 x 224. Another maybe more intuitive way to describe this transformation is
to see it as a resize to have the smaller edge matched to 256 then center a
crop of 224 x 224.

.. note::
   It's useful to know that ``benzina.torch.operations.SimilarityTransform``
   will automatically center the output frame on the center of the input image.
   This makes a vanilla ``benzina.torch.operations.SimilarityTransform``
   equivalent a center crop of size of the output.

.. code-block:: python

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
        warp_transform=ops.CenterResizedCrop(224/256))

As demonstrated in the `full example loading ImageNet to feed a PyTorch model
<https://github.com/obilaniu/Benzina/blob/master/Users/satya/travail/examples/python/imagenet>`_,
code change between a pure PyTorch implementation and an implementation using
Benzina holds in only a few lines.

.. code-block:: bash

    $ diff -ty --suppress-common-lines examples/python/imagenet/main.py examples/python/imagenet/imagenet_pytorch.py

.. code-block:: none

                                                                    >  import torchvision.transforms as transforms
                                                                    >  import torchvision.datasets as datasets
    ### Benzina       ###                                           <
    import benzina.torch as bz                                      <
    import benzina.torch.operations as ops                          <
    ### Benzina - end ###                                           <
                                                                    <
                                                                    >  parser.add_argument('-j', '--workers', default=4, type=int, met
                                                                    >                      help='number of data loading workers (defau
        ### Benzina       ###                                       |      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406]
        train_dataset = bz.dataset.ImageNet(args.data, split="train |                                       std=[0.229, 0.224, 0.225])
                                                                    <
        bias = ops.ConstantBiasTransform(bias=(0.485 * 255, 0.456 * <
        std = ops.ConstantNormTransform(norm=(0.229 * 255, 0.224 *  <
        train_loader = bz.DataLoader(                               |      train_dataset = datasets.ImageNet(
            train_dataset, shape=(224, 224), batch_size=args.batch_ |          args.data, "train",
            shuffle=True, seed=args.seed,                           |          transforms.Compose([
            bias_transform=bias,                                    |              transforms.RandomResizedCrop(224),
            norm_transform=std,                                     |              transforms.RandomHorizontalFlip(),
            warp_transform=ops.SimilarityTransform(                 |              transforms.ToTensor(),
                scale=(0.08, 1.0),                                  |              normalize,
                ratio=(3./4., 4./3.),                               |          ]))
                flip_h=0.5,                                         |
                random_crop=True))                                  |      train_loader = torch.utils.data.DataLoader(
                                                                    |          train_dataset, batch_size=args.batch_size, shuffle=True
        val_loader = bz.DataLoader(                                 |          num_workers=args.workers, pin_memory=True)
            bz.dataset.ImageNet(args.data, split="val"), shape=(224 |
            batch_size=args.batch_size, shuffle=args.batch_size, se |      val_loader = torch.utils.data.DataLoader(
            bias_transform=bias,                                    |          datasets.ImageNet(args.data, "val", transforms.Compose(
            norm_transform=std,                                     |              transforms.Resize(256),
            warp_transform=ops.CenterResizedCrop(224/256))          |              transforms.CenterCrop(224),
        ### Benzina - end ###                                       |              transforms.ToTensor(),
                                                                    >              normalize,
                                                                    >          ])),
                                                                    >          batch_size=args.batch_size, shuffle=False,
                                                                    >          num_workers=args.workers, pin_memory=True)
