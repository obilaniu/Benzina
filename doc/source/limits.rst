=====================================
Known limitations and important notes
=====================================


As of September 2020
====================

* No TensorFlow integration
* Currently only supports ImageNet
* Unknown effect on model accuracy of transcoding from various JPEG formats to
  H.265
* Current transcoding filters failed on 81 images of the :ref:`imagenet_2012`
  dataset forcing them to be excluded. More information can be found in the
  dataset's README.
* Current transcoding filters required 111 images of the :ref:`imagenet_2012`
  dataset to first be transcoded to PNG prior to the final H.265 format. More
  information can be found in the dataset's README.
* High resolution images stored in the
  :ref:`bzna_input track of the input samples <imagenet_2012>` are currently
  not available through the :class:`Dataloader`. Their varying size prevent
  them from being decoded using a single hardware decoder configuration. The
  selected solution is to represent the images in the HEIF format which will be
  completed in future development.
* It is currently not possible to *compose* transformations like you can with
  ``torchvision.transforms.Compose`` but
  :py:class:`~benzina.torch.operations.SimilarityTransform` should cover most
  of the necessary images transformations.
* :py:class:`~benzina.torch.operations.SimilarityTransform` and
  :py:class:`~benzina.torch.operations.RandomResizedCrop` slightly differ from
  the behaviour of ``torchvision.transforms.RandomResizedCrop`` where, instead
  of falling back to a center crop when the random crop area doesn't fit after
  10 tries, :class:`SimilarityTransform` will still perform the crop and only
  center it on the dimension not fitting. Due to the encoding methods used in
  Benzina, this will usually result in an image with a black top border and a
  smeared bottom border or a black left border and a smeared right border if
  the crop area did not fit vertically or horizontally respectively.