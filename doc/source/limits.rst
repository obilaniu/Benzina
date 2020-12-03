=====================================
Known limitations and important notes
=====================================


As of September 2020
====================

* No TensorFlow integration
* Currently only supports ImageNet
* Unknown effect on model accuracy of transcoding from various JPEG formats to
  H.265
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
