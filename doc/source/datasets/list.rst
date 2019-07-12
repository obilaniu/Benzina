=============
Datasets List
=============


ImageNet 2012
=============

This is the first dataset created for Benzina. It includes the following preprocessing of the
images:

* Resize the image to have its smallest edge be of length 256
* Center crop the image to have a 256 x 256 image

Dataset Composition
-------------------

The dataset is composed of a train set, followed by a validation set then a
test set for a total of 1 431 167 entries:

* | **Train set**
  | Entries 1 to 1281167 (1 281 167 entries)
* | **Validation set**
  | Entries 1281168 to 1331167 (50 000 entries)
* | **Test set**
  | Entries 1331168 to 1431167 (100 000 entries)

Dataset Files Structure
-----------------------

The data is separated into:

* | *data.bin*
  | A concatenation of each image in the H.264 format
* | *data.filenames*
  | The list of the original JPEG filenames including the target directories
* | *data.lengths*
  | The length in bytes of all H.264 images in *data.bin*
* | *data.nvdecode*
  | A concatenation of headers needed to decode the H.264 images using nvdecode
* | *data.protobuf*
  | The protobuf describing the H.264 images
* | *data.protobuf.txt*
  | The description of the fields in *data.protobuf*
* | *data.targets*
  | A concatenation of integers in bytes representing the target for each image
* | *SHA256SUMS*
  | The sha256 sums for all files above
