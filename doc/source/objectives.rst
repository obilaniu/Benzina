Objectives
==========

In much of the work in the field of machine learning and deep learning, a bottleneck exists in the dataloading itself. This is becoming increasingly recognised as an issue which needs to be solved.

Benzina aims to become a go-to tool for dataloading large datasets. Other tools exist, such as `Dali <https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html>`. Yet Benzina concentrates itself on two aspects : 

  * Highest level of performance for dataloading using GPU as loading device
  * Creation of a generalist storage format as a single file.

The use of a single file aims to facilitate distribution of datasets and is useful in the context of file system limits.

Generalist DNN framework methods are provided to integrate Benzina to PyTorch and TensorFlow.
