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
