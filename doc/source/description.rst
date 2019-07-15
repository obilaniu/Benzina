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
