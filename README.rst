.. use rst_include to compile the README.rst on GitHub.
   pip install rst-include
   rst_include include -s README_src.rst -t README.rst

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

==========
Objectives
==========

In much of the work in the field of machine learning and deep learning, a bottleneck exists in the dataloading phase itself. This is becoming increasingly recognised as an issue which needs to be solved.

Benzina aims to become a go-to tool for dataloading large datasets. Other tools exist, such as `Dali <https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html>`_. Yet Benzina concentrates itself on two aspects :

* Highest level of performance for dataloading using GPU as loading device
* Creation of a generalist storage format as a single file facilitating distribution of datasets and useful in the context of file system limits.


Further feature points
======================

* Generalist DNN framework methods provided to integrate Benzina to PyTorch and TensorFlow
* Command line programs will be created to assist in Bezina - compatible datasets
* API interface to interact with Benzina

=================
Known limitations
=================


As of July 2019
===============

* No TensorFlow integration
* Currently only supports ImageNet
* Dataset storage is not yet normalized as a single file
* Unknown effect of compression algorithm on model accuracy

=======
RoadMap
=======


Summer 2019
===========

* Collaboration phase with researchers

* TensorFlow implementation

* Normalized format
   * Specification freeze
   * Dataset creation utils
   * More tests
   * Collaboration with researchers using new format


Autumn 2019
===========

Conference Talk on Benzina

=================
How to Contribute
=================

This section is heavily based on
`Contributing to Open Source Projects <https://github.com/bitprophet/contribution-guide.org/blob/master/index.rst>`_

Submitting bugs
===============

Due diligence
-------------

Before submitting a bug, please do the following:

* Perform **basic troubleshooting** steps:

    * **Make sure you're on the latest version.** If you're not on the most
      recent version, your problem may have been solved already! Upgrading is
      always the best first step.
    * **Try older versions.** If you're already *on* the latest release, try
      rolling back a few minor versions (e.g. if on 1.7, try 1.5 or 1.6) and
      see if the problem goes away. This will help the devs narrow down when
      the problem first arose in the commit log.
    * **Try switching up dependency versions.** If the software in question has
      dependencies (other libraries, etc) try upgrading/downgrading those as
      well.

* **Search the project's bug/issue tracker** to make sure it's not a known
  issue.
* If you don't find a pre-existing issue, consider **checking with the mailing
  list and/or IRC channel** in case the problem is non-bug-related.

What to put in your bug report
------------------------------

Make sure your report gets the attention it deserves: bug reports with missing
information may be ignored or punted back to you, delaying a fix.  The below
constitutes a bare minimum; more info is almost always better:

* **What version of the core programming language interpreter/compiler are you
  using?** For example, if it's a Python project, are you using Python 2.7.3?
  Python 3.3.1? PyPy 2.0?
* **Which version or versions of the software are you using?** Ideally, you
  followed the advice above and have ruled out (or verified that the problem
  exists in) a few different versions.
* **How can the developers recreate the bug on their end?** If possible,
  include a copy of your code, the command you used to invoke it, and the full
  output of your run (if applicable.)

    * A common tactic is to pare down your code until a simple (but still
      bug-causing) "base case" remains. Not only can this help you identify
      problems which aren't real bugs, but it means the developer can get to
      fixing the bug faster.

`Contributing changes <doc/source/contribution/contributing_changes.rst>`_
==========================================================================

