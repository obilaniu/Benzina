To compile Benzina on the Cedar cluster, most of the commands must be executed
on the login nodes.

.. Note::
   It is recommended to setup a virtual environment before compiling and using
   Benzina.

**Load the Python module**

.. code-block:: bash

    $ module load python/3.6.3

**Clone the Nauka and Benzina projects**

.. code-block:: bash

    $ git clone https://github.com:obilaniu/Nauka.git
    $ git clone https://github.com:obilaniu/Benzina.git

**Compile and install Nauka**

Install the Nauka's dependency then compile and install Nauka:

.. code-block:: bash

    $ pip install --no-index numpy
    $ cd Nauka
    $ python setup.py install

**Install the dependencies of Benzina**

Ninja requires *SciKit-build* to compile itself. To prevent *SciKit-build* from
installing the most recent version of *setup-tools* and use instead the version
provided in the Cedar environment, create a constraints files and use it while
installing *SciKit-build*:

.. code-block:: bash

    $ echo 'setuptools>=27.2.0,<=28.8.0' > pip_constraints
    $ pip install -c pip_constraints scikit-build

Then, install Meson and Ninja:

.. code-block:: bash

    $ pip install meson ninja

.. Note::
   To use the PyTorch interface, you will also need to install PyTorch:

   .. code-block:: bash

       $ pip install torch

**Compile and install Benzina**

Request a GPU on the cluster:

.. code-block:: bash

    $ salloc --time=0:10:0 --account=account_id --gres=gpu:1

Load the CUDA module:

.. code-block:: bash

    $ module load cuda/10

.. Note::
   On Cedar, only ``cuda/9`` and ``cuda/10`` are compatible since the module
   ``cuda/8`` comes with an incompatible version of Video Codec SDK.

Then, compile and install Benzina:

.. code-block:: bash

    $ cd Benzina
    $ CUDA_HOME=${CUDA_HOME%:*} python setup.py install

.. Note::
   ``${CUDA_HOME%:*}`` will trim the ``:`` and what follows in the variable
