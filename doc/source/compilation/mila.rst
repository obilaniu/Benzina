To compile Benzina on the Mila's cluster, most of the commands can be executed
on one of the login nodes.

.. Note::
   It is recommended to setup a virtual environment before compiling and using
   Benzina.

**Clone the Nauka and Benzina projects**

.. code-block:: bash

    $ git clone https://github.com:obilaniu/Nauka.git
    $ git clone https://github.com:obilaniu/Benzina.git

**Compile and install Nauka**

.. code-block:: bash

    $ cd Nauka
    $ python setup.py install

**Install the dependencies of Benzina**

.. code-block:: bash

    $ pip install meson ninja

.. Note::
   To use the PyTorch interface, you will also need to install PyTorch:

   .. code-block:: bash

       $ pip install torch

**Compile and install Benzina**

Request a GPU on the cluster:

.. code-block:: bash

    $ sinter --gres=gpu

Then, compile and install Benzina:

.. code-block:: bash

    $ cd Benzina
    $ python setup.py install
