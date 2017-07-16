************
Installation
************

.. contents::
  :local:


ELEKTRONN2
==========

There are three supported ways to install ELEKTRONN2:
With the package managers ``conda``, ``pip`` or via the
`AUR <https://wiki.archlinux.org/index.php/Arch_User_Repository>`_. We highly
recommend using the ``conda`` install method.


.. note:: ELEKTRONN2 is supported on Linux (x86_64), with Python versions
  2.7, 3.4, 3.5 and 3.6.
  Everything else is untested, but some other platforms might work as well.
  Please create an `issue <https://github.com/ELEKTRONN/ELEKTRONN2/issues>`_ if
  you are interested in support for other platforms.


Installing with ``conda``
-------------------------

The recommended way to install ELEKTRONN2 is to use the
`conda <https://conda.io/docs/>`_ package manager, which is included
in `Anaconda <https://www.continuum.io/downloads>`_
(`Miniconda <https://conda.io/miniconda.html>`_ also works).
The ELEKTRONN2 package is hosted by the
`conda-forge <https://conda-forge.github.io/>`_
channel, so you first need to add it to your local channel list
if you haven't yet done this::

  conda config --add channels conda-forge

Then you can either install ELEKTRONN2 directly into your current
environment::

  conda install elektronn2

... or create a new `conda env <https://conda.io/docs/using/envs.html>`_
just for ELEKTRONN2::

  conda create -n elektronn2_env elektronn2

Optionally run ``source activate elektronn2_env``
(or ``conda activate elektronn2_env`` if you use the fish shell) to activate
the new environment and ensure all ELEKTRONN2 executables are on your PATH.
The effects of the activation only last for the current shell session, so
remember to repeat this step after re-opening your shell.

.. TODO: "conda activate" works in all shells starting with conda 4.4, so we
  can remove the "activate" distinction above in, let's say 2018.


Installing with ``pip``
-----------------------

You can install the current version of ELEKTRONN2 and all of its
dependencies with the ``pip`` package manager. For Python 3, run::

  python3 -m pip install numpy
  python3 -m pip install elektronn2

Or if you want to install ELEKTRONN2 for Python 2::

  python2 -m pip install numpy
  python2 -m pip install elektronn2

To prevent permission errors and conflicts with other packages,
we suggest that you run these ``pip install`` commands
inside a `virtualenv <https://virtualenv.pypa.io>`_
or a `conda env <https://conda.io/docs/using/envs.html>`_.

Please **do not** attempt to use ``sudo`` or the ``root`` account for ``pip install``,
because this can overwrite system packages and thus potentially destroy your
operating system (this is not specific to ELEKTRONN2, but a general flaw in ``pip``
and applies to all packages (see `this issue <https://github.com/pypa/pip/issues/1668>`_).
``pip install --user`` can be used instead, but this method can also break other
Python packages due to the version/precedence conflicts between system and user packages.

.. TODO: Manual numpy install is only necessary because numba doesn't provide
  wheels. Once wheels are public, delete the "pip install numpy" lines.

.. TODO: Maybe describe an example setup of a virtualenv.


AUR (``pacaur``)
----------------

If you have ``pacaur`` (in Arch or derivative distros), you can install the
`ELEKTRONN2 AUR packages <https://aur.archlinux.org/packages/python-elektronn2/>`_
by running::

  pacaur -S python-elektronn2 # for Python 3
  pacaur -S python2-elektronn2 # for Python 2

(Other AUR helpers like ``yaourt`` will work too, of course.)

.. note:: In the Python 2 AUR package, a ``2`` is appended to command names
  to prevent file name conflicts (e.g. ``elektronn2-train`` becomes
  ``elektronn2-train2``)



CUDA and cuDNN
==============

In order to use Nvidia GPUs for accelleration, you will need to additionally install CUDA.
Install Nvidia's CUDA toolkit by following the instructions on the
`Nvidia website <https://developer.nvidia.com/cuda-downloads>`_ or install it with your
system package manager.

Even higher performance for training and inference in deep convolutional neural networks
can be enabled by installing the `cuDNN library <https://developer.nvidia.com/cuDNN>`_.
Once it is installed, ELEKTRONN2 will automatically make use of it.

For example if you use Arch Linux, both libraries can be installed with::

  sudo pacman -S cuda cudnn

If you don't have root rights on your machine, you can install CUDA and cuDNN to a
custom path in your ``$HOME`` directory. If you do that, don't forget to
update your environment variables, e.g. by adding these lines to you ``~/.bashrc``-file
(assuming you have installed both to ``~/opt/cuda/``)::

  export PATH=~/opt/cuda/bin:$PATH
  export LD_LIBRARY_PATH=~/opt/cuda/lib64:$LD_LIBRARY_PATH


Theano configuration
====================

Lastly, the `Theano <http://deeplearning.net/software/theano/index.html>`_
back end needs to be configured. It is responsible for optimizing and
compiling ELEKTRONN2's computation graphs.
Create the file ``~/.theanorc`` and put the following lines into it::

  [global]
  floatX = float32
  linker = cvm_nogc

  [nvcc]
  fastmath = True

.. note::
  The ``cvm_nogc`` linker option disables garbage collection. This increases
  GPU-RAM usage but gives a significant performance boost. If you run out
  of GPU-RAM, remove this option (or set it to ``cvm``).

If your CUDA installation is in a custom location (e.g. ``~/opt/cuda/``) and is
not found automatically, additionally set the ``cuda.root`` option in your
``~/.theanorc``::

  [cuda]
  root = ~/opt/cuda

If you always want to use the same GPU (e.g. GPU number 0) for ELEKTRONN2 and
don't want to specify it all the time in your command line, you can also
configure it in ``~/.theanorc``::

  [global]
  device = cuda0

More options to configure Theano can be found in the
`theano.config <http://deeplearning.net/software/theano/library/config.html>`_
documentation.
