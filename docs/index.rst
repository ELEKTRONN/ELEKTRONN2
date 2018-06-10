ELEKTRONN2
==========

ELEKTRONN2 is a Python-based toolkit for training 3D/2D CNNs
and general neural networks.

.. note::
   ELEKTRONN2 is being superceded by the more flexible, PyTorch-based
   `elektronn3 <https://github.com/ELEKTRONN/elektronn3>`_ library. elektronn3
   is actively developed and supported, so we encourage you to use it instead
   of ELEKTRONN2 (if elektronn3's more experimental status and currently less
   extensive documentation are acceptable for you).

Introduction
------------

.. General information about ELEKTRONN2 and neural networks.

.. toctree::
   :maxdepth: 2

   elektronn2
   intro_nn
   installation


Tutorial
--------

Code examples for creating, training and deploying neural networks with ELEKTRONN2.

.. toctree::
   :maxdepth: 3

   examples
   predictions


API documentation
-----------------

Auto-generated documentation from docstrings and signatures.

ELEKTRONN2 consists of 4 main sub-modules:

* ``neuromancer``: Classes and functions for designing neural network models
* ``training``: Training neural networks
* ``data``: Reading and processing data sets
* ``utils``: Utility functions and data structures

.. toctree::
   :maxdepth: 3

   source/elektronn2.neuromancer
   source/elektronn2.training
   source/elektronn2.data
   source/elektronn2.utils
..   source/elektronn2.malis
..   source/elektronn2.config
..    source/elektronn2.tests


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
