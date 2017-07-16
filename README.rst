.. image:: https://readthedocs.org/projects/elektronn2/badge/?version=latest
  :target: http://elektronn2.readthedocs.io/en/latest/?badge=latest
.. image:: https://img.shields.io/pypi/v/elektronn2.svg
  :target: https://pypi.org/project/elektronn2/
.. image:: https://anaconda.org/conda-forge/elektronn2/badges/version.svg
  :target: https://anaconda.org/conda-forge/elektronn2

****************
About ELEKTRONN2
****************

.. contents::
  :local:


Introduction
============


What is ELEKTRONN2?
-------------------

ELEKTRONN2 is a flexible and extensible **Python toolkit** that facilitates **design,**
**training and application of neural networks**.

It can be used for general machine learning tasks, but its main focus is on
**convolutional neural networks (CNNs) for high-throughput 3D and 2D image analysis**.

ELEKTRONN2 is especially useful for efficiently assessing experimental
neural network architectures thanks to its **powerful interactive shell** that can be
entered at any time during training, temporarily pausing all calculations.

The shell interface provides shortcuts and autocompletions for
frequently used operations (e.g. adjusting the learning rate)
and also provides a complete python shell with full read/write access to the network model, the
plotting subsystem and all training parameters and hyperparameters.
Changes made in the shell take effect immediately, so you can **monitor, analyse and**
**manipulate your training sessions directly during their run time**, without losing
any training progress.

Computationally expensive calculations are **automatically compiled and transparently**
**executed as highly-optimized CUDA** binaries on your GPU if a CUDA-compatible
graphics card is available [#f1]_.

ELEKTRONN2 is written in Python (2.7 / and 3.4+) and is a complete rewrite of the
previously published `ELEKTRONN <http://elektronn.org>`_ library. The largest
improvement is the development of a **functional interface that allows easy creation of**
**complex data-flow graphs** with loops between arbitrary points in contrast to
simple "chain"-like models.
Currently, the only supported platform is Linux (x86_64).

.. [#f1] You can find out if your graphics card is compatible
  `here <https://developer.nvidia.com/cuda-gpus>`_.
  Usage on systems without CUDA is possible but generally not recommended
  because it is very slow.


Use cases
---------

Although other high-level libraries are available (Keras, Lasagne), they all
lacked desired features_ and flexibility for our work,
mostly in terms of an intuitive method to specify complicated computational
graphs and utilities for training and data handling, especially in the domain
of specialised large-scale 3-dimensional image data analysis for
`connectomics <https://en.wikipedia.org/wiki/Connectomics>`_ research
(e.g. tracing, mapping and segmenting neurons in in SBEM [#f2]_ data sets).

Although the mentioned use cases are ELEKTRONN2's specialty, it can be used and
extended for a wide range of other tasks thanks to its modular object-oriented
API design_ (for example, new operations can be implemented as subclasses of the
`Node <http://elektronn2.readthedocs.io/en/latest/source/elektronn2.neuromancer.html#elektronn2.neuromancer.node_basic.Node>`_
class and seamlessly integrated into neural network models).

.. [#f2] `Serial Block-Face Scanning Electron Microscopy <http://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0020329>`_,
  a method to generate high resolution 3D images from small samples, such as
  brain tissue.

.. figure:: https://raw.githubusercontent.com/ELEKTRONN/ELEKTRONN2/master/docs/_images/j0126_blend.jpg

  Example visualisation of ELEKTRONN2's usage on a 3D SBEM data set
  (blending input to output from left to right).

  +--------------------------------------------------+------------------------------------------------------------+
  | Left (input: raw data)                           | Right (output: predictions by ELEKTRONN2, color-coded)     |
  +==================================================+============================================================+
  | 3D electron microscopy images of a zebra finch   | Probability of barriers (union of cell boundaries and      |
  | brain (area X dataset j0126 by Jörgen Kornfeld). | extracellular space, marked in **white**) and mitochondria |
  |                                                  | (marked in **blue**) predicted by ELEKTRONN2.              |
  +--------------------------------------------------+------------------------------------------------------------+


Installation
------------

See the installation instructions at
https://elektronn2.readthedocs.io/en/latest/installation.html.


.. _design:

Design principles
=================

ELEKTRONN2 adds another abstraction layer to Theano. To create a model, the
user has to connect different types of node objects and thereby builds a graph
as with Theano. But the creation of the raw Theano graph, composed of symbolic
variables and trainable model parameters, is hidden and managed through usage
of sensible default values and bundling of stereotypical Theano operations into
a single ELEKTRONN2 node.
For example, creating a convolution layer consists of initialising weights,
performing the convolution, adding the bias, applying the activation function
and optional operations such as dropout or batch normalisation. Involved
parameters might be trainable (e.g. convolution weights) or non-trainable but
changeable during training (e.g. dropout rates).


.. _features:

Features
========


Operations
----------

* Perceptron / fully-connected / dot-product layer, works for arbitrary
  dimensional input
* Convolution, 1-,2-,3-dimensional
* Max/Average Pooling, 1,2,3-dimensional
* UpConv, 1,2,3-dimensional
* Max Fragment Pooling (MFP), 1,2,3-dimensional
* Gated Recurrent Unit (GRU) and Long Short Term Memory (LSTM) unit
* Recurrence / Scan over arbitrary sub-graph: support of multiple inputs
  multiple outputs and feed-back of multiple values per iteration
* Batch normalisation with automatic accumulation of whole data set statistics
  during training
* Gaussian noise layer (for Variational Auto Encoders)
* Activation functions: tanh, sigmoid, relu, prelu, abs, softplus, maxout,
  softmax-layer
* Local Response Normalisation (LRN), feature-wise or spatially
* Basic operations such as concatenation, slicing, cropping, or element-wise
  functions


Loss functions
--------------

* Bernoulli / Multinoulli negative log likelihood
* Gaussian negative log likelihood
* Squared Deviation Loss, (margin optional)
* Absolute Deviation Loss, (margin optional)
* Weighted sum of losses for multi-task training


Optimisers
----------

* Stochastic Gradient Descent (SGD)
* AdaGrad
* AdaDelta
* Adam


Trainer
-------

* Automatic creation of training directory to which all files (parameters,
  log files, previews etc.) will be saved
* Frequent printing and logging of current state, iteration speed etc.
* Frequent plotting of monitored states (error samples on training and
  validation data, classification errors and custom monitoring targets)
* Frequent saving of intermediate parameter states and history of monitored
  variables
* Frequent preview prediction images for CNN training
* Customisable schedules for non-trainable meta-parameters (e.g. dropout rates,
  learning rate, momentum)
* Fully functional python command line during training, usable for
  debugging/inspection (e.g. of inputs, gradient statistics) or for changing
  meta-parameters


Training Examples for CNNs
--------------------------

* Randomised patch extraction from a list of of input/target image pairs
* Data augmentation trough histogram distortions, rotation, shear, stretch,
  reflection and perspective distortion
* Real-time data augmentation through a queue with background threads.


Utilities
---------

* Array interface for `KNOSSOS <https://knossostool.org/>`_ data sets with
  caching, pre-fetching and support for multiple data sets as channel axis.
* Viewer for multichannel 3-dimensional image arrays within the Python runtime
* Function to convert ID images to boundary images
* Utilities needed for skeltonisation agent training and application
* Visualisation of the computational graph
* Class for profiling within loops
* KD Tree that supports append (realised through mixture of KD-Tree and
  brute-force search and amortised rebuilds)
* Daemon script for the synchronised start of experiments on several hosts,
  based on resource occupation.


Documentation and usage examples
================================

The documentation is hosted at `<https://elektronn2.readthedocs.io/>`_
(built automatically from the sources in the ``docs/`` subdirectory of the
code repository).


Contributors
============

* `Marius Killinger <https://github.com/xeray>`_ (main developer)
* `Martin Drawitsch <https://github.com/mdraw>`_
* `Philipp Schubert <https://github.com/pschubert>`_

ELEKTRONN2 was funded by `Winfried Denk's lab <http://www.neuro.mpg.de/denk>`_
at the Max Planck Institute of Neurobiology.

`Jörgen Kornfeld <http://www.neuro.mpg.de/person/43611/3242677>`_
was academic advisor to this project.


License
=======

ELEKTRONN2 is published under the terms of the GPLv3 license.
More details can be found in the `LICENSE.txt
<https://github.com/ELEKTRONN/ELEKTRONN2/blob/master/LICENSE.txt>`_ file.
