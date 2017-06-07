******************
Making Predictions
******************


Predicting with a trained model
===============================

Once you have found a good neural network architecture for your task
and you have already trained it with ``elektronn2-train``, you can load
the saved model from its training directory and make predictions on
arbitrary data in the correct format.
In general, use :py:meth:`elektronn2.neuromancer.model.modelload()` to load a
trained model file and
:py:meth:`elektronn2.neuromancer.model.Model.predict_dense()` to create
*dense* predictions for images/volumes of arbitrary sizes (the input
image must however be larger than the input patch size). Normally,
predictions can only be made with some offset w.r.t. the input image
extent (due to the convolutions) but this method provides an option to mirror
the raw data such that the returned prediction covers the full extent of
the input image (this might however introduce some artifacts because
mirroring is not a natural continuation of the image).

For making predictions, you can write a custom prediction script. Here is
a small example for its content:

.. code-block:: python

  # TODO: Example prediction script

.. TODO: Mention/explain non-image predictions?

Optimal patch sizes
===================

Prediction speed benefits greatly from larger input patch sizes and MFP
(see below). It is recommended to impose a larger patch size when making
predictions by loading an already trained model with with the
``imposed_patch_size`` argmument:

.. code-block:: python

  # TODO: Example here, something like "model = modelload(model_file, imposed_patch_size=...)"

To find an optimal patch size that works on your hardware, you can use the
``elektronn2-profile`` command, which varies the input size of a given
network model until the RAM limit is reached. The script creates a
CSV table of the respective speeds. You can find the fastest input size that
just fits in your RAM in that table and use it to make predictions.

Theoretically, predicting the whole image in a single patch, instead of
several tiles, would be fastest. For each tile some calculations have to be
repeated and the larger the tiles, the more intermediate results can be
shared. But this is obviously impossible due to limited GPU-RAM.


.. note::
  GPU-RAM usage can be lowered by enabling garbage collection (set
  ``linker = cvm`` in the ``[global]`` section of ``.theanorc``) and by using cuDNN.



.. _mfp:

Max Fragment Pooling (MFP)
==========================

MFP is the computationally optimal way to avoid redundant calculations when
making predictions with strided output (as arises from pooling).
It requires more GPU RAM (you may need to adjust the input size) but it can
speed up predictions by a factor of 2 - 10. The larger the patch size (i.e.
the more RAM you have) the faster. Compilation time is significantly longer.

.. TODO Explain why it's fast and how it works ###TODO
