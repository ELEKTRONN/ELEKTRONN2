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
a small example:


Prediction example for ``neuro3d``
----------------------------------

(Optional) If you want to predict on a GPU that is not already assigned in
``~/.theanorc`` or ``~/.elektronn2rc``, you need to initialise it
**before** the other imports::

  from elektronn2.utils import initgpu
  initgpu('auto')  # or "initgpu('0')" for the first GPU etc.

Find the save directory and choose a model file (you probably want the
one called ``<save_name>-FINAL.mdl``) and a file that contains the raw
images on which you want to execute the neural network, e.g.::

  model_path = '~/elektronn2_examples/neuro3d/neuro3d-FINAL.mdl'
  raw_path = '~/neuro_data_zxy/raw_2.h5'  # raw cube for validation

For loading data from (reasonably small) hdf5 files, you can use the
:py:meth:`h5load() <elektronn2.utils.utils_basic.h5load()>` utility
function. Here we load the 3D numpy array called `'raw'`` from the
input file::

  from elektronn2.utils import h5load
  raw3d = h5load(raw_path, 'raw')

Now we load the neural network model::

  from elektronn2 import neuromancer as nm
  model = nm.model.modelload(model_path)

Input sizes should be at least as large as the spatial input shape of
the model's input node (which you can query by
``model.input_node.shape.spatial_shape``). Smaller inputs are
automatically padded. Here we take an arbitrary ``32x160x160``
subvolume of the raw data to demonstrate the predictions::

  raw3d = raw3d[:32, :160, :160]

.. TODO: Explain in general why we need (f,z,x,y)

To match the input node's expected input shape, we need to prepend an empty
axis for the single input channel. An empty axis is sufficient because we
trained with only 1 input channel here (the uint8 pixel intensities)::

  raw4d = raw3d[None, :, :, :]  # shape: (f=1, z=32, x=160, y=160)

  pred = model.predict_dense(raw4d)

.. TODO: Link to complete copy-pastable example "predict.py"? Or even automate (templated) predict.py creation in save_dir and refer to it?

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

.. TODO Explain why it's fast and how it works
