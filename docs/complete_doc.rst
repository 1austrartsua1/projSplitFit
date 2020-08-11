
###########################
Detailed Documentation
###########################


ProjSplitFit Class
===================

.. autoclass:: projSplitFit.ProjSplitFit
   :members:

   .. automethod:: __init__

Regularizer Class
==================

.. autoclass:: regularizers.Regularizer
   :members:

   .. automethod:: __init__

Built-in Regularizers
======================

.. autofunction:: regularizers.L1

.. autofunction:: regularizers.L2sq

.. autofunction:: regularizers.L2

User-Defined Losses (LossPlugIn Class)
=========================================

.. autoclass:: losses.LossPlugIn
  :members:

  .. automethod:: __init__


Loss Processors
=================

These classes instruct projective splitting how to process the blocks of variables
associated with the loss, and are added in the ``ProjSplitFit.addData`` method
as the ``process`` input.

If you leave the ``process`` argument to ``ProjSplitFit.addData`` unused, then ``ProjSplitFit`` will use the
default loss processor, ``Forward2Backtrack``.

Forward-step (Gradient) Based Loss Processors
--------------------------------------------------

Forward2Fixed
^^^^^^^^^^^^^^^^^

.. autoclass:: lossProcessors.Forward2Fixed
  :members:

  .. automethod:: __init__

Forward2Backtrack
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: lossProcessors.Forward2Backtrack
  :members:

  .. automethod:: __init__

Forward2Affine
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: lossProcessors.Forward2Affine
  :members:

  .. automethod:: __init__

Forward1Fixed
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: lossProcessors.Forward1Fixed
  :members:

  .. automethod:: __init__

Forward1Backtrack
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: lossProcessors.Forward1Backtrack
  :members:

  .. automethod:: __init__

Backward-Step (Proximal) Based Loss Processors
------------------------------------------------

Backward Exact
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: lossProcessors.BackwardExact
  :members:

  .. automethod:: __init__

Backward Step with Conjugate Gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: lossProcessors.BackwardCG
  :members:

  .. automethod:: __init__

Backward Step with L-BFGS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: lossProcessors.BackwardLBFGS
  :members:

  .. automethod:: __init__


Other Methods
----------------
Each loss processor object also inherits the following useful methods.

.. autofunction:: lossProcessors.LossProcessor.getStep

.. autofunction:: lossProcessors.LossProcessor.setStep
