
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

The loss processor classes instruct projective splitting how to process the
loss function.   The loss processor is specified by the ``process`` argument
of the ``ProjSplitFit.addData``.

If you omit the ``process`` argument to ``ProjSplitFit.addData``, then
``ProjSplitFit`` will use the default loss processor, ``Forward2Backtrack``.

When a loss processor for block :math:`i` is invoked within the projective
splitting algorithm, it is provided with the vector :math:`Hz^k` derived from
the current primal solution estimate :math:`z^k` (which just equals
:math:`z^k` if :math:`H` was not specified) and the dual solution estimate
:math:`w_i^k`.  It returns two vectors :math:`x_i^k` and :math:`y_i^k`, which
have should have the same dimension as :math:`w_i^k`.  These returned vectors
must have specific properties in order to guarantee convergence of the
algorithm; all the provided loss processor have these properties, with one
caveat mentioned below.

Forward-step (Gradient) Loss Processors
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
