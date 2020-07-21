###############
Tutorial
###############

Adding Data
==============

Consider the least-squares problem defined as

.. math::
  \min_{z\in\mathbb{R}^d}\frac{1}{2n}\|Az - y\|^2_2
  :label: eqLS

Assuming the matrix :math:`A` is a 2D *NumPy* array, and :math:`y` is a 1D *NumPy* array, or list, then
to solve this problem with projSplitFit, use
the following code ::

  import projSplitFit as ps
  projSplit = ps.ProjSplitFit()
  projSplit.addData(A,y,loss=2,intercept=False)
  projSplit.run()

The argument ``loss`` is set to 2 in order to use the :math:`\ell_2^2` loss. Other possible choices
are any :math:`p> 1` for the :math:`\ell_p^p` loss and the string "logistic" for the logistic loss.
The user may also define their own loss via the ``losses.LossPlugIn`` class (see below).

Dual Scaling
=============

The dual scaling parameter, called :math:`\gamma` in most projective splitting papers,
plays an important role in the empirical convergence rate of the method. It must be selected carefully.
There are two ways to set :math:`\gamma`. Set it when calling the constructor::

  projSplit = ps.ProjSplitFit(dualScaling=gamma)

(the default value is 1), or via the ``setDualScaling`` method::

  projSplit.setDualScaling(gamma)

Including an Intercept Variable
================================

It is common in machine learning to fit an intercept for a linear model. That is, instead of solving
:eq:`eqLS` solve

.. math::
  \min_{z_0\in\mathbb{R},z\in\mathbb{R}^d}\frac{1}{2n}\|z_0 e + Az - y\|^2

where :math:`e` is a vector of all ones. To do this, set the ``intercept`` argument to
the ``addData`` method to True (which is the default). Note that added regularizers
never apply to the intercept variable.

Normalization
================================

The performance of first-order methods is effected by the scaling of the features. A common tactic
to improve performance is to scale the features so that they have commensurate size. This is controlled
by setting the ``normalize`` argument of ``addData`` to True (which is the default). If this is done,
then the observations matrix :math:`A` is copied and the columns of the copy are normalized
to have unit :math:`\ell_2` norm.

Adding a Regularizer
================================

A common strategy in machine learning is to add a regularizer to the model. Consider the lasso

.. math::
  \min_{z\in\mathbb{R}^d}\frac{1}{2n}\|Az - y\|^2 +\lambda_1\|z\|_1


where :math:`\|z\|_1=\sum_i |z_i|`. To solve this model instead, before calling ``run()`` we can invoke the
``addRegularizer`` method::

  from regularizers import L1
  regObj = L1(scaling=lam1)
  projSplit.addRegularizer(regObj)
  projSplit.run()

The built-in method ``L1`` returns an object of class ``regularizers.Regularizer`` which may be used
to describe any convex function to be used as a regularizer. Other built-in regularizers include
``regularizers.L2sq`` which creates the regularizer :math:`0.5\|x\|_2^2` and ``regularizers.L2``,
which creates the regularizer :math:`\|x\|_2`.

User-Defined and Multiple Regularizers
========================================

In addition to these built-in regularizers, the user may define their own. In *ProjSplitFit*, a regularizer is defined
by a *prox* method and a *value* method. The *prox* method must be defined. The *value* method
is optional and is only used if the user wants to calculate function values for performance tracking.
The *prox* method returns the proximal operator for the function scaled by some amount.
That is

.. math::
  \text{prox}_{\sigma f}(t)=\arg\min_x\left\{ \sigma f(x) + \frac{1}{2}\|x-t\|^2_2\right\}.

The value function simply returns the value :math:`f(x)`. Both of these functions must
handle NumPy arrays. Value must return a float and prox must return a NumPy array
with the same length as the input.

Adding multiple regularizers in *projSplitFit* is easy. Suppose one wants to solve
the lasso with an additional constraint that each component of the solution must be non-negative.
That is solve

.. math::
  \min_{z\in\mathbb{R}^d, z\geq 0}\frac{1}{2n}\|Az - y\|^2 +\lambda_1\|z\|_1.

The non-negativity constraint can be thought of as another regularizer. That is

.. math::
  \min_{z\in\mathbb{R}^d}\frac{1}{2n}\|Az - y\|^2 +\lambda_1\|z\|_1 + g(z)

where

.. math::
  g(z)=\left\{
  \begin{array}{cc}
    \infty & \text{if some }z_i<0\\
    0 & \text{else}
  \end{array}
  \right.

To solve this problem with *projSplitFit* the user must define the regularizer object
for :math:`g` and then add it to the model with ``addRegularizer``. This is done as
follows::

  from regularizers import Regularizer
  def prox_g(z,sigma):
    return (z>=0)*z
  regObj = Regularizer(prox_g)
  projSplit.addRegularizer(regObj)
  projSplit.run()

The proximal operator is just the projection onto the constraint set.
Note that ``prox_g`` must still have a second argument for the scaling even though
for this particular function it is not used.


Linear Operator Composed with a Regularizer
============================================

Sometimes, one would like to compose a regularizer with a linear operator. This occurs
in Total Variation deblurring for example. *ProjSplitFit* handles this with ease.
Consider the problem

.. math::
  \min_{z\in\mathbb{R}^d}\frac{1}{2n}\|Az - y\|^2 +\lambda_1\|G z\|_1

for some linear operator (matrix) :math:`G`. The linear operator can be added as an
argument to the ``addRegularizer`` method as follows::

  regObj = L1(scaling=lam1)
  projSplit.addRegularizer(regObj,linearOp=G)
  projSplit.run()

:math:`G` must be a 2D NumPy array (or similar). The number of columns of
:math:`G` must equal the number of primal variables,
as defined by the matrix :math:`A` which is input to ``addData``. If not, *ProjSplitFit*
will raise an Exception.

User-Defined Losses
====================

Just as the user may define their own regularizers, they may define their own loss. This is achieved
via the ``losses.LossPlugIn`` class. Objects of this class can be passed into ``addData`` as the ``process``
argument. To define a loss, one needs to define its derivative method. Optionally, one may also define
its value method if one would like to compute function values for performance tracking.

For example, consider the one-sided :math:`\ell_2^2` loss:

.. math::
  \ell(x,y) =
  \left\{
  \begin{array}{cc}
    0 & x\leq y\\
    \frac{1}{2}(x-y)^2 &\text{ else}
  \end{array}
  \right.

To use this loss::

  import losses as ls

  def deriv(x,y):
    return (x>=y)*(x-y)
  def val(x,y):
    return (x>=y)*(x-y)**2

  loss = ls.LossPlugIn(deriv,val)
  projSplit.addData(A,y,loss=loss)


Complete Example: Rare Feature Selection
==========================================

Let's look at a complete example from page 34 of our paper :cite:`coco`. The problem of interest is

.. math::
  \min_{\substack{\gamma_0\in \mathbb{R} \\ \gamma\in \mathbb{R}^{|\mathcal{T}|}}}
  \left\{
  \frac{1}{2n}\|\gamma_0 e + X H\gamma - y\|_2^2
  +
  \lambda
  \big(
  \mu\|\gamma_{-r}\|_1
  +
  (1-\mu)\|H\gamma\|_1
  \big)
  \right\}

First let's deal with the loss. The loss is the :math:`\ell_2^2` loss. Note that
it is composed with a linear operator :math:`H`. There are two ways to deal with this.
If the size of the matrices is not too much of a concern, one may pre-compute a
new observation matrix as ``Xnew = X*H``. If this is prohibitive, the linear operator
can be composed with the loss, meaning the *ProjSplitFit* handles it
internally and does not explicitly compute the matrix product.
This option is controlled via the ``linearOp`` argument to ``addData``.

Taking this option, the loss is dealt with as follows::

  import projSplitFit as ps
  projSplit = ps.ProjSplitFit()
  projSplit.addData(X,y,loss=2,linearOp=H,normalize=False)

Note that, by default, the intercept term :math:`\gamma_0` is added.

The first regularizer needs to be custom-coded, as it leaves out the first variable,
which is the root of the tree. It is dealt with as follows::

  from regularizers import Regularizer
  def prox(gamma,sigma):
    temp = numpy.zeros(gamma.shape)
    temp[1:] = (gamma[1:]>sigma)*(gamma[1:]-sigma)
    temp[1:] += (gamma[1:]<-sigma)*(gamma[1:]+sigma)
    temp[0]=gamma[0]
    return temp
  regObj = Regularizer(prox,scaling=lam*mu)
  projSplit.addRegularizer(regObj)

The second regularizer is more straightforward and may be dealt with via the
built-in ``L1`` function and composing with the linear operator :math:`H`
as follows::

  from regularizers import L1
  regObj2 = L1(scaling=lam*(1-mu))
  projSplit.addRegularizer(regObj2,linearOp=H)

Finally we are ready to run the method via::

  projSplit.run()

One can obtain the final objective value and solution via::

  optimalVal = projSplit.getObjective()
  gammastar = projSplit.getSolution()

Loss Process Objects
=====================
Projective splitting comes with a rich array of ways to update the hyperplane
at each iteration. In the original paper :cite:`proj1`, the computation was based
on the *prox*. Since then, several new calculations have been devised based on
*forward steps*, i.e. *gradient* calculations, making projective splitting a
true first-order method :cite:`for1`, :cite:`coco`.

In *ProjSplitFit*, there are a large number of options for which update method to
use with respect to the blocks of variables associated with the *loss*.
This is controlled by the ``process`` argument to the ``addData`` method.
This argument must be a class derived from ``lossProcessors.LossProcessor``.
*ProjSplitFit* supports the following built-in loss processing classes defined in ``lossProcessors.py``:

* ``Forward2Fixed`` two-forward-step update with fixed stepsize, see :cite:`for1`
* ``Forward2Backtrack`` two-forward-step update with backtracking stepsize, see :cite:`for1`.
  Note this is the *default* loss processor if the `process` argument is ommitted from
  ``addData``
* ``Forward2Affine`` two-forward-step with the affine trick, see :cite:`for1`. Only available
  when ``loss=2``
* ``Forward1Fixed`` one-forward-step with fixed stepsize, see :cite:`coco`
* ``Forward1Backtrack`` one-forward-step with backtracking stepsize, see :cite:`coco`
* ``BackwardExact`` Exact backward step for :math:`\ell_2^2` loss via matrix inversion.
  Only available with ``loss=2``
* ``BackwardCG`` Backward step via conjugate gradient, only available when ``loss=2``
* ``BackwardLBFGS`` Backward step via LBFGS solver.

To select a loss processor, one creates an object of the appropriate class from above,
calling the constructor with the desired parameters, and then passes the object
into ``addData`` as the ``process`` argument. For example, to use ``BackwardLBFGS``::

  import lossProcessors as lp
  processObj = lp.BackwardLBFGS()
  projSplit.addData(A,y,loss=2,process=processObj)

This will use BackwardLBFGS with all of the default parameters. See the detailed documentation
for all of the possible parameters and settings for each loss process class.

The user may wish to define their own loss process classes. They must derive from
``lossProcessors.LossProcessor`` and they must implement the ``initialize``
and ``update`` methods. Of course, convergence cannot be guaranteed unless the user
knows of a supporting mathematical theory for their process update method.

Embedding Regularizers
=======================

Projective splitting handles regularizers via their proxes. A regularizer is typically
handled by including a new block of variables. However, it is possible to embed one
regularizer into the block that handles the loss. In this case, the loss is handled
in a forward-backward manner, with the forward step calculated, and then the backward step
on the same block of variables. For example, with ``Forward2Fixed`` and embedding
the update would be

.. math::
  x_i^k = \text{prox}_{\rho g}(z^k - \rho (\nabla f_i(z^k)-w_i^k))

Note that the prox is computed in-line with the forward step.

To enable this option, use the ``embed`` argument to the ``addRegularizer`` call,
when adding the regularizer to the method.

If ``nblocks`` is greater than 1, the prox is performed on each block.

Options for the ``run()`` Method
==================================
The ``run`` method has several important options which we briefly discuss.
The first is ``nblocks``. This controls how many blocks projective splitting
breaks the loss into for processing. Recall the loss is

.. math::
  \frac{1}{n}\sum_{i=1}^n \ell (z_0 + a_i^\top H z,y_i)

An important property of projective splitting is *block iterativeness*: It does not
need to process every observation at each iteration. Instead, it may break the
:math:`n` observations into ``nblocks`` and process as few as one block at a time.
``nblocks`` may be anything from ``1``, meaning all observations are processed at
each iteration, to ``n``, meaning every observation is treated as a block.
``nblocks`` defaults to 1.

The blocks are contiguous runs of indices. If :math:`nblocks` does not divide the
number of rows/observations, then we use the formula

.. math::
  n = \lceil n/n_b \rceil n\%n_b + \lfloor n/n_b \rfloor(n_b - n \%n_b).

so that there are two groups of blocks, those with :math:`\lceil n/n_b\rceil`
number of indices and those with :math:`\lfloor n/n_b\rfloor`. That way,
the number of indices in any two blocks differs by at most 1.

The number of blocks processed per iteration is controlled via the argument ``blocksPerIteration``
which defaults to 1.

There are three ways to choose *which* blocks are processed at each iteration. This is
controlled with the ``blockActivation`` argument and may be set to

* "random", randomly selected block
* "cyclic", cycle through the blocks
* "greedy", (default) use the greedy heuristic of :cite:`for1` page 24 to select blocks.

Other Important Methods of ProjSplitFit
========================================

The ``keepHistory`` and ``historyFreq`` arguments to ``run()`` allow you to choose to record the progress of
the algorithm in terms of objective function values, running time, primal and dual residuals, and hyperplane values.
These may be extracted later via the ``getHistory()`` method.

``getObjective()`` simply returns the objective value at the current primal iterate.

``getSolution()`` returns the primal iterate :math:`z^k`. If the ``descale`` argument is set to True, then the
scaling vector used to scale each column of the data matrix is applied to the elements of :math:`z^k`.
That way, the coefficient vector can be used with unnormalized data such as new test data.
However the method ``getScaling()`` returns this scaling vector. This scaling vector can then be applied to normalize new test
data. To normalize a new test datapoint ``xtest``::

  scaling = projSplit.getScaling()
  x_test_normalized = xtest/scaling 
