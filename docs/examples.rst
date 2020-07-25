###############
Tutorial
###############

Solving a problem with ``projSplitFit`` requires the following fundamental steps:

#.  Create an empty object from the ``ProjSplitFit`` class
#.  Add data to set up the object's data/loss term
#.  Add regularizers to the object
#.  Run the optimization
#.  Retrieve the solution and/or optimal value.

This chapter gives simple examples of each of these operations.  Full
descriptions of the methods used are in the following chapter.

Note that ``projSplitFit`` with a lower-case initial 'p' denotes the name of
the ``projSplitFit`` Python package, whereas ``ProjSplitFit`` with an
upper-case initial 'P' denotes the primary class defined in that package.


Basic Setup with a Quadratic Loss Term
=======================================================================

Assume that the matrix :math:`A` is a 2D ``NumPy`` array whose rows are the
observations of some dataset and :math:`y` is a list or 1D ``NumPy`` array
containing the corresponding response values. Consider the classical
least-squares problem defined as

.. math::
  \min_{z\in\mathbb{R}^d}\frac{1}{2n}\|Az - y\|^2_2
  :label: eqLS

to solve this problem with ``projSplitFit``, one would use
the following code ::

  import projSplitFit as ps
  projSplit = ps.ProjSplitFit()
  projSplit.addData(A,y,loss=2,intercept=False)
  projSplit.run()
  optimalVal = projSplit.getObjective()
  z = projSplit.getSolution()

The first line after the ``import`` statement calls the contructor to set up
an empty ``ProjSplitFit`` object.  Next, the invocation of the ``addData``
method provides the object with the model data and defines the loss term.

In the ``addData`` call, the argument ``loss`` is set to 2 in order to use the
:math:`\ell_2^2` loss. Other possible choices are any :math:`p > 1` for the
:math:`\ell_p^p` loss and the string "logistic" for the logistic loss. The
user may also define their own loss via the ``losses.LossPlugIn`` class
(see below).  The ``intercept=False`` argument specifies that the model
does not have an intercept (constant) term.

This classical model has no regularizers, so it is not necessary to add
regularizers.  The ``run`` method then solves the optimization problem. After
solving the problem, the ``getObjective`` method returns the optimal solution
value and the ``getSolution`` value returns the solution vector :math:`z`.

Dual Scaling
=============

The dual scaling parameter, called :math:`\gamma` in most projective splitting
papers, plays an important role in the empirical convergence rate of the
method. It must be selected carefully. There are two ways to set
:math:`\gamma`. It may be set when calling the ``projSplitfit`` constructor, as in::

  projSplit = ps.ProjSplitFit(dualScaling=gamma)

(the default value is 1).  It may also modified later through the
``setDualScaling`` method::

  projSplit.setDualScaling(gamma)


Including an Intercept Variable
================================

It is common in machine learning to fit an intercept for a linear model. That is, instead of solving
:eq:`eqLS` solve

.. math::
  \min_{z_0\in\mathbb{R},z\in\mathbb{R}^d}\frac{1}{2n}\|z_0 e + Az - y\|^2

where :math:`e` is a vector of all ones. To do this, set the ``intercept`` argument to
the ``addData`` method to ``True`` (which is the default). Note that regularizers
never apply to the intercept variable.


Normalization
================================

The performance of first-order methods is effected by the scaling of the
features. A common tactic to improve performance is to scale the features so
that they have commensurate size. This is controlled by setting the
``normalize`` argument of ``addData`` to ``True`` (which is the default). If this
is done, then the observations matrix :math:`A` is copied and the columns of
the copy are normalized to have unit :math:`\ell_2` norm.


Adding a Regularizer
================================

A common strategy in machine learning is to add a regularizer to the model. Consider the lasso

.. math::
  \min_{z\in\mathbb{R}^d}\frac{1}{2n}\|Az - y\|^2 +\lambda_1\|z\|_1 ,
  :label: lasso


where :math:`\|z\|_1=\sum_i |z_i|`. To solve this model instead, we call the 
``addRegularizer`` method of the ``ProjSplitFit`` object before invoking 
``run()``::

  from regularizers import L1
  regObj = L1(scaling=lam1)
  projSplit.addRegularizer(regObj)

The built-in method ``L1`` returns an object derived from the class
``regularizers.Regularizer`` The ``regularizers.Regularizer`` class may be
used to describe any convex function to be used as a regularizer. Other
built-in regularizers include ``regularizers.L2sq`` which creates the
regularizer :math:`0.5\|x\|_2^2` and ``regularizers.L2``, which creates the
regularizer :math:`\|x\|_2`.

To recap, the entire code to solve :eq:`lasso` with
:math:`\lambda_1=0.1` and the default dual scaling of :math:`\gamma=1` is ::

  import projSplitFit as ps
  from regularizers import L1
  lam1 = 0.1
  projSplit = ps.ProjSplitFit()
  projSplit.addData(A,y,loss=2,intercept=False)
  regObj = L1(scaling=lam1)
  projSplit.addRegularizer(regObj)
  projSplit.run()
  optimalVal = projSplit.getObjective()
  z = projSplit.getSolution()

If an intercept variable is desired, the keyword argument ``intercept`` should
be set to ``True`` or omitted.



User-Defined and Multiple Regularizers
========================================

In addition to these built-in regularizers, the user may define their own. In
``projSplitFit``, a regularizer is defined by a ``prox`` method and a
``value`` method. The ``prox`` method must be defined. The ``value`` method is
optional and is only used if the user specifies calculation of function values
for performance tracking, or uses the ``getObjective`` method. The ``prox``
method returns the proximal operator of :math:`\sigma f`, where :math:`f` is
the regularizer function and :math:`\sigma` is a positive scaling factor. That
is, the ``prox`` method should be defined so that

.. math::
  f.\mathtt{prox(}t,\sigma\mathtt{)} = \text{prox}_{\sigma f}(t)=\arg\min_x\left\{ \sigma f(x) + \frac{1}{2}\|x-t\|^2_2\right\}.
  :label: proxDef

The ``prox`` method should expect its first argument to be a 1D ``numpy``
array and its second argument to be a positive ``float``; it should return a ``numpy`` array of the same dimensions as the first argument. 

The ``value`` method
*f*\ ``.value``\ (:math:`x`), if defined, should simply returns the function value
:math:`f(x)`; it should expect its argument to be a 1D ``numpy`` array and
return a ``float``.

Using multiple regularizers in ``projSplitFit`` is straightforward:  one simply
calls ``addRegularizer`` multiple times before calling ``run``. Suppose one
wants to solve the lasso with an additional constraint that each component of
the solution must be non-negative.  That is, one wishes to solve

.. math::
  \min_{z\in\mathbb{R}^d, z\geq 0}\frac{1}{2n}\|Az - y\|^2 +\lambda_1\|z\|_1.
  :label: posLasso

The non-negativity constraint can be formulated as a second regularizer. That is, one may rewrite :eq:`posLasso` as

.. math::
  \min_{z\in\mathbb{R}^d}\frac{1}{2n}\|Az - y\|^2 +\lambda_1\|z\|_1 + g(z) ,

where

.. math::
  g(z)=\left\{
  \begin{array}{ll}
    +\infty & \text{if }z_i<0\text{ for any } i\\
    0 & \text{otherwise.}
  \end{array}
  \right.

The proximal operator :eq:proxDef for this function is simply projection onto
the nonnegative orthant, and is independent of :math:`\sigma`. To include this
regularizer in ``projSplitFit`` object, one defines the regularizer object for
:math:`g` and then adds it to the model with ``addRegularizer``.  These
operations may be accomplished as follows:

.. raw:: latex

   \newpage

::

  from regularizer import Regularizer
  def prox_g(z,sigma):
    return (z>=0)*z
  def value_g(x):
    if any(x < 0):
       return float('Inf')
    return 0.0
  regObjNonneg = Regularizer(prox=prox_g, value=value_g)
  projSplit.addRegularizer(regObjNonneg)

Note that ``prox`` function must still have a second argument ``sigma`` even
in cases, like this one, where the returned value is independent of ``sigma``.

In summary, the entire code to solve :eq:`posLasso` with (for example)
:math:`\lambda_1 = 0.1` and the default dual scaling of :math:`\gamma=1` would
be ::

  import projSplitFit as ps
  from regularizers import L1, Regularizer

  def prox_g(z,sigma):
    return (z>=0)*z

  def value_g(x):
    if any(x < -1e-7):
       return float('Inf')
    return 0.0

  lam1 = 0.1

  projSplit = ps.ProjSplitFit()
  projSplit.addData(A,y,loss=2,intercept=False)
  regObj = L1(scaling=lam1)
  projSplit.addRegularizer(regObj)
    regObjNonneg = Regularizer(prox=prox_g, value=value_g)
  projSplit.addRegularizer(regObjNonneg)
  projSplit.run()
  optimalVal = projSplit.getObjective()
  z = projSplit.getSolution()

Here, for numerical reasons, we have slightly modified the ``value_g``
function to treat very small-magnitude negative numbers as if they were zero.


Linear Operator Composed with a Regularizer
============================================

Sometimes, one would like to compose a regularizer with a linear operator. Total variation deblurring is an example of such a situation. ``ProjSplitFit`` handles this with ease.
Consider the problem

.. math::
  \min_{z\in\mathbb{R}^d}\frac{1}{2n}\|Az - y\|^2 +\lambda_1\|G z\|_1

for some linear operator  or matrix :math:`G`. The linear operator can be added
as an argument to the ``addRegularizer`` method as follows, assuming the
matrix variable ``G`` has been defined::

  regObj = L1(scaling=lam1)
  projSplit.addRegularizer(regObj,linearOp=G)

:math:`G` must be a 2D ``numpy`` array ``scipy`` linear operator.   If
:math:`G` is an array The number of columns of
:math:`G` must equal the dimension of the solution vector :math:`z`.

Documentation for ``scipy`` linear operators may be found in the package
``scipy.sparse.linalg``.  When used with ``projSplitFit``, such operators
should have a ``shape`` :math:`(m,n)` and define the methods ``matvec`` and
``rmatvec``, which respectively compute the actions of the linear operator and
its adjoint (the equivalent of multiplication by the matrix transpose).
Consider the 1D total variation operator on :math:`\mathbb{R}^n` given by

.. math::
   [x_1 \;\;\; x_2 \;\;\; \cdots \;\;\; x_n] \;\;\; \mapsto \;\;\;
   [x_1 - x_2 \;\;\; x_2 - x_3 \;\;\; \cdots \;\;\; x_{n-1} - x_n].

The adjoint of this operator is the map

.. math::
   [u_1 \;\;\; u_2 \;\;\; \cdots \;\;\; u_{n-1}] \;\;\; \mapsto \;\;\;
   [u_1 \;\;\; u_2 - u_1 \;\;\; 
               u_3 - u_2 \;\;\; \cdots \;\;\; u_{n-1} - u_{n-2} \;\;\; -u_{n-1}].

Calling ``varop1d(n)`` as defined in the code below will create such an operator::

   import numpy
   import scipy

   def applyOperator(x):
      return x[:(len(x)-1)] - x[1:]

   def applyAdjoint(u):
      v = numpy.zeros(len(u) + 1)
      for i in range(len(u)):
        v[i] += u[i]
        v[i+1] -= u[i]
      return v

   def varop1d(n):
      return scipy.sparse.linalg.LinearOperator(shape=(n-1,n),
                                                matvec=applyOperator,
                                                rmatvec=applyAdjoint)


User-Defined Losses
====================

Just as the you may define their own regularizers, you may define your own
loss function, using the ``losses.LossPlugIn`` class. Objects of this class
can be passed into ``addData`` as the ``loss`` argument. To define a loss, you
need to define its ``derivative`` method. Optionally, you may also define its
``value`` method if you would like to compute function values (either for
performance tracking or to call the ``getObjective`` method).

For example, consider the one-sided :math:`\ell_2^2` loss:

.. math::
  \ell(x,y) =
  \left\{
  \begin{array}{ll}
    0 & \text{if }x\leq y\\
    \frac{1}{2}(x-y)^2 &\text{otherwise.}
  \end{array}
  \right.

To use this loss, you would proceed as follows::

  import losses as ls

  def deriv(x,y):
    return (x>=y)*(x-y)
  def val(x,y):
    return (x>=y)*(x-y)**2

  loss = ls.LossPlugIn(derivative=deriv,value=val)
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
