###############
Tutorial
###############

Solving a problem with ``projSplitFit`` requires the following fundamental steps:

#.  Create an empty object from the ``ProjSplitFit`` class
#.  Add data to set up the object's data/loss term
#.  Add regularizers to the object
#.  Run the algorithm to solve the optimization problem
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

We assumed :math:`A` was a 2D ``NumPy`` array. However, ``ProjSplitFit`` also supports
*sparse* data matrices of a class derived from ``scipy.sparse.spmatrix``.
See `here <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_ for documentation
on sparse matrices in ``scipy``.

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

(the default value is 1).  It may also be modified later through the
``setDualScaling`` method::

  projSplit.setDualScaling(gamma)


Including an Intercept Variable
================================

It is common in machine learning to fit an intercept for a linear model. That is, instead of solving
:eq:`eqLS` solve

.. math::
  \min_{z_0\in\mathbb{R},z\in\mathbb{R}^d}\frac{1}{2n}\|z_0 e + Az - y\|^2

where :math:`e` is a vector of all ones of the same length as :math:`y`. To do this, set the ``intercept`` argument to
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
  projSplit.addData(A,y,loss=2,intercept=False,normalize=False)
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
the solution must be nonnegative.  That is, one wishes to solve

.. math::
  \min_{z\in\mathbb{R}^d, z\geq 0}\frac{1}{2n}\|Az - y\|^2 +\lambda_1\|z\|_1.
  :label: posLasso

One possible approach to solving this problem is to formulate the
nonnegativity constraint  as a second regularizer. That is, one may rewrite
:eq:`posLasso` as

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

The proximal operator :eq:`proxDef` for this function is simply projection onto
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
  projSplit.addData(A,y,loss=2,intercept=False,normalize=False)
  regObj = L1(scaling=lam1)
  projSplit.addRegularizer(regObj)
  regObjNonneg = Regularizer(prox=prox_g, value=value_g)
  projSplit.addRegularizer(regObjNonneg)
  projSplit.run()
  optimalVal = projSplit.getObjective()
  z = projSplit.getSolution()

Here, for numerical reasons, we have slightly modified the ``value_g``
function to treat very small-magnitude negative numbers as if they were zero.

Note that we present the code above mainly for purposes of example.  A
potentially more efficient approach to solving the nonnegative lasso problem
would be use a single user-defined regularizer of the form

.. math::

   h(x) = \left\{
          \begin{array}{ll}
          x, & \text{if } x \geq 0 \\
          +\infty, & \text{otherwise.}
          \end{array}
          \right.

This regularizer imposes both :math:`\ell_1` regularization and the nonnegativity
constraint, while having a proximal operation that is still easily evaluated.



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

:math:`G` must be a 2D ``numpy`` array, a ``scipy`` linear operator, or a ``scipy`` sparse matrix.   If
:math:`G` is an array, the number of columns of
:math:`G` must equal the dimension of the solution vector :math:`z`.

Documentation for ``scipy`` linear operators may be found in the package
``scipy.sparse.linalg``.  When used with ``projSplitFit``, such operators
should have a ``shape`` :math:`(m,n)` and define the methods ``matvec`` and
``rmatvec``, which respectively compute the actions of the linear operator and
its adjoint (the equivalent of multiplication by the matrix transpose).
Consider the 1D total variation operator :math:`\mathbb{R}^n \rightarrow
\mathbb{R}^{n-1}` given by

.. math::
   [x_1 \;\;\; x_2 \;\;\; \cdots \;\;\; x_n] \;\;\; \mapsto \;\;\;
   [x_1 - x_2 \;\;\; x_2 - x_3 \;\;\; \cdots \;\;\; x_{n-1} - x_n].

This map is equivalent to the action of :math:`n-1 \times n` matrix

.. math::

   V =
   \left[
   \begin{array}{cccccc}
   1 & - 1 \\
   & 1 & -1 \\
   && 1 & -1 \\
   &&& \ddots & \ddots \\
   &&&& 1 & -1
   \end{array}
   \right].

The adjoint of this operator is the map, equivalent to multiplication by the
transpose :math:`V^{{\scriptscriptstyle\top}}` of :math:`V`, is therefore

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
      return numpy.pad(u,(0,1)) - numpy.pad(u,(1,0))

   def varop1d(n):
      return scipy.sparse.linalg.LinearOperator(shape=(n-1,n),
                                                matvec=applyOperator,
                                                rmatvec=applyAdjoint)


User-Defined Losses
====================

Just as you may define your own regularizers, you may define your own
loss function, using the class ``losses.LossPlugIn``. Objects of this class
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

  loss = ls.LossPlugIn(derivative=deriv, value=val)
  projSplit.addData(A,y,loss=loss)


Complete Example: Rare Feature Selection
==========================================

Let's look at a complete example from page 34 of our paper :cite:`coco`, which originated from :cite:`YB18`.
The problem of interest is

.. math::
  \min_{\substack{\boldsymbol{\gamma_0}\in \mathbb{R} \\ \boldsymbol{\gamma}\in \mathbb{R}^{|\mathcal{T}|}}}
  \left\{
  \frac{1}{2n}\|\boldsymbol{\gamma_0} e + X H\boldsymbol{\gamma} - y\|_2^2
  +
  \lambda
  \big(
  \mu\|\boldsymbol{\gamma_{-r}}\|_1
  +
  (1-\mu)\|H\boldsymbol{\gamma}\|_1
  \big)
  \right\}

The loss function here is the :math:`\ell_2^2`, but with the regression
coefficients composed with a linear operator :math:`H`. There are two ways to
deal with such situations. If the size and density of the matrices is not of
great concern concern, one may pre-compute a new matrix through ``Xnew =
X*H``, and use ``Xnew`` as the observation matrix passed to ``projSplitFit``.
If forming :math:`XH` directly in this manner is prohibitive or causes an unacceptable increase in the
number of nonzero entries, the linear
operator can be instead composed with the loss, meaning that ``projSplitFit``
handles the composition internally and does not explicitly compute the matrix
product. This option is controlled via the ``linearOp`` argument to
``addData``.

Taking this option, and electing not to normalized the input data, one may set
up the loss term as follows::

  import projSplitFit as ps
  projSplit = ps.ProjSplitFit()
  projSplit.addData(X,y,loss=2,linearOp=H,normalize=False)

Note that, by default, the intercept term :math:`\boldsymbol{\gamma}_0` is incorporated into the loss.

The first regularizer does not apply to the root node variable of :math:`\boldsymbol{\gamma}`,
which is stored as the last entry of the vector.
A simple way to encode this is to treat it as the :math:`\ell_1` norm composed
with a linear operator which simply drops the last entry. That is,
:math:`\|G \boldsymbol{\gamma}\|_1` where

.. math::
  G\boldsymbol{\gamma} = [\boldsymbol{\gamma}_1 \quad \boldsymbol{\gamma}_2 \quad \ldots\quad \boldsymbol{\gamma}_{|\mathcal{T}|-1}]

i.e.

.. math::
  G = \left[\begin{array}{ccccc}
        1 &   & & & 0 \\
          & 1 & & & 0\\
          &   & \ddots & & \vdots \\
          &   &        & 1 & 0
      \end{array}
      \right]
      \quad
      \text{and}
      \quad 
  G^\top = \left[
    \begin{array}{cccc}
    1 &   & & \\
      & 1 & & \\
      &   & \ddots &  \\
      &   &        & 1 \\
    0 & 0 &\hdots & 0
    \end{array}
           \right].



This can be included in the model with the ``scipy.sparse.linalg.LinearOperator`` class
as follows::

  from scipy.sparse.linalg import LinearOperator
  import numpy as np

  def applyG(x):
    return x[:-1]

  def applyGtranspose(v):
    return np.concatenate((v,np.array([0])))

  (_,ngamma) = H.shape
  shape = (ngamma-1,ngamma)
  G = LinearOperator(shape,matvec=applyG,rmatvec = applyGtranspose)
  psObj.addRegularizer(regularizers.L1(scaling = mu*lam),linearOp=G)

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


Loss Processor Objects
=======================================
Projective splitting offers numerous choices as to how to process the various
operators making up a problem --- in the current setting, "operators"
corresponding to various elements in the summation in :eq:`masterProb` --- so
as to construct a separating hyperplane. In the original papers
:cite:`proj1,proj1n`, all operators were processed with some form of proximal
step, that is, essentially the calculation :eq:`proxDef` or some
approximation thereof.  Such calculations are also called `backward
steps`.   This feature persisted in later work such as :cite:`ACS14,CE18`.
More recently, however, new ways of processing operators have been
devised, based on *forward steps*, that is, simple gradient calculations
:cite:`for1`, :cite:`coco`.  These innovations
make projective splitting into a true first-order method.

``ProjSplitFit`` assumes that all regularizers employed have a computationally
efficient proximal operation.  It invokes the proximal operation of every
regularizer at every iteration.  For the loss function terms, however,
``projSplitFit`` affords a large number of options.  First, it permits the
loss function to be divided into an arbitrary number of blocks, each
containing the same number of observations (give or take one observation). You
may determine how many of these blocks to process at each iteration, and among
several rules to select blocks for processing.  Second, it provides eight
different options for processing each block.

The number of loss blocks and their activation scheme are controlled by
keyword arguments to the ``run`` method, as described in
:numref:`run-options` below. The procedure used to process each block is
determined by the optional ``process`` argument to the ``addData`` method.
This argument must be an object whose class is derived from
``lossProcessors.LossProcessor``. The file ``lossProcessors.py`` pre-defines
the following eight classes that may be used for this purpose :

* ``Forward2Fixed``: two-forward-step update with fixed stepsize, see :cite:`for1`
* ``Forward2Backtrack``: two-forward-step update with backtracking stepsize,
  see :cite:`for1`. This is the default loss processor if the ``process``
  argument is ommitted from ``addData``
* ``Forward2Affine``:  a specialized two-forward-step update for quadratic
  loss functions, automatically selecting a valid stepsize without
  backtracking, see :cite:`for1`. Only available when ``loss=2``
* ``Forward1Fixed``: one-forward-step update with fixed stepsize, see :cite:`coco`
* ``Forward1Backtrack``: one-forward-step update with backtracking stepsize,
  see :cite:`coco`
* ``BackwardExact``: Exact proximal/backward step for :math:`\ell_2^2` loss via matrix factoring.    Only available with ``loss=2``
* ``BackwardCG``:  approximate proximal/backward step computed by a conjugate gradient method, only available when ``loss=2``
* ``BackwardLBFGS``: approximate backward/proximal step computed by a
  limited-memory Broyden-Fletcher-Goldfarb-Shanno (LBFGS) solver.

To select a loss processor, you call the constructor of the desired class with any desired parameters,
and then pass the resulting
object into ``addData`` as the ``process`` argument. For example, to use
``BackwardLBFGS`` with its default parameters on the :math:`\ell_{1.5}^{1.5}`
loss, you would use the code fragment ::

  import lossProcessors as lp
  processObj = lp.BackwardLBFGS()
  projSplit.addData(A,y, loss=1.5, process=processObj)

See the detailed documentation section below for a complete listing of the
parameters for each loss processing class.

..  It is possible to create your own loss processing classes. They must derive
    from ``lossProcessors.LossProcessor`` and must implement the ``initialize``
    and ``update`` methods. Of course,
    convergence cannot be guaranteed unless you are aware of mathematical theory
    establishing the correctness of your procedure.

It is possible to create your own loss processing classes, although
guaranteeing convergence may requires significant mathematical analysis.
Please contact the authors for more information on extending ``projSplitFit``
in this manner.


.. _run-options:

Blocks of Observations
=========================

The ``run`` method of class ``ProjSplitFit`` has three important options which control the division of
the loss function into blocks, and how these blocks are processed at each
iteration. The first is ``nblocks``. This controls how many blocks projective
splitting breaks the loss into for processing. Recall the loss is

.. math::
  \frac{1}{n}\sum_{i=1}^n \ell (z_0 + a_i^\top H z,y_i)

An important property of projective splitting is *block iterativeness*:  the method does not
need to process every observation at each iteration. Instead, it may break the
:math:`n` observations into ``nblocks`` blocks and process as few as one block at a
time. ``nblocks`` may be any integer ranging from ``1``, meaning all observations are
processed at each iteration, up to ``n``, meaning every individual observation is
treated as a block. ``nblocks`` currently defaults to 1, but better
performance is often observed for larger values.

At present, blocks may only be contiguous spans of observation indices.
Suppose that ``nblocks`` is set to some value :math:`b`.  If :math:`n` is divisible by
:math:`b`, then each block simply contains :math:`n/b` contiguous indices.  If
:math:`b` does not divide the number of observations, then the first
:math:`n\!\!\mod b` blocks have :math:`\lceil n / b \rceil` observations and
the remaining blocks have :math:`\lfloor n / b \rfloor` observations.

The number of blocks processed per iteration is controlled via the argument
``blocksPerIteration``, which defaults to 1.  It can take any integer value
between 1 and ``nblocks``.

There are three ways to choose *which* blocks are processed at each iteration.
The selection of blocks is controlled with the ``blockActivation`` argument, which may be set to

* ``'random'``: select blocks at random, with equal probabilities
* ``'cyclic'``: cycle through the blocks in a round-robin manner
* ``'greedy'`` (the default): use the "greedy" heuristic of :cite:`for1`, page 24
  to select blocks.  This heuristic estimates which blocks are most important
  to process to make progress toward the optimal solution.

For example, to use 10 blocks and evaluate one block
per iteration using a greedy selection scheme, one would run the optimization
by (assuming that ``projSplit`` is a ``projSplitFit`` object) ::

   projSplit.run(nBlocks=10, blockActivation='greedy', blocksPerIteration=1)

However, greedy activation and one block per iteration being the defaults,
the above could be shortened to ::

   projSplit.run(nBlocks=10)

For some problem classes, it has been empirically been observed that
processing one or two blocks per iteration, selected in this greedy manner,
yields similar convergence to processing the entire loss term, but with much
lower time required per iteration.


..  JE moved the section below because I think it makes more sense after we discuss blocks.

Embedding Regularizers
=======================

Projective splitting handles regularizers through their proximal operations
:eq:`proxDef`. Regularizers added to a ``ProjSplitFit`` object are processed
at every iteration.  Such regularizers cause ``projSplitFit`` to allocate
three internal vector variables whose dimension matches the regularizer
argument.

However, the "forward" loss processors also have the option to "embed" a
single regularizer into each loss block; please see :numref:`run-options`
above for a discussion of dividing the loss function into blocks.  Each time a
loss block is processed, the loss processor also performs a backward
(proximal) step on the embedded regularizer, and no additional working memory
needs to allocated to the regularizer.

The embedding feature is controlled by the ``embed`` keyword argument of the ``addData`` method.
To solve a standard lasso problem with this technique, using 10 loss blocks,
one would proceed as follows::

  import projSplitFit as ps
  from regularizers import L1
  lam1 = 0.1
  projSplit = ps.ProjSplitFit()
  regObj = L1(scaling=lam1)
  projSplit.addData(A, y, loss=2, intercept=False, normalize=False, embed=regObj)
  projSplit.run(nblocks=10)
  optimalVal = projSplit.getObjective()
  z = projSplit.getSolution()

Note that when a regularizer is embedded in the loss function, it should not
also be added to the problem with ``addRegularizer``.  But only one
regularizer can be embedded in the loss term; if further regularizers are
needed, then those should be introduced into the problem with ``addRegularizer``.
If the loss term also contains a linear operator, that linear operator applies
to both the loss term and regularizer.

The embedded regularizer and the loss processor
must use the same stepsize. If they are different, a warning is printed and the
stepsize for the regularizer is set to be the stepsize of the loss processor.
For backtracking loss processors which modify the stepsize as the algorithm runs,
the embedded regularizer's stepsize will be automatically set to the correct stepsize before
it's prox operator is applied.

The ``embed`` feature cannot be used with the backward loss processors nor with ``Forward2Affine``.

Other Important Features
========================================

The ``keepHistory`` and ``historyFreq`` arguments to ``run()`` allow you to
record the progress of the algorithm in terms of objective function values,
running time, primal and dual residuals, and hyperplane values. These may be
extracted later via the ``getHistory()`` method.  Set ``keepHistory=True`` to
record history information.  The ``historyFreq`` parameter controls how often
information is recorded: for example, setting ``historyFreq=1`` causes the
information to be recorded every iteration, while setting ``historyFreq=10``
causes it to be recorded once every ten iterations.

The ``getObjective()`` method of the ``ProjSplitFit`` class simply returns the
objective value at the current primal iterate.

If you use either the ``keepHistory`` feature or the ``getObjective`` function
in conjunction with a user-defined loss function, then that loss function must
have a ``value`` method.  Similarly, using either the ``keepHistory`` feature
or the ``getObjective`` function in conjunction with a user-defined
regularizer requires that the regularizer have ``value`` method.

After using ``run()``, the ``getSolution()`` method of the ``ProjSplitFit``
class returns the primal iterate :math:`z^k`. If its ``descale`` argument is
set to ``True``, then the scaling vector used to scale each column of the data
matrix is applied to the elements of :math:`z^k`, so that the returned vector
of coefficients is in the coordinate system of the original data. Thus, the
returned coefficient vector may be directly used to make predictions using
unnormalized data, such as new test data.  The ``descale`` option is not
available when the loss term is composed with a linear operator.

The ``ProjSplitFit`` method ``getScaling()`` returns the scaling vector used in normalization.
This scaling vector can then be applied to normalize new test data. For
example, to normalize a new test datapoint ``xtest``, one could write::

  scaling = projSplit.getScaling()
  x_test_normalized = xtest/scaling

If the model was formulated with an intercept term, then the intercept term is the
first element of the vector returned by ``getSolution``.
