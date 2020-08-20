##############
Introduction
##############

``ProjSplitFit`` is a Python package for solving general linear data fitting problems
involving multiple regularizers and compositions with linear operators. The solver is
the *projective splitting* algorithm, a highly flexible and scalable first-order solver
framework.
This package implements most variants of projective splitting including
*backward steps* (proximal steps), various kinds of
*forward steps* (gradient steps), and *block-iterative operation*.
The implementation is based on ``numpy``.

The basic optimization problem that this code solves is the following:

.. math::
   \min_{z\in\mathbb{R}^d,z_0\in \mathbb{R}} \left\{ \frac{1}{n}\sum_{i=1}^n \ell (z_0 + a_i^\top H z,r_i) + \sum_{j=1}^{n_r} \nu_j h_j(G_j z) \right\}
   :label: masterProb

where

* :math:`z_0\in\mathbb{R}` is the intercept variable (which may be optionally fixed to zero)
* :math:`z\in\mathbb{R}^d` is the regression parameter vector
* :math:`\ell:\mathbb{R}\times\mathbb{R}\to\mathbb{R}_+` is the loss
* :math:`r_i` for :math:`i=1,\ldots,n` are the responses (or labels)
* :math:`H\in\mathbb{R}^{d' \times d}` is a matrix (typically the identity)
* :math:`a_i\in\mathbb{R}^{d'}` are the observations, forming the rows of the :math:`n\times d'` observation/data matrix :math:`A`
* :math:`h_j` for :math:`j=1,\ldots,n_r` are convex functions which are *regularizers*, typically nonsmooth
* :math:`G_j` for :math:`j=1,\ldots,n_r` are matrices, typically the identity.
* :math:`\nu_j` are positive scalar penalty parameters that multiply the regularizer functions.

The first summation in this formulation is the *loss*, measuring how well the
predictions :math:`z_0 + a_i^\top H z` obtained from the dataset using the
regression parameters :math:`(z_0,z)` match the observed responses
:math:`r_i`.  ``ProjSplitFit`` supports the following choices for the loss :math:`\ell`:

* :math:`\ell_p^p`, that is, :math:`\ell(a,b)=\frac{1}{p}|a-b|^p` for any :math:`p > 1`
* logistic, that is, :math:`\ell(a,b)=\log(1+\exp(-ab))`
* Any user-defined convex loss.

The second summation consists of regularizers that encourage specific
structural properties in the :math:`z` vector, most typically some form of
sparsity. ``ProjSplitFit`` supports the following choices for the
regularizers:

* The :math:`\ell_1` norm, that is, :math:`\|x\|_1=\sum_i |x_i|`
* The :math:`\ell_2^2` squared norm, that is, :math:`\|x\|_2^2`
* The :math:`\ell_2` norm that is, :math:`\|x\|_2`
* The group :math:`\ell_2` norm, that is :math:`\sum_{G\in\mathcal{G}}\|z_G\|_2` where :math:`\mathcal{G}` is a set of non-overlapping groups of indices
* Any user-defined convex regularizer.

The package does not impose any limits on the number of regularizers present
in a single problem formulation.

The linear transformations :math:`H` and :math:`G_j` may be any linear operators.
They may be passed to ``projSplitFit`` as 2D ``NumPy`` arrays, abstract linear opertors
as defined by the ``scipy.sparse.linalg.LinearOperator`` class, or sparse matrices
deriving from the ``scipy.sparse.spmatrix`` class. The data matrix :math:`A` may
be passed in as a 2D ``NumPy`` array or a sparse matrix
deriving from the ``scipy.sparse.spmatrix`` class.


Brief technical overview
==================================

The projective splitting algorithm is a primal-dual algorithm based on separating
hyperplanes.  A *dual solution* is a tuple of vectors :math:`\mathbf{w} = (w_1, \ldots,
w_d)` that certify the optimality of the "primal" vector :math:`z` for
:eq:`masterProb`.  At each iteration, the algorithm maintains an estimate
:math:`(z,\mathbf{w})` of primal and dual solutions.  Each iteration has two phases:
first, the algorithm "processes" some of the summation terms in the
problem formulation.  The results of the processing step allow the
algorithm to construct a hyperplane that separates the current primal-dual
solution estimate from the set of optimal primal-dual pairs.  The next
iterate is then obtained by projecting the current solution pair estimate
onto this hyperplane.

Within this overall framework, there are many alternatives for processing the
various summation terms in the formulation.  ``ProjSplitFit`` processes all
the regularizer terms at every iteration, using a standard proximal step (see
below for more information).  For the loss terms, however, it provides
considerable flexibility: the terms in the loss summation may be divided into
blocks, and only a subset of these blocks need be processed at each iteration
-- this mode of operation is called *block iterative*.
The subset of blocks processed in each iteration may be chosen at random, cyclically,
or using a greedy heuristic which selects those blocks most likely to yield
the best separating hyperplane.
Furthermore, there are
numerous options for processing each block, including approximate backward
(proximal) steps and various kinds of forward steps.

Projective splitting, generally, is an *operator splitting* method that is
defined for "monotone inclusion" problems.  This problem class includes all
convex optimization problems, but also other problems not representable as
convex optimization, and which do not have objective functions.  For this
reason, ``projSplitFit`` does not need to calculate the value of the objective
function in :eq:`masterProb` while solving the problem.  Instead, it monitors
how closely the current primal and dual solutions estimates come to certifying
their joint optimality.  However, if you call the ``getObjective`` method (see
below) or elect to keep a history of the solution trajectory, ``projSplitFit``
will attempt to compute objective function values.
