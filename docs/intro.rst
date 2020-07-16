##############
Introduction
##############

*ProjSplitFit* is a *Python* package for solving general linear data fitting problems
involving multiple regularizers and compositions with linear operators. The solver is
the *projective splitting* algorithm, which is a highly-flexible and scalable first-order solver.
This package implements most variants of projective splitting including
*backward steps* (proximal steps), *forward steps* (gradient steps), and *block-iterative operation*.
The implementation is based on *NumPy*.

The basic optimization problem that this code solves is the following:

.. math::
   \min_{z\in\mathbb{R}^d,z_0\in \mathbb{R}} \frac{1}{n}\sum_{i=1}^n \ell (z_0 + a_i^\top H z,y_i) + \sum_{j=1}^{n_r}h_j(G_j z)

where

* :math:`z_0\in\mathbb{R}` is the intercept variable
* :math:`z\in\mathbb{R}^d` is the parameter vector
* :math:`\ell:\mathbb{R}\times\mathbb{R}\to\mathbb{R}_+` is the loss
* :math:`y_i` for :math:`i=1,\ldots,n` are the labels
* :math:`H\in\mathbb{R}^{p \times d}` is a matrix (typically the identity)
* :math:`a_i\in\mathbb{R}^p` are the observations, forming the rows of the :math:`n\times p` observation/data matrix :math:`A`
* :math:`h_j` for :math:`j=1,\ldots,n_r` are convex functions which are *regularizers*, typically nonsmooth
* :math:`G_j` for :math:`j=1,\ldots,n_r` are matrices, typically the identity.

*ProjSplitFit* supports the following choices for the loss :math:`\ell`:

* :math:`\ell_p^p`, i.e. :math:`\ell(a,b)=\frac{1}{p}|a-b|^p` for any :math:`p\geq 1`
* logistic, i.e. :math:`\ell(a,b)=\log(1+\exp(-ab))`
* Any user-defined convex loss.

*ProjSplitFit* supports the following choices for the regularizers:

* The :math:`\ell_1` norm, i.e. :math:`\|x\|_1=\sum_i |x_i|`
* The :math:`\ell_2^2` norm, i.e. :math:`\|x\|_2^2`
* The :math:`\ell_2` norm i.e. :math:`\|x\|_2`
* Any user-defined convex regularizer.

The matrices :math:`H` and :math:`G_j` may be any linear operator. An easy way to define them is to use the
*scipy.sparse.linalg.LinearOperator* class, or simply 2D NumPy arrays.
