# ProjSplitFit: Projective Splitting for Data Fitting Problems

*ProjSplitFit* is an implementation of the projective splitting algorithm suitable for convex data fitting problems such as lasso and logistic regression. It is highly flexible and may be applied to a wide variety of problems involving multiple regularizers and different types of loss functions, including user-defined losses and regularizers. It can handle regularizers which are composed with linear operators.

Projective splitting is a scalable first-order optimization solver. This package is implemented using *NumPy* and is suitable for large-scale problems.

## User Guide

Please read the [user guide](user_guide.pdf). This is a comprehensive and complete guide to how to use *ProjSplitFit*.

## Files

The most important file here is the [user guide](user_guide.pdf). Your first step should be to consult the user guide which has installation instructions, a tutorial, and the complete documentation of the package.

Here are the key modules related to the package

* [projSplitFit.py](projSplitFit.py), the main module including the key class *ProjSplitFit*
* [losses.py](losses.py) classes for defining the loss
* [lossProcessors.py](lossProcessors.py) classes for instructing *ProjSplitFit* how to process the loss
* [regularizers.py](regularizers.py) classes for adding regularizers to the model.

The following are helper modules used internally in *ProjSplitFit*

* [userInputVal.py](userInputVal.py) User input validation code
* [projSplitUtils.py](projSplitUtils.py) Miscellaneous utilities.

The following files are used to generate the documentation: *index.rst*, *conf.py*, *make.bat*, *MakeFile*, the *docs* directory and its contents.

Test files are in the [tests](tests) directory. You will need `pytest` installed to run the tests.

The [tests/results](tests/results) directory has cached optimal values used in the tests.
