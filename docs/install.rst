###############
Installation
###############

``ProjSplitFit`` depends on the standard ``numpy`` and ``scipy`` packages,
and has only been tested with Python 3.7.  It is not compatible with Python
2.7.

Installing from the Linux/Unix Command Line
=============================================

Using Git, navigate to the directory of the desired location, type::

  $ git clone https://github.com/1austrartsua1/projSplitFit.git

To use the ``projSplitFit`` module, make sure the project root directory is in
your  Python path (given by the ``PYTHONPATH`` environment variable on Unix
and Linux systems). Alternatively, run Python from the project root directory.

Installing Directly into Pycharm
==================================

If you wish to use ``projSplitFit`` from within PyCharm, you should be able to
use Pycharm's VCS (Version Control System) integration.

Click VCS->enable VCS. Then click VCS->Clone and enter the URL https://github.com/1austrartsua1/projSplitFit.git.

Running the Tests
==================
You may verify that ``projSplitFit`` is correctly installed and operating by
running its test suite, located in the ``tests`` subdirectory.  To run these
tests, you need to have the `pytest
<https://docs.pytest.org/en/stable/getting-started.html>`_ module installed
(in addition to ``numpy`` and ``scipy``). To initiate the tests from the
command line, descend into the ``tests`` subdirectory and enter::

  $ pytest

This command will run all the tests.  On systems in which the ``python``
command defaults to Python 2.7 and later versions of Python use the
``python3`` command, instead enter the command::

  $ python3 -m pytest

Depending on your CPU speed, it may take 5 to 10 minutes to run all the tests.

Specific tests can be run by specifying an individual test file.  For example::

  $ pytest test_multiple_norms.py

will only run the tests in the file ``test_multiple_norms.py``.  To accomplish
the same thing on systems defaulting to Python 2.7, you would instead enter::

  $ python3 -m pytest test_multiple_norms.py

To run tests from within PyCharm, issue  ``pytest`` commands as above within
PyCharm's Python Console tool pane, from the tests folder.

Most of the tests operate by running the algorithm on an optimization problem
and checking that ``projSplitFit`` finds the optimal value of this problem to
some desired accuracy.  The optimal values are stored in the ``tests/results``
subdirectory that is downloaded with the distribution.

If you wish, you may refresh these optimal values by creating new random
optimization problems with randomly drawn data.  Code at top of each test file
creates a boolean variable called ``getNewOptVals``, set to ``False``.   If
you change this assignment to ``True``, the tests will create new optimization
problems with randomly drawn data, and store their optimal values in the
``tests/results`` subdirectory.  In order to use this feature, however, you
must have the `cvxpy <https://www.cvxpy.org/install/>`_ package installed,
since the target optimal values are computed with ``cvxpy``.  Using this
feature will also slow down the testing process.
