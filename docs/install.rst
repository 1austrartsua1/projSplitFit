###############
Installation
###############

*ProjSplitFit* depends on NumPy and SciPy.

Linux terminal Install
=======================

Using Git, navigate to the directory of the desired location, type::

  $git clone https://github.com/1austrartsua1/projSplitFit.git

Make sure the project root directory is in the path of Python.

Pycharm Install
================
You should be able to use the VCS (Version Control System) in Pycharm.

Click VCS->enable VCS. Then click VCS->Clone and enter the URL https://github.com/1austrartsua1/projSplitFit.git

Running the Tests
==================
To run the tests, you need to have `Pytest <https://docs.pytest.org/en/stable/getting-started.html>`_ installed.
Navigate to the tests directory. From the command line, enter::

  $ pytest

This will run all the tests. Depending on your CPU, it may take 5-10min to run all the tests.

Specific tests can be run by::

  $ pytest test_file_name.py

For example::

  $ pytest test_multiple_norms.py

will just run the tests in ``test_multiple_norms.py``.

Most of the tests operate by running the algorithm on an optimization problem with a cached optimal value and checking that
*ProjSplitFit* reaches a desired accuracy. The optimal values are cached in the tests/results directory. You may refresh these
optimal values by creating new random optimization problems with randomly drawn data. In order to do this, you need to have
`cvxpy <https://www.cvxpy.org/install/>`_ installed.

Each test file has a boolean variable at the top called
``getNewOptVals``. If this is set to True, new optimization problems will be created and the optimal values cached.
