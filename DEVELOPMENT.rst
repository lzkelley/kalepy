Development
===========

Change-Log
----------

Updates and changes to the newest version of `kalepy` will not always be backwards compatible.  The package is consistently versioned, however, to ensure that functionality and compatibility can be maintained for dependencies.  Please consult the `change-log <https://github.com/lzkelley/kalepy/blob/master/CHANGES.md>`_ for summaries of recent changes.


Test Suite
----------

If you are making, or considering making, changes to the `kalepy` source code, the are a large number of built in continuous-integration tests, both in the `kalepy/tests <https://github.com/lzkelley/kalepy/tree/master/kalepy/tests>`_ directory, and in the `kalepy notebooks <https://github.com/lzkelley/kalepy/tree/master/notebooks>`_.  Many of the notebooks are automatically converted into test scripts, and run during continuous integration.  If you are working on a local copy of `kalepy`, you can run the tests using the `tester.sh script <https://github.com/lzkelley/kalepy/tree/master/tester.sh>`_ (i.e. running `$ bash tester.sh`), which will include the notebook tests.


Deploying to pypi (pip)
-----------------------

.. code-block:: bash

  $ python setup.py sdist bdist_wheel
  $ twine check dist/<PACKAGE> 
  $ twine upload dist/<PACKAGE>
