Development
===========

Please visit the `github page to make contributions to the package. <https://github.com/lzkelley/kalepy>`_  Particularly if you encounter any difficulties or bugs in the code, please `submit an issue <https://github.com/lzkelley/kalepy/issues>`_, which can also be used to ask questions about usage, or to submit general suggestions and feature requests.  Direct additions, fixes, or other contributions are very welcome which can be done by submitting `pull requests <https://github.com/lzkelley/kalepy/pulls>`_.  If you are considering making a contribution / pull-request, please open an issue first to make sure it won't clash with other changes in development or planned for the future.  Some known issues and indended future-updates are noted in the `change-log <https://github.com/lzkelley/kalepy/blob/master/CHANGES.md>`_ file.  If you are looking for ideas of where to contribute, this would be a good place to start.


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
