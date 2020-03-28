python build_notebook_tests.py


# To fail tests that take longer than (e.g.) 1.0 seconds:
#   `--timer-warning 5.0 --timer-fail error`
# To avoid logging messages (particularly from `matplotlib` use:
#   `--nologcapture`
nosetests --with-coverage --cover-inclusive --with-doctest --cover-package=kalepy --nologcapture --with-timer --timer-top-n 10 kalepy/
