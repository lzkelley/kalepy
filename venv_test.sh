python3.5 -m venv venv/
source venv/bin/activate
pip install -r requirements.txt
bash tester.sh
# nosetests --with-coverage --cover-inclusive --with-doctest --cover-package=kalepy /Users/lzkelley/Programs/kalepy/kalepy/tests/test_notebooks/test_utils.py
