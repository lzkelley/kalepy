rm -r venv/
# python3.5 -m venv venv/
python3.7 -m venv venv/
source venv/bin/activate
pip install -r requirements.txt
# bash tester.sh
python convert_notebook_tests.py
pytest
