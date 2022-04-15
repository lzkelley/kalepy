# jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600
#     --to notebook --execute notebooks/kde.ipynb

import os
import sys
import shutil
import glob
import subprocess
import logging

if '-v' in sys.argv:
    logging.getLogger().setLevel(0)

TEST_NOTEBOOK_NAMES = ["demo", "kde", "kernels", "plotting", "utils", "sampling"]

args = sys.argv[1:]
notebook_names = []
if len(args) > 0:
    notebook_names = [aa.split('.')[0] for aa in args if not aa.startswith('-')]

if len(notebook_names) == 0:
    notebook_names = [nb for nb in TEST_NOTEBOOK_NAMES]

# Path of this file and the overall package (top-level) directory
PATH = os.path.dirname(os.path.abspath(__file__))
# Path in the package in which the notebooks are stored
PATH_NOTEBOOKS = os.path.join(PATH, 'notebooks')
# Path in the package in which tests are stored
PATH_TESTS = os.path.join(PATH, 'kalepy', 'tests')

NOTEBOOK_SUFFIX = ".ipynb"
PYTHON_SUFFIC = ".py"
TEST_PREFIX = "test_"

# This is the temporary directory to place converted notebooks for testing
PATH_NOTEBOOK_TESTS = os.path.join(PATH_TESTS, 'test_notebooks')

PREPEND = """
    def get_ipython():
        return type('Dummy', (object,), dict(run_line_magic=lambda *args, **kwargs: None))
"""


def main():
    logging.info("Path: '{}'".format(PATH))
    logging.info("\tnotebooks: '{}'".format(PATH_NOTEBOOKS))
    logging.info("\ttests: '{}'".format(PATH_TESTS))

    nb_pattern = os.path.join(PATH_NOTEBOOKS, "*" + NOTEBOOK_SUFFIX)
    all_notebooks = sorted(glob.glob(nb_pattern))
    for nb in all_notebooks:
        base = os.path.basename(nb).replace(NOTEBOOK_SUFFIX, '')
        logging.debug("\t" + str(base))
        if base not in TEST_NOTEBOOK_NAMES:
            logging.warning("Found notebook '{}' not in test list".format(base))

    logging.info("`PATH_NOTEBOOKS_TESTS` = '{}'".format(PATH_NOTEBOOK_TESTS))
    if not os.path.exists(PATH_NOTEBOOK_TESTS):
        os.makedirs(PATH_NOTEBOOK_TESTS)

    # Remove old files
    for root, dirs, files in os.walk(PATH_NOTEBOOK_TESTS):
        for file in files:
            logging.info("removing '{}'".format(file))
            os.remove(os.path.join(root, file))

    convert_notebooks()

    return


def convert_notebooks():
    path_input = PATH_NOTEBOOKS
    path_output = PATH_NOTEBOOK_TESTS
    # names = TEST_NOTEBOOK_NAMES

    for nn in notebook_names:
        logging.warning("Converting '{}'".format(nn))
        src = os.path.join(path_input, nn)
        src_nb = src + NOTEBOOK_SUFFIX
        src_py = src + PYTHON_SUFFIC
        temp = TEST_PREFIX + nn + PYTHON_SUFFIC
        dst_py = os.path.join(path_output, temp)

        logging.info("'{}'".format(src_nb))
        logging.info("\t'{}'".format(src_py))
        logging.info("\t'{}'".format(dst_py))

        args = ['jupyter', 'nbconvert', '--to', 'python', src_nb]
        logging.info(str(args))
        # `capture_output` only works in python3.7; but should be the same as passing to PIPE
        # subprocess.run(args, timeout=500, check=True, capture_output=True)
        try:
            subprocess.run(args, timeout=500, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        except:
            logging.error(f"FAILED on command:  `{' '.join(args)}` !")
            raise

        shutil.move(src_py, dst_py)

        with open(dst_py, 'r') as original:
            data = original.read()
        with open(dst_py, 'w') as modified:
            modified.write(PREPEND.strip() + "\n# `convert_notebook_tests.py` appended\n" + data)

    return


if __name__ == "__main__":
    main()
