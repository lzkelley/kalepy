# jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600 --to notebook --execute notebooks/kde.ipynb

import os
import sys
import shutil
import glob
import subprocess
import logging

if '-v' in sys.argv:
    logging.getLogger().setLevel(0)

TEST_NOTEBOOK_NAMES = ["demo", "kde", "kernels", "performance", "theory", "utils"]

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

    convert_notebooks()

    return


def convert_notebooks():
    path_input = PATH_NOTEBOOKS
    path_output = PATH_NOTEBOOK_TESTS
    names = TEST_NOTEBOOK_NAMES

    for nn in names:
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
        subprocess.run(args, timeout=500, check=True, capture_output=True)
        shutil.move(src_py, dst_py)

    return


if __name__ == "__main__":
    main()
