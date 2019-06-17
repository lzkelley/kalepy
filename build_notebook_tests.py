# jupyter nbconvert --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=600 --to notebook --execute notebooks/kde.ipynb

import os
import shutil
import glob
import subprocess
import logging

logging.getLogger().setLevel(0)

TEST_NOTEBOOK_NAMES = ["demo", "kde", "kernels", "performance", "theory", "utils"]

PATH = os.path.dirname(os.path.abspath(__file__))
PATH_NOTEBOOKS = os.path.join(PATH, 'notebooks')
PATH_TESTS = os.path.join(PATH, 'kalepy', 'tests')
NOTEBOOK_SUFFIX = ".ipynb"

# This is the temporary directory to place converted notebooks for testing
PATH_NOTEBOOK_TESTS = os.path.join(PATH_TESTS, 'tests_nb')


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
        src = os.path.join(path_input, nn)
        src_nb = src + NOTEBOOK_SUFFIX
        src_py = src + '.py'
        dst_py = os.path.join(path_output, "test_" + nn + '.py')

        logging.info("'{}'".format(src_nb))
        logging.info("\t'{}'".format(src_py))
        logging.info("\t'{}'".format(dst_py))

        args = ['jupyter', 'nbconvert', '--to', 'python', src_nb]
        logging.info(str(args))
        subprocess.run(args, timeout=5000, check=True)
        shutil.move(src_py, dst_py)

    return


if __name__ == "__main__":
    main()
