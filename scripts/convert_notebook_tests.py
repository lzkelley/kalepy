#!/usr/bin/env python
"""
"""

import os
import sys
import shutil
import glob
import subprocess
import logging

TEST_NOTEBOOK_NAMES = ["demo", "kde", "kernels", "plotting", "utils", "sampling"]

REPLACEMENTS = [
    ["plt.show()", "plt.close('all')"]
]

# Path of the holodeck package (top-level) directory
PATH = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.realpath(os.path.join(PATH, os.path.pardir))
# PATH = os.path.realpath(PATH)
# Path in the package in which the notebooks are stored
PATH_NOTEBOOKS = os.path.join(PATH, 'notebooks')
# Path in the package in which tests are stored
PATH_TESTS = os.path.join(PATH, 'kalepy', 'tests')

NOTEBOOK_SUFFIX = ".ipynb"
PYTHON_SUFFIC = ".py"
TEST_PREFIX = "test_notebook_"   # should start with 'test' and should include additional to avoid clashes

# This is the temporary directory to place converted notebooks for testing
PATH_NOTEBOOK_TESTS = os.path.join(PATH_TESTS, 'converted_notebooks')

PREPEND = """
def get_ipython():
    return type('Dummy', (object,), dict(run_line_magic=lambda *args, **kwargs: None))
"""


def main():
    if '-v' in sys.argv:
        logging.getLogger().setLevel(0)

    args = sys.argv[1:]
    notebook_names = []
    if len(args) > 0:
        notebook_names = [aa.split('.')[0] for aa in args if not aa.startswith('-')]

    if len(notebook_names) == 0:
        notebook_names = [nb for nb in TEST_NOTEBOOK_NAMES]

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

    convert_notebooks(notebook_names)

    return


def convert_notebooks(notebook_names):
    path_input = PATH_NOTEBOOKS
    path_output = PATH_NOTEBOOK_TESTS
    # names = TEST_NOTEBOOK_NAMES

    for nn in notebook_names:
        logging.warning("Converting '{}'".format(nn))
        src = os.path.join(path_input, nn)
        src_nb = src + NOTEBOOK_SUFFIX
        src_py = src + PYTHON_SUFFIC

        # convert from name `nn` to basename only in case it includes a notebooks subdirectory
        # e.g. "notebooks/sam/quickstart" ==> "tests/quickstart" instead of "tests/sam/quickstart"
        dst_py = TEST_PREFIX + os.path.basename(nn) + PYTHON_SUFFIC
        dst_py = os.path.join(path_output, dst_py)

        logging.info("'{}'".format(src_nb))
        logging.info("\t'{}'".format(src_py))
        logging.info("\t'{}'".format(dst_py))

        args = ['jupyter', 'nbconvert', '--ClearOutputPreprocessor.enabled=True', '--to', 'python', src_nb]
        logging.info(str(args))
        # `capture_output` only works in python3.7; but should be the same as passing to PIPE
        # subprocess.run(args, timeout=500, check=True, capture_output=True)
        try:
            subprocess.run(args, timeout=500, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        except Exception as err:
            logging.error(f"ERROR: failed running `{' '.join(args)}`")
            logging.error(f"ERROR: {err}")
            raise

        shutil.move(src_py, dst_py)

        import re
        pat = r'# In\[([\d\s]*)\]:'
        pat = r'# In\[([\d\s]*)\]:'
        pat = re.compile(pat)
        num = 0

        with open(dst_py, 'r') as original:
            data = original.readlines()

        last_blank = False
        with open(dst_py, 'w') as modified:
            first = True
            for ii, line in enumerate(data):
                if len(line.strip()) == 0:
                    if last_blank:
                        continue
                    last_blank = True
                else:
                    last_blank = False

                for old, new in REPLACEMENTS:
                    line = line.replace(old, new)

                match = re.match(pat, line)
                if (match is not None):
                    if first:
                        modified.write("# `convert_notebook_tests.py` prepended:\n" + PREPEND.strip() + "\n\n")
                        first = False

                    if num > 0:
                        evil = "    globals().update(locals())\n\n\n"
                        modified.write(evil)

                    prep = f"def test_cell_{num}():\n    print(f'cell: {num}')"
                    modified.write(prep)
                    num += 1
                elif first and (line.startswith('#') or len(line.strip()) == 0):
                    modified.write(line)
                    continue

                modified.write("    " + line)

    return


if __name__ == "__main__":
    main()
