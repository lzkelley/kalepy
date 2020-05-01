"""Generate a README.md file
"""

import os
import sys
import shutil
import re
import git
import subprocess
import logging


NOTEBOOK_FNAME = "./notebooks/demo.ipynb"
NB_SUFFIX = "ipynb"
# Path for output resources (i.e. image files)
OUTPUT_PATH_RESOURCES = "./docs/media/"
# Path for output markdown file
OUTPUT_FNAME = "./README.md"
README_BASE = "./_README.md"

# Only the output of the notebook *after* this line, is written to output
NB_START_HEADER = "# demo"

CONVERT_IMAGE_REGEX = r"^\!\[png\]\((.*)\)"
GITHUB_RAW_ADDR = "https://raw.githubusercontent.com/lzkelley/kalepy/{branch:}/docs/media/"

ALLOWED_BRANCHES = ['dev', 'master']

EXECUTE = False

# repo.git.describe


def main():
    global EXECUTE

    if '-v' in sys.argv:
        idx = sys.argv.index('-v')
        level = 20
        if len(sys.argv) > idx:
            _lev = sys.argv[idx+1]
            if not _lev.startswith('-'):
                level = int(_lev)

        logging.getLogger().setLevel(level)

    if '-x' in sys.argv:
        EXECUTE = True
    elif '-nx' in sys.argv:
        EXECUTE = False

    logging.debug("\t`EXECUTE` = {}".format(EXECUTE))
    logging.debug("sys.argv = '{}'".format(sys.argv))

    if not os.path.isfile(README_BASE):
        raise ValueError("Base/template readme file '{}' does not exist!".format(README_BASE))

    repo = git.Repo('.')
    if repo.bare:
        raise RuntimeError("Failed to initialize `git.Repo`!")

    try:
        branch = [bb for bb in repo.git.branch().split('\n') if bb.strip().startswith('* ')]
        if len(branch) != 1:
            raise
        branch = branch[0].strip('*').strip()
    except:
        logging.error("Failed to get current git branch!")
        raise

    logging.info("Current branch: '{}'".format(branch))
    if branch not in ALLOWED_BRANCHES:
        raise ValueError("Current branch '{}' not expected!".format(branch))

    nb_fname = os.path.abspath(NOTEBOOK_FNAME)
    exists = os.path.isfile(nb_fname)
    logging.info("Input notebook: '{}' exists: {}".format(NOTEBOOK_FNAME, exists))
    if not exists:
        raise ValueError("Notebook '{}' does not exist!".format(nb_fname))

    nb_base = nb_fname.split("." + NB_SUFFIX)
    if len(nb_base) != 2:
        raise ValueError("Could not split filename on '{}'".format(NB_SUFFIX))

    out_fil_temp = nb_base[0] + ".md"
    out_dir_temp = nb_base[0] + "_files"
    out_fil = os.path.abspath(OUTPUT_FNAME)
    out_dir = os.path.basename(out_dir_temp)
    # if `out_dir_temp` ends with a '/' then basename is empty string
    if len(out_dir) == 0:
        raise ValueError("Something wrong with output path!")
    out_dir = os.path.join(os.path.abspath(OUTPUT_PATH_RESOURCES), out_dir, '')
    check_names = [out_fil_temp, out_fil, out_dir_temp, out_dir]
    check_types = [False, False, True, True]
    for dpath, isdir in zip(check_names, check_types):
        if os.path.exists(dpath):
            logging.info("Removing previous output '{}'".format(dpath))
            if isdir:
                shutil.rmtree(dpath)
            else:
                os.remove(dpath)

            if os.path.exists(dpath):
                raise ValueError("Output still exists '{}'!".format(dpath))

    args = ['jupyter', 'nbconvert', nb_fname]
    if EXECUTE:
        args.append('--execute')
    args = args + ['--to', 'markdown']
    logging.debug("Running args:\n" + " ".join(args))

    # `capture_output` only works in python3.7; but should be the same as passing to PIPE
    subprocess.run(args, timeout=500, check=True,
                   stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    names_in = [out_fil_temp, out_dir_temp]
    names_out = [out_fil, out_dir]
    types = [False, True]
    for path_in, path_out, isdir in zip(names_in, names_out, types):
        if isdir and not os.path.isdir(path_in):
            raise ValueError("Failed to create output directory '{}'!".format(path_in))
        if not isdir and not os.path.isfile(path_in):
            raise ValueError("Failed to create output file '{}'!".format(path_in))
        if os.path.exists(path_out):
            raise ValueError("Move destination already exists!  '{}'".format(path_out))

        msg = "Moving '{}' ==> '{}'".format(path_in, path_out)
        logging.debug(msg)
        shutil.move(path_in, path_out)
        if not os.path.exists(path_out):
            raise ValueError("Failed to move file {}".format(msg))

    # Make sure image files in `out_dir` are committed to the git repo
    for fil in sorted(os.listdir(out_dir)):
        if os.path.basename(fil) not in repo.head.commit.tree:
            logging.info("File '{}' is not in git repo".format(fil))
            repo.index.add([os.path.join(out_dir, fil)])

    # Combine `_README.md` template with md-converted demo Notebook
    out_comps = os.path.split(out_fil)
    temp = os.path.join(out_comps[0], '_temp.md')
    logging.debug("Moving output to temporary file: '{}'".format(temp))
    shutil.move(out_fil, temp)
    github_addr = os.path.join(GITHUB_RAW_ADDR.format(branch=branch), '')
    logging.info("Replacing paths with '{}'".format(github_addr))
    try:
        with open(out_fil, 'w') as stout:

            # Add template to beginning
            with open(README_BASE, 'r') as stin:
                stout.write(stin.read() + "\n")

            # Append demo notebook content
            with open(temp, 'r') as stin:
                start = True
                for line in stin.readlines():
                    if start:
                        if line.strip().lower() == NB_START_HEADER.strip().lower():
                            start = False
                        continue

                    match = re.match(CONVERT_IMAGE_REGEX, line)
                    if (match is not None):
                        pattern = match.groups()
                        if len(pattern) != 1:
                            raise ValueError("Unexpected matched line: '{}'!".format(line))

                        pattern = pattern[0]
                        replace = github_addr + pattern
                        line = re.sub(pattern, replace, line)

                    stout.write(line)

    except Exception as err:
        logging.error("Failed on final write to '{}'".format(out_fil))
        raise err

    finally:
        logging.debug("Removing temporary file '{}'".format(temp))
        os.remove(temp)

    return


if __name__ == "__main__":
    main()
