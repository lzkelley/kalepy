"""Generate the `plot_api.rst` file.
"""

import os
import sys
import shutil
import re
import git
import subprocess
import logging


NOTEBOOK_FNAME = "./notebooks/demo_kde.ipynb"
NB_SUFFIX = "ipynb"
# Path for output resources (i.e. image files)
OUTPUT_PATH_RESOURCES = "./docs/media/"
# Path for output markdown file
OUTPUT_FNAME = "./docs/source/kde_api.rst"

# Only the output of the notebook *after* this line, is written to output
NB_START_HEADER = "demo_kde"

# CONVERT_IMAGE_REGEX = r"^\!\[png\]\((.*)\)"    # MARKDOWN
CONVERT_IMAGE_REGEX = r"^\.\. image\:\: (.*)"   # RST
GITHUB_RAW_ADDR = "https://raw.githubusercontent.com/lzkelley/kalepy/{branch:}/docs/media/"

ALLOWED_BRANCHES = ['dev', 'master', 'test']

EXECUTE = True


def main():
    global EXECUTE

    if '-v' in sys.argv:
        idx = sys.argv.index('-v')
        level = 20
        if len(sys.argv) > idx+1:
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

    # Load repository information
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

    # Check that input notebook exists
    nb_fname = os.path.abspath(NOTEBOOK_FNAME)
    exists = os.path.isfile(nb_fname)
    logging.info("Input notebook: '{}' exists: {}".format(NOTEBOOK_FNAME, exists))
    if not exists:
        raise ValueError("Notebook '{}' does not exist!".format(nb_fname))

    nb_base = nb_fname.split("." + NB_SUFFIX)
    if len(nb_base) != 2:
        raise ValueError("Could not split filename on '{}'".format(NB_SUFFIX))

    out_fil_temp = nb_base[0] + ".rst"
    out_dir_temp = nb_base[0] + "_files"
    out_fil = os.path.abspath(OUTPUT_FNAME)
    out_dir = os.path.basename(out_dir_temp)
    # if `out_dir_temp` ends with a '/' then basename is empty string
    if len(out_dir) == 0:
        raise ValueError("Something wrong with output path!")
    out_dir = os.path.join(os.path.abspath(OUTPUT_PATH_RESOURCES), out_dir, '')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    print("out_dir = ", out_dir, os.path.isdir(out_dir))
    print("out_dir_temp = ", out_dir_temp, os.path.isdir(out_dir_temp))

    # Remove previously added image files (in `out_dir`) from the git repo before deleting
    for fil in os.listdir(out_dir):
        if not fil.endswith('.png'):
            raise ValueError("`out_dir` '{}' contains non-png files!".format(out_dir))
        _fil = os.path.join(out_dir, fil)
        logging.info("Removing '{}' from repo".format(_fil))
        repo.index.remove([_fil])  # , working_tree = True)

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

    args = ['jupyter', 'nbconvert', "--ExecutePreprocessor.kernel_name=python3", nb_fname]
    if EXECUTE:
        args.append('--execute')
    args = args + ['--to', 'rst']
    logging.debug("Running args:\n" + " ".join(args))

    # `capture_output` only works in python3.7; but should be the same as passing to PIPE
    try:
        subprocess.run(args, timeout=500, check=True,
                       stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    except:
        logging.error(f"FAILED TO RUN: {' '.join(args)}")
        raise

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
    temp = os.path.join(out_comps[0], '_temp.rst')
    logging.debug("Moving output to temporary file: '{}'".format(temp))
    shutil.move(out_fil, temp)
    github_addr = os.path.join(GITHUB_RAW_ADDR.format(branch=branch), '')
    logging.info("Replacing paths with '{}'".format(github_addr))
    img_matches = 0
    try:
        with open(out_fil, 'w') as stout:

            # Append demo notebook content
            with open(temp, 'r') as stin:
                start = 0
                # print("\n")
                for line in stin.readlines():
                    if start < 2:
                        # print(line)
                        if line.strip().lower() == NB_START_HEADER.strip().lower():
                            start = 1
                            logging.info("Found start line")
                        elif start == 1:
                            start = 2

                        continue

                    match = re.match(CONVERT_IMAGE_REGEX, line)
                    if (match is not None):
                        pattern = match.groups()
                        if len(pattern) != 1:
                            raise ValueError("Unexpected matched line: '{}'!".format(line))

                        pattern = pattern[0]
                        replace = github_addr + pattern
                        line = re.sub(pattern, replace, line)
                        img_matches += 1

                    stout.write(line)

                # print("\n")

        if start < 2:
            raise RuntimeError("Never found starting line matching '{}'!".format(NB_START_HEADER))

    except Exception as err:
        logging.error("Failed on final write to '{}'".format(out_fil))
        raise err

    finally:
        logging.debug("Removing temporary file '{}'".format(temp))
        os.remove(temp)

    # Update new README file in repo
    logging.info("Found {} image matches".format(img_matches))
    logging.info("Staging new file for git commit")
    repo.index.add([out_fil])

    return


if __name__ == "__main__":
    main()
