"""Run performance test and save profiler output.

Usage:
    $ python performance NAME

    where `NAME` is the name of the test directory that must contain a `test.py` file, and `NAME.npz` file.

"""

import os
import sys
from datetime import datetime
# import cProfile  # noqa    import
import subprocess
import git
import pstats
from pstats import SortKey

import kalepy as kale


def main(in_name, out_name):

    args = ['python', '-m', 'cProfile', '-o', out_name, in_name]
    # `capture_output` only works in python3.7; but should be the same as passing to PIPE
    subprocess.run(args, timeout=500, check=True, capture_output=True)
    # subprocess.run(args, timeout=500, check=True,
    #               stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    # 'python -m cProfile [-o output_file] [-s sort_order] (-m module | myscript.py)'

    return


if __name__ == "__main__":
    now = datetime.now()
    print(f"{__file__} :: {now}")

    repo_path = os.path.dirname(kale.__file__)
    repo_path = os.path.abspath(os.path.join(repo_path, os.path.pardir))
    print(repo_path)
    repo = git.Repo(repo_path)
    if repo.bare:
        raise RuntimeError("Failed to initialize `git.Repo`!")

    branch = repo.head.ref.name
    commit = repo.head.ref.commit.hexsha
    git_key = f"{branch}#{commit[:7]}"
    print(f"{git_key=} || {branch=}, {commit=}")
    
    if (len(sys.argv) != 2) or ('-h' in sys.argv):
        print("")
        print(__doc__)
        print("")
        sys.exit(0)

    _name = sys.argv[1]
    name = f"pt-{_name}"
    print(f"running on '{name}'")
    path = os.path.abspath(name)
    fname = os.path.join(path, 'test.py')

    if not os.path.isfile(fname):
        raise FileNotFoundError(f"File '{fname}' does not exist!")

    now_str = now.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(path, "results")
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
        
    out_name = f"{now_str}_{git_key}__{name}"
    print(f"output: '{out_name}' @ '{out_path}'")
    out_name_prof = os.path.join(out_path, out_name + ".prof")
    out_name_txt = os.path.join(out_path, out_name + ".txt")

    main(fname, out_name_prof)

    if not os.path.isfile(out_name_prof):
        raise RuntimeError(f"Profiler output not found, profiling failed!  '{out_name_prof}'")

    stats = pstats.Stats(out_name_prof)
    stats.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(100)

    # print(f"profiler output: '{out_name_prof}'")
    with open(out_name_txt, 'w') as stream:
        stats = pstats.Stats(out_name_prof, stream=stream)
        stats.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(100)

    print(f"\nFinished after {datetime.now()-now}")
    # print(f"text output: '{out_name_txt}'")
    sys.exit(0)
