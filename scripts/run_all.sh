# ---------------------------------------------------------------
# THIS SHOULD BE RUN FROM THE PROJECT ROOT DIRECTORY, i.e. ./scripts/run_all.sh
#
# ---------------------------------------------------------------

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command failed with exit code $?."' EXIT


printf $"\n\n=====================  RUNNING TEST SUITE  ========================\n\n"
# zsh scripts/tester.sh

printf $"\n\n================  RUNNING NOTEBOOK CONVERSIONS  ===================\n\n"

python scripts/gen_kde_api.py
python scripts/gen_plot_api.py
python scripts/gen_readme.py -v 0

printf $"\n\n====================  BUILDING SPHINX DOCS  =======================\n\n"
cd docs/
./docs.sh
cd ..

printf $"\n\n=======================  KALEPY DONE  =============================\n\n"

exit
