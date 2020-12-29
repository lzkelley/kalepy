
#
#
# ---------------------------------------------------------------

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT


echo "\n\n\n=====================  RUNNING TEST SUITE  ========================"
bash tester.sh

echo "\n\n\n================  RUNNING NOTEBOOK CONVERSIONS  ==================="
python gen_kde_api.py
python gen_plot_api.py
python gen_readme.py

echo "\n\n\n====================  BUILDING SPHINX DOCS  ======================="
cd docs/
bash docs.sh

echo "\n\n\n=======================  KALEPY DONE  ============================="
