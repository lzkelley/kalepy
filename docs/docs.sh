# Generate documentation from in-package docstrings
make clean
sphinx-apidoc -o ./source ../kalepy
make html
