# Generate documentation from in-package docstrings
sphinx-apidoc -o ./source ../kalepy
make clean
make html
