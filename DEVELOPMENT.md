# Development

## Deploying to pypi (pip)
  $ python setup.py sdist bdist_wheel
  $ twine check dist/<PACKAGE>
  $ twine upload dist/<PACKAGE>
