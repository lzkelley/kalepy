"""
"""

from setuptools import setup

with open('requirements.txt') as inn:
    requirements = inn.read().splitlines()

with open("README.md", "r") as inn:
    long_description = inn.read().strip()

with open('kalepy/VERSION.txt') as inn:
    version = inn.read().strip()

setup(
    name="kalepy",
    version=version,
    author="Luke Zoltan Kelley",
    author_email="lzkelley@northwestern.edu",
    description=("Kernel density estimation and sampling."),
    license="MIT",
    url="https://github.com/lzkelley/kalepy/",
    download_url="https://github.com/lzkelley/kalepy/archive/v{}.tar.gz".format(version),
    packages=['kalepy'],
    include_package_data=True,
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['utilities', 'physics', 'astronomy', 'cosmology',
              'astrophysics', 'statistics',
              'kernel density estimation', 'kernel density estimate'],
)
