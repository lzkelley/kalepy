"""Kernel Density Estimation (KDE) and sampling.
"""

from setuptools import setup, find_packages

short_description = __doc__.strip()

with open('requirements.txt') as inn:
    requirements = inn.read().splitlines()

with open("README.md", "r") as inn:
    long_description = inn.read().strip()

with open('kalepy/VERSION.txt') as inn:
    version = inn.read().strip()

setup(
    name="kalepy",
    author="Luke Zoltan Kelley",
    author_email="lzkelley@northwestern.edu",
    url="https://github.com/lzkelley/kalepy/",
    version=version,
    description=short_description,
    download_url="https://github.com/lzkelley/kalepy/archive/v{}.tar.gz".format(version),
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type='text/markdown',
    # Python version restrictions
    python_requires=">=3.7",

    keywords=['utilities', 'physics', 'astronomy', 'cosmology',
              'astrophysics', 'statistics',
              'kernel density estimation', 'kernel density estimate'],
)
