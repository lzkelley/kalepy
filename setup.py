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
    keywords="",
    url="https://github.com/lzkelley/kalepy/",
    packages=['kalepy'],
    include_package_data=True,
    install_requires=requirements,
    long_description=long_description,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
    ],
)
