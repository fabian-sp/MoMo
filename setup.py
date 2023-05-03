import setuptools
import momo

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()


# Package meta-data.
NAME = 'momo-opt'
DESCRIPTION = 'MoMo: Momentum models for Adaptive Learning Rates'
URL = 'https://github.com/fabian-sp/MoMo'
EMAIL = 'fabian.schaipp@gmail.com'
AUTHOR = 'Fabian Schaipp'
REQUIRES_PYTHON = '>=3.8.0'
VERSION = momo.__version__

REQUIRED = [
    "torch",
    "numpy",
]

EXTRAS = {}


setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    packages=setuptools.find_packages(),
    install_requires=REQUIRED,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=REQUIRES_PYTHON,
)