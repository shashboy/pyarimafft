from io import open
from os import path

from setuptools import find_packages, setup

from pyarimafft.constants import VERSION

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def get_requirements(kind: str = None):
    if kind:
        filename = f"requirements-{kind}.txt"
    else:
        filename = "requirements.txt"
    with open(filename) as f:
        requires = (line.strip() for line in f)
        return [req for req in requires if req and not req.startswith("#")]


setup(
    name="pyarimafft",
    version=VERSION,
    description="A Time Series Forecasting library which performs outlier cleaning with LOESS regression, extracts multiple cyclicities with fast fourier transform & performs time series forecast via ARIMA.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shashboy/pyarimafft",
    author="Shashank Sharma",
    author_email="shashboy@gmail.com",
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    python_requires=">=3.8",
    install_requires=get_requirements(),
)
