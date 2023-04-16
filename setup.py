# Imports:
from setuptools import setup, find_packages

# Setup:
setup(
    name='atmospheric_gammas',
    version="dev",
    url='https://github.com/ckarwin/atmospheric_gammas.git',
    author='Chris Karwin',
    author_email='christopher.m.karwin@nasa.gov',
    packages=find_packages(),
    description = "Calculates atmospheric response and background for \
            MeV gamma rays.",
)
