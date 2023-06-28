# Imports:
from setuptools import setup, find_packages

# Setup:
setup(
    name='cosi_atmosphere',
    version="dev",
    url='https://github.com/cositools/cosi-atmosphere.git',
    author='COSI Team',
    author_email='christopher.m.karwin@nasa.gov',
    packages=find_packages(),
    description = "Tools for calculating atmospheric response and background"
)
