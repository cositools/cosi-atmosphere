# Imports:
from setuptools import setup, find_packages

# Setup:
setup(
    name='cosi_atmosphere',
    version="0.0.1",
    url='https://github.com/cositools/cosi-atmosphere.git',
    author='COSI Team',
    author_email='christopher.m.karwin@nasa.gov',
    packages=find_packages(),
    install_requires = ['histpy==1.0.2','pandas','healpy','pymsis'],
    description = 'Tools for calculating atmospheric response and background'
)
