# Imports:
from setuptools import setup, find_packages

# Setup:
setup(
    name='cosi_atmosphere',
    version="0.1.4a0",
    url='https://github.com/cositools/cosi-atmosphere.git',
    author='COSI Team',
    author_email='christopher.m.karwin@nasa.gov',
    packages=find_packages(),
    install_requires = ['histpy','pandas','healpy','pymsis'],
    include_package_data=True,
    package_data={'':['response/data_files/*']},
    description = 'Tools for calculating atmospheric response and background'
)
