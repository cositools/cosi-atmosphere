# cosi-atmosphere

## Introduction  <br />
This package contains code for dealing with atmospheric effects in gamma-ray astronomy. Currently, this only includes atmospheric response, but future versions of the code will handle backgrounds, albedo emission, and the reflection component for GRBs. 

## Requirements <br />
This module requires the MEGAlib code, available [here](http://megalibtoolkit.com/home.html). Among other things, MEGAlib simulates the emission from any (MeV) gamma-ray source, simulates the instrument response, performs the event reconstruction, and performs the high-level data analysis. See the above link for more details regarding the MEGAlib package.   

## Getting Help
For issues with the code please open an issue in github. For further assistance, please email Chris Karwin at christopher.m.karwin@nasa.gov.

## Documentation
Documentation can be found here: https://cosi-atmosphere.readthedocs.io/en/latest/

## Installation
Using pip 
```
pip install cosi-atmosphere
```
From source (for developers)
```
git clone https://github.com/cositools/cosi-atmosphere.git
cd cosi_atmosphere
pip install -e .
```
