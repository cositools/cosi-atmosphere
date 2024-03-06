
Welcome to cosi-atmosphere's documentation!
===========================================

Introduction
------------
This package contains code for dealing with atmospheric effects in gamma-ray astronomy. Currently, this only includes atmospheric response, but future versions of the code will handle backgrounds, albedo, and reflection component.

Methodology
-----------
A mass model of Earth's atmosphere is created, using atmospheric data. 

Requirements
------------
The cosi atmosphere pipeline requires MEGAlib (available `here <https://megalibtoolkit.com/home.html>`_). 

Getting Help
------------
For issues with the code please open an issue in github. For further assistance, please email Chris Karwin at christopher.m.karwin@nasa.gov. 

.. warning::
   While many features are already available, fermi-stacking is still actively under development. Note that the current releases are not stable and various components can be modified or deprecated shortly.

Contributing
------------
This library is open source and anyone can contribute. If you have code you would like to contribute, please fork the repository and open a pull request. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   tutorials/index
   api/index
