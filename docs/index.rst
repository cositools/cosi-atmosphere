
Welcome to cosi-atmosphere's documentation!
===========================================

Introduction
------------
The purpose of this library is to provide tools for dealing with atmospheric effects in gamma-ray astronomy. Currently, only the atmospheric response module is available, but future versions of the code will include modules for atmoshperic backgrounds, albedo emission, and reflection.

Methodology
-----------
A mass model of Earth's atmosphere is created, and the transport of gamma rays through the atmoshpere is simulated using MEGAlib. The atmosphere is characterized using the latest version (v2.1) of the Naval Research Laboratory's Mass Spectrometer Incoherent Scatter Radar Model (`NRLMSIS <https://swx-trec.com/msis>`_), implemented in the COSI atmosphere pipeline via the python interface, `pymsis <https://swxtrec.github.io/pymsis/>`_. NRLMSIS is an empirical model of Earth’s atmosphere that describes the average observed behavior of temperature and density, from the ground to an altitude of roughly 1000 km. More specifically, the model specifies the altitude profile of the number density for the primary species of the atmosphere (i.e. nitrogen, oxygen, argon, and helium). The simulation pipeline does not use any specific detector model, but rather makes use of so-called watched volumes, whereby photon properties are tracked for both an intital state, as well as at any other spatial location, as specified by the user. 

Requirements
------------
The cosi atmosphere pipeline requires MEGAlib, available `here <https://megalibtoolkit.com/home.html>`_.

Getting Help
------------
For problems with the code please open an issue in github. For further assistance, please email Chris Karwin at christopher.m.karwin@nasa.gov. 

Citing
---------
Please cite `Karwin+24 <https://arxiv.org/abs/2406.03534>`_ if you make use of this library in a publication. 

Contributing
------------
This library is open source and anyone can contribute. If you have code you would like to contribute, please fork the repository and open a pull request. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   tutorials/index
   api/index
   data_products
