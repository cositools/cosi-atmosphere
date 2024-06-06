Data Products
============

Atmospheric Response Matrices
-----------------------------

Precomputed atmospheric response matrices are available for altitudes between 25.5 - 45.5 km, in 1 km steps. Each response file was generated using 10 million photons. Note that these response matrices use the first representation, as discussed in Karwin+24, and they are integrated over all azimuth angles. The files are available on wasabi, and they can be downloaded by using the following command line prompt::
  
  AWS_ACCESS_KEY_ID=GBAL6XATQZNRV3GFH9Y4 AWS_SECRET_ACCESS_KEY=GToOczY5hGX3sketNO2fUwiq4DJoewzIgvTCHoOv aws s3api get-object  --bucket cosi-pipeline-public --key COSI_Atmosphere/Response/atm_response_[alt]p5km.hdf5 --endpoint-url=https://s3.us-west-1.wasabisys.com atm_response_[alt]p5km.hdf5

The general file name is atm_response_[alt]p5km.hdf5, where [alt] should be replaced by an integer value between 25 - 40. 

Change in the energy dispersion matrices and dection fraction with altitude:

.. image:: /images/Edispmatrix_total_alt_variation.gif
        :width: 25%
        :class: no-scaled-link

.. image:: /images/Edispmatrix_beam_alt_variation.gif
        :width: 25%
        :class: no-scaled-link

.. image:: /images/Edispmatrix_scattered_alt_variation.gif
        :width: 25%
        :class: no-scaled-link

.. image:: /images/TPprob_alt_variation.gif
        :width: 50%
        :align: center
        :class: no-scaled-link
