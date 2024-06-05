Data Products
============

Response matrices
-----------------

Precomputed atmospheric response matrices are available for altitudes between 25 - 45 km, in 0.5 km steps. Each response file was generated using 10 million photons. Note that these response matrices are for the first representation, as discussed in Karwin+24, and they are integrated over all azimuth angles. The files are available on wasabi, and they can be downloaded by using the following command line prompt::
  
  AWS_ACCESS_KEY_ID=GBAL6XATQZNRV3GFH9Y4 AWS_SECRET_ACCESS_KEY=GToOczY5hGX3sketNO2fUwiq4DJoewzIgvTCHoOv aws s3api get-object  --bucket cosi-pipeline-public --key COSI_Atmosphere/Response/filename --endpoint-url=https://s3.us-west-1.wasabisys.com filename

The proper filename will be updated shortly. 

