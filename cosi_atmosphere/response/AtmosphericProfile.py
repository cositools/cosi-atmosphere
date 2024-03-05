# Imports
from pymsis import msis
import numpy as np
import pandas as pd

class Atmosphere:

    def get_atm_profile(self,filename,dates,lons,lats,alts,version=2.1):

        """Get atmospher model from NRLMSIS: 
        Naval Research Laboratory's Mass Spectrometer Incoherent Scatter Radar model.
        
        For pymsis homepage see: https://swxtrec.github.io/pymsis/index.html

        Parameters
        ----------
        filename : str
            Name of output data file.
        dates : ArrayLike 
            Dates and time of interest.
        lons : ArrayLike: 
            Longitudes of interest in degrees.
        lats : ArrayLike 
            Latitudes of interest in degrees.
        alts : ArrayLike 
            Altitudes of interest in km.
        version : float, optional
            MSIS version number, one of (0,2.0,2.1). Default is 2.1.
        
        Note
        ----
        All inputs take arrays; however, for now use single 
        values for dates, lons, and lats. The only array should be alts, 
        since for this method we mainly want the altitude profile. 
        
        Returns
        -------
        atm_dict : dict
         Dictionary with atmospheric information. Keys are altitude[km],
         mass_density[kg/m3], N2[m-3], O2[m-3], O[m-3], He[m-3], H[m-3],
         Ar[m-3], N[m-3], anomalous_oxygen[m-3], NO[m-3], and Temperature[k].  
        """

        # Get atmospheric data:
        atm_output = msis.run(dates, lons, lats, alts, version=version)
        atm_output = np.squeeze(atm_output)

        # Construct dictionary:
        keys = ["mass_density[kg/m3]","N2[m-3]","O2[m-3]","O[m-3]","He[m-3]",\
            "H[m-3]","Ar[m-3]","N[m-3]","anomalous_oxygen[m-3]","NO[m-3]","Temperature[k]"]
       
        # Initialize dictionary with altitudes:
        atm_dict = {"altitude[km]":alts}
        
        # Add keys and data:
        for i in range(0,len(keys)):
            atm_dict[keys[i]] = atm_output[:,i]
            
            # Set nan to zero
            # Note: msis returns nan when there is no available data. 
            nan_index = np.isnan(atm_dict[keys[i]])
            atm_dict[keys[i]][nan_index] = 0

        # Write to file:
        df = pd.DataFrame(data=atm_dict)
        df.to_csv(filename,float_format='%10.6e',index=False,sep="\t",\
                columns=["altitude[km]","mass_density[kg/m3]","N2[m-3]",\
                "O2[m-3]","O[m-3]","He[m-3]","H[m-3]","Ar[m-3]",\
                "N[m-3]","anomalous_oxygen[m-3]","NO[m-3]","Temperature[k]"])
        
        return atm_dict
