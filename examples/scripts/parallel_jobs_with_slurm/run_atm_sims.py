# Imports:
from cosi_atmosphere.response.AtmosphericProfile import Atmosphere
from cosi_atmosphere.response.MassModels import MakeMassModels
from cosi_atmosphere.response.RunSims import Simulate 
from cosi_atmosphere.response.ProcessSphericalSims import ProcessSpherical
from cosi_atmosphere.response.ProcessSphericalSimsRep2 import ProcessSphericalRep2
from cosi_atmosphere.response.TPCalc import TPCalculator
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# First run the sims:
#instance = Simulate()
#instance.run_sim("Atmosphere_Isotropic.source")
#instance.parse_sim_file_all_info("Atmosphere_Isotropic.inc1.id1.sim",unique=True)

# After all parallel jobs finish, you can combine the files from the 
# top directory of the analysis. Based on the shell script in this example,
# it would look like this:

# Combine all events:
#file_list = ["sim_%s/" %str(i) + "all_thrown_events.dat" for i in range(0,1001)]
#instance.combine_sims(file_list,10000,"all_thrown_events_combined")

# Combine measured events:
#file_list = ["sim_%s/" %str(i) + "event_list.dat" for i in range(0,1001)]
#instance.combine_sims(file_list,10000,"event_list_combined")

# Now plot the distributions of the interactions:
#instance.plot_sequence("event_list_combined.dat.gz")
