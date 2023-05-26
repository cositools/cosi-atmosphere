# Imports:
from atmospheric_gammas.response.AtmosphericProfile import Atmosphere
from atmospheric_gammas.response.MassModels import MakeMassModels
from atmospheric_gammas.response.RunSims import Simulate 
from atmospheric_gammas.response.ProcessSims import Process
import numpy as np

# Get Atmospheric model:
#instance = Atmosphere()
#date = np.array(['2016-06-13 12:00:00'], dtype="datetime64[h]")
#lat = -5.66
#lon = -107.38
#alts = np.linspace(0, 200, 2001) # km; spacing is 0.1 km (100 m) 
#atm_model = instance.get_atm_profile("rep_atm_model.dat",date,lon,lat,alts)

# Make mass model:
#instance = MakeMassModels("rep_atm_model.dat")
#instance.plot_atmosphere()
#instance.get_cart_vectors(angle, altitude)
#instance.rectangular_model(33)
#instance.spherical_model()

# Run sim:
#instance = Simulate()
#instance.run_sim("AtmospherePencilBeam.source", seed=3000)
#instance.parse_sim_file("Atmosphere_PencilBeam.inc1.id1.sim")

# Process sim:
#instance = Process("rectangular",0)
#instance.bin_sim()
#instance.make_scattering_plots()
#instance.calc_tp()
#instance.get_total_edisp_matrix()
#model_flux=instance.PL_interp(2)
#instance.ff_correction(model_flux)
