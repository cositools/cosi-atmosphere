This is an example of running parallel jobs with slurm, using a packable partition. We simulate a total of 10^7 photons using 10^3 parallel cores, and so on each core we simulate 10^4 photons.

This example also shows you how to get the distributions of all interactions that occured for the measured photons. The key difference is that we need to use "StoreSimulationInfo all" in the source file, in which case the simulation file contains information for each interaction. 
