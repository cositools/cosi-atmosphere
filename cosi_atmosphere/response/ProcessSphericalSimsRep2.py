# Imports:
import pandas as pd
from histpy import Histogram
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import sys,os
import matplotlib.colors as colors
from scipy.interpolate import interp1d
from scipy import integrate
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.convolution import convolve, Gaussian2DKernel
from  cosi_atmosphere.response import ProcessSphericalSims
import astropy
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
import matplotlib.colors as colors
from mhealpy import HealpixMap, HealpixBase
from numpy.linalg import norm
import h5py

class ProcessSphericalRep2(ProcessSphericalSims.ProcessSpherical):

    def bin_sim(self, name, elow=10, ehigh=10000, num_ebins=17, \
            anglow=0, anghigh=100, num_angbins=26,  dtheta_low=0,\
            dtheta_high=180, num_dtheta_bins = 45):

        """Bins the simulations, and writes output files 
        for both starting and measured photons. 
           
        Events are weigthed by the ratio of the cosine of incident 
        angle to cosine of measured angle. This is a geometric 
        correction factor to account for the effect of the projected area.

        Parameters
        ----------
        name : str
            Name of output response file (use .hdf5 or .h5 extension). 
        elow : float, optional
            Lower energy bound in keV (defualt is 100 keV). 
        ehigh : float, optional 
            Upper energy bound in keV (default is 10000 keV.
        num_ebins : int, optional 
            Number of energy bins to use (default is 17). Only log binning for now. 
        anglow : float, optional
            Lower bound of angle in degrees (default is 0).
        anghigh : float, optional
            Upper bound of anlge in degrees (default is 100)
        num_angbins : int, optional
            Number of angular bins (default is 26).
        dtheta_low : float, optional
            Lower bound of delta theta in degrees (default is 0).
        dtheta_high : float, optional
            Upper bound of delta theta in degrees (default is 180).
        num_dtheta_bins : int, optional
            Number of delta theta bins (default is 90).

        Note
        ----
        Response file contains 2 different histograms, each stored as 
        its own group. The group names are:
        1. starting_photons_rsp
        2. measured_photons
        """
        
        # Define energy bin edges: 
        self.energy_bin_edges = np.logspace(np.log10(elow), np.log10(ehigh), num_ebins)
         
        # Define incident angle bin edges (for initial and measured):
        # Using 4 deg resolution for now.
        self.incident_ang_bins = np.linspace(anglow,anghigh,num_angbins)
        print()
        print("Theta bins:")
        print(self.incident_ang_bins)

        # Define delta theta bins:
        self.dtheta_bins = np.linspace(dtheta_low,dtheta_high,num_dtheta_bins)
        print()
        print("Delta theta bins:")
        print(self.dtheta_bins)

        # Get index for photons that reach the watched volume:
        keep_index = np.isin(self.idi,self.idm)

        print("Number of simulated events: " + str(len(self.idi)))
        print("Number of events that reached watched volume: " + str(len(self.idm)))
    
        # Make histogram for starting photons response:        
        self.starting_photons_rsp = Histogram([self.energy_bin_edges, self.incident_ang_bins],\
                labels=["Ei [keV]", "theta_i [deg]"], sparse=True)
        self.starting_photons_rsp.fill(self.ei, self.incident_angle_i)
        savefile=name
        self.starting_photons_rsp.write(savefile, name ='starting_photons_rsp', overwrite=True)

        # Get second representation of response:
        self.primary_rsp2 = Histogram([self.energy_bin_edges, self.energy_bin_edges,\
                self.incident_ang_bins, self.incident_ang_bins, self.dtheta_bins],\
                labels=["Ei [keV]", "Em [keV]", "theta_i [deg]", "theta_m [deg]", "delta_theta [deg]"], sparse=True)
        self.primary_rsp2.fill(self.ei[keep_index], self.em, \
                self.incident_angle_i[keep_index], self.incident_angle_m, self.delta_theta,\
                weight=np.cos(np.deg2rad(self.incident_angle_i[keep_index]))/np.cos(np.deg2rad(self.incident_angle_m)))    
        self.primary_rsp2.write(savefile, name='primary_rsp_2', overwrite=True)
    
        return

    def get_binning_info(self):

        """Get info from histogram."""

        # Energy info:
        self.energy_bin_centers = self.primary_rsp2.axes["Ei [keV]"].centers
        self.energy_bin_widths = self.primary_rsp2.axes["Ei [keV]"].widths
        self.energy_bin_edges = self.primary_rsp2.axes["Ei [keV]"].edges

        # Get mean energy:
        emean_list = []
        for i in range(0,len(self.energy_bin_edges)-1):
            emean = np.sqrt(self.energy_bin_edges[i]*self.energy_bin_edges[i+1])
            emean_list.append(emean)
        self.emean = np.array(emean_list)
        
        # Incident angle bins:
        self.incident_ang_centers = self.primary_rsp2.axes["theta_i [deg]"].centers
        self.incident_ang_bins = self.primary_rsp2.axes["theta_i [deg]"].edges

       
        # Delta theta bins:
        self.dtheta_bins = self.primary_rsp2.axes["delta_theta [deg]"].edges
        self.dtheta_centers = self.primary_rsp2.axes["delta_theta [deg]"].centers
        return


    def load_response(self, name):

        """Load response file and corresponding normalization.

        Parameters
        ----------
        name, str
            Name of response file.
        """
    
        savefile = name 
        self.primary_rsp2 = Histogram.open(savefile, name='primary_rsp_2')
        self.starting_photons_rsp = Histogram.open(savefile, name='starting_photons_rsp')
        self.get_binning_info()

        return


    def make_scattering_plots(self, angle, energy, name, sig_y=0.1, \
            sig_x=1.5, scale=100, rsp_file=None, dist_fig_kwargs={},
            rainbow_fig_kwargs={}):

        """Visualize the response.
        
        Parameters
        ----------
        angle : float
            Off-axis angle of source in degrees. 
        energy : float 
            Energy band in keV. 
        name : str
            Prefix of saved plots (do not include extension). 
        sig_y : float, optional 
            Standard deviation of Gaussian kernal in y-direction 
            used for smoothing image (default is 0.1). 
        sig_x : float, optional
            Standard deviation of Gaussian kernal in y-direction
            used for smoothing image (default is 0.1).
        scale : float, optional
            Factor of array  max used for plotting contour.
            Default is 100. 
        rsp_file : str, optional
            Name of response file to use. 
        dist_fig_kwargs : dict, optional: 
            Pass any kwargs to plt.gca().set(), for distribution plot.
        rainbow_fig_kwargs : dict, optional:
            Pass any kwargs to plt.gca().set(), for rainbow plot.
        """

        # Make sure response is loaded:
        if rsp_file != None:
            self.load_response(rsp_file)
        try:
            self.incident_ang_bins
        except:
            print("ERROR: Need to load response file.")
            sys.exit()

        # Get bin info:
        ang_bin = self.get_theta_bin(angle)
        e_bin = self.get_energy_bin(energy)
        e_low = '{0:.2f}'.format(self.energy_bin_edges[e_bin]) 
        e_high = '{0:.2f}'.format(self.energy_bin_edges[e_bin+1])

        # Delta theta distribution plot:
        fig = plt.figure()
        ax = plt.gca()
        dist = self.primary_rsp2.slice[{"theta_i [deg]":ang_bin}].project(["delta_theta [deg]"]).contents.todense()
        plt.stairs(dist,edges=self.dtheta_bins)
        plt.title(r"$\theta_i = %s^\circ$" %str(angle), fontsize=14)
        plt.yscale("log")
        plt.xlabel(r"$\Delta \theta_\mathrm{s}$", fontsize=14)
        plt.ylabel("Counts",fontsize=14)
        ax.set(**dist_fig_kwargs)
        plt.savefig("%s_dtheta_dist.png" %str(name))
        plt.show()
        plt.close()

        # Rainbow plot:
        rainbow = np.array(self.primary_rsp2.slice[{"theta_i [deg]":ang_bin,"Em [keV]":e_bin}].project(["delta_theta [deg]","Ei [keV]"]).contents.todense())
       
        # Smooth image with Gaussian kernel:
        gauss_kernel = Gaussian2DKernel(sig_y,y_stddev=sig_x)
        filtered_arr = convolve(rainbow, gauss_kernel, boundary='extend')

        # Setup figure:
        fig = plt.figure(figsize=(7.5,6))
        ax = plt.gca()
        
        # Plot hist:
        img = ax.pcolormesh(self.dtheta_bins, self.energy_bin_edges, filtered_arr.T, 
                norm=colors.LogNorm(vmin=1),cmap="inferno")
        
        # Plot contours:
        first = np.amax(filtered_arr.T)/scale
        second = np.amax(filtered_arr.T)*2
        plt.contour(self.dtheta_centers, self.energy_bin_centers, 
                filtered_arr.T,levels = (first,second), colors='limegreen',linestyles=['--','--'], alpha=1,linewidths=2)
        
        ax.set_yscale('log')
        cbar = plt.colorbar(img,fraction=0.045)
        cbar.set_label("Counts", fontsize=14)
        plt.title(r"$\theta_i: %s^\circ$; Em: $%s - %s$ keV" %(str(angle),str(e_low),str(e_high)))
        plt.xlabel(r"$\Delta \theta_\mathrm{s}$", fontsize=14)
        plt.ylabel("Ei [keV]", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.set_facecolor("black")
        ax.set(**rainbow_fig_kwargs)
        plt.savefig("%s_rainbow.png" %str(name),dpi=600)
        plt.show()
        plt.close()
    
        return

    def get_dtheta_bin(self,dtheta,rsp_file=None):

        """Returns index of the delta theta bin.
        
        Parameters
        ----------
        dtheta : float
            delta theta  in degrees.
        rsp_file : str, optional
            Prefix of response file to load.

        Returns
        -------
        theta_bin : int
            Index of theta bin. 
        """
       
        # Make sure response is loaded:
        if rsp_file != None:
            self.load_response(rsp_file)
        try:
            self.incident_ang_bins
        except:
            print("ERROR: Need to specify response file.")
            sys.exit()

        # Find bin index:
        for i in range(0,len(self.dtheta_bins)):
            try:
                if (dtheta >= self.dtheta_bins[i]) & (dtheta < self.dtheta_bins[i+1]):
                    dtheta_index = i
                    break
            except: 
                print("ERROR: dtheta not found.")
                sys.exit()
    
        return dtheta_index

    def get_total_edisp_matrix(self, theta, dtheta_max, make_plots=True, rsp_file=None, tp_file=None):

        """Get the energy dispersion matrix. 
        
        The total energy dispersion is the sum of the transmitted 
        photons (which don't scatter) and the scattered photons. 
        Here I calculate all three: transmitted, scattered, and total. 
        Likewise, I calculate the transmission probability for all three. 

        Parameters
        ----------
        theta : float
            Incident angle of source in degrees. 
        dtheta_max : float
            Max delta theta to use in degrees. 
        make_plots : bool, optional 
            Show plots (default is True).
        rsp_file : str, optional 
            Name of response file to load. 
        tp_file : str, optional
            Option to overlay TP from analytical calculation in plot. 
            Specify file name from TPCalc class. Default is None.  
        """

        # Option to load response from file:
        if rsp_file != None:
            self.load_response(rsp_file)
    
        # Get theta bin:
        theta_bin = self.get_theta_bin(theta)

        # Get dtheta bin:
        dtheta_max_bin = self.get_dtheta_bin(dtheta_max)
        print()
        print("Using dtheta max [deg]: %s" %str(dtheta_max))
        print("Corresponding to bin: %s" %str(dtheta_max_bin))
        print

        # Normalize response:
        self.normalize_response()

        # Make transmitted edisp array:
        self.normed_edisp_array_beam, \
        self.tp_beam = self.make_edisp_matrix(theta_bin,dtheta_max_bin,comp="transmitted")

        # Save TP Beam to file:
        d = {"energy[keV]":self.emean,"TP":self.tp_beam}
        df = pd.DataFrame(data = d, columns = ["energy[keV]","TP"])
        df.to_csv("tp_beam.dat",sep="\t",index=False)

        # Make scattered edisp array:
        self.normed_edisp_array_scattered, \
        self.tp_scattered = self.make_edisp_matrix(theta_bin,dtheta_max_bin,comp="scattered")

        # Make total edisp array:
        self.normed_edisp_array_total, \
        self.tp_total = self.make_edisp_matrix(theta_bin,dtheta_max_bin,comp="total")
 
        if make_plots == True:

            # Plot edisp:
            print("plotting transmitted edisp matrix...")
            self.plot_edisp_matrix(self.normed_edisp_array_beam,\
                    "edisp_matrix_beam.png")
            print("plotting scattered edisp matrix...")
            self.plot_edisp_matrix(self.normed_edisp_array_scattered,\
                    "edisp_matrix_scattered.png")
            print("plotting total edisp matrix...")
            self.plot_edisp_matrix(self.normed_edisp_array_total,\
                    "edisp_matrix_total.png")
       
            # Plot transmission probability:
            print("plotting transmission probability...")
            self.plot_tp_from_edisp(tp_file=tp_file)

        return

    def make_edisp_matrix(self, theta_bin, dtheta_max, comp="total"):

        """Make energy dispersion matrix.
        
        Parameters
        ----------
        theta_bin : int 
            index of incident angle.
        dtheta_max : int
            Bin of max delta theta to use.
        comp : str, optional
            Component to use for constructing matrix. Either 
            transmitted, scattered, or total (default is total). 
        """
    
        # Project onto em, ei:
        if comp == "transmitted":
            self.edisp_array = self.primary_rsp2.slice[{"theta_i [deg]":theta_bin,
                "delta_theta [deg]":0}].project(["Em [keV]", "Ei [keV]"]).contents
            self.edisp_array = np.array(self.edisp_array) 

        if comp == "scattered":
            
            # Sum over all measured angles except source incident angle and 90 degrees:
            counter = 0
            for i in range(1,len(self.dtheta_centers)+1):
                if (i <= dtheta_max):
                    if counter == 0:
                        this_edisp_array = self.primary_rsp2.slice[{"theta_i [deg]":theta_bin, 
                            "delta_theta [deg]":i}].project(["Em [keV]", "Ei [keV]"]).contents
                        self.edisp_array = np.array(this_edisp_array) 
                    if counter != 0:
                        this_edisp_array = self.primary_rsp2.slice[{"theta_i [deg]":theta_bin, 
                            "delta_theta [deg]":i}].project(["Em [keV]", "Ei [keV]"]).contents
                        self.edisp_array += np.array(this_edisp_array) 
                    counter += 1
            
        if comp == "total":
            self.edisp_array = self.normed_edisp_array_beam + self.normed_edisp_array_scattered
        
        # Transmission probability:
        self.tp = self.edisp_array.sum(axis=0)
        
        return self.edisp_array, self.tp

    def normalize_response(self):

        """Normalizes the atmospheric response matrix by the total 
        photons thrown in E_i and theta_i. 
        """
     
        # Normalization factor:
        N = self.starting_photons_rsp.contents.todense()
        N = np.array(N)
        print()
        print("Total number of photons in response normalization:")
        print(np.sum(N))
        print()

        # Check statistics:
        bad_stats = N < 5
        Nbad = len(N[bad_stats]) 
        if Nbad != 0:
            print()
            print("WARNING: Number of normalization bins with counts < 5: %s" %Nbad)
            print("Make sure you have appropriate statistics!")
            print("Check normalization matrix: self.starting_photons_rsp.contents.todense()")
            print()

        # Set 0 bins to arbitrary small number to avoid division by 0:
        N[N==0] = 1e-12
        
        # Normalize:
        self.primary_rsp2 = self.primary_rsp2.todense() / N[:,None,:,None,None]
         
        return


