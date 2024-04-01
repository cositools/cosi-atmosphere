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
from  cosi_atmosphere.response import ProcessSims
import astropy
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
import matplotlib.colors as colors

class Process:

    """Analyze atmophere simulations.
        
    Parameters
    ----------
    theta : float 
        Off-axis angle of source in degrees.
    all_events_file : str, optional 
        Event file with all thrown events (default is output 
        from ParseSims method). 
    measured_events_file : str, optional 
        Event file with all measured events (default is output 
        from ParseSims method). 
    """

    def __init__(self, theta, all_events_file="all_thrown_events.dat", \
            measured_events_file="event_list.dat"):
     
        # Get test directory:
        path_prefix = os.path.split(ProcessSims.__file__)[0]
        self.test_dir = os.path.join(path_prefix,"data_files")

        # Off-axis angle of source:
        self.theta = theta

        # Read in all thrown events:
        df = pd.read_csv(all_events_file, delim_whitespace=True)
        self.idi = df["id"] # event IDs
        self.ei = df["ei[keV]"] # starting energy
        self.xi = np.array(df["xi[cm]"]) # starting x
        self.yi = np.array(df["yi[cm]"]) # starting y
        self.zi = np.array(df["zi[cm]"]) # starting z
        self.xdi = np.array(df["xdi[cm]"]) # starting x direction
        self.ydi = np.array(df["ydi[cm]"]) # starting y direction
        self.zdi = np.array(df["zdi[cm]"]) # starting z direction
        
        # Read in all measured events:
        df = pd.read_csv(measured_events_file, delim_whitespace=True)
        self.idm = df["id"] # measured IDs (they correspond to initial IDs above)
        self.em = df["em[keV]"] # measured energy
        self.xm = np.array(df["xm[cm]"]) # measured x
        self.ym = np.array(df["ym[cm]"]) # measured y
        self.zm = df["zm[cm]"] # measured z
        self.xdm = df["xdm[cm]"] # measured x direction
        self.ydm = df["ydm[cm]"] # measured y direction
        self.zdm = np.array(df["zdm[cm]"]) # measured z direction

        # Calculate radial distance from zero:
        self.ri = np.sqrt(self.xi**2 + self.yi**2) # cm
        self.ri *= 1e-5 # km
        self.rm = np.sqrt(self.xm**2 + self.ym**2) # cm
        self.rm *= 1e-5 # km
        
        # Define vector array:
        v = np.array([self.xdm,self.ydm,self.zdm]).T
    
        # Spherical coordinates for measured position:
        self.sp_coords = astropy.coordinates.cartesian_to_spherical(self.xm,self.ym,self.zm)
        self.lat = self.sp_coords[1].deg # deg
        self.lon = self.sp_coords[2].deg # deg

        # Spherical coordinates for measured direction:
        self.sp_coords_dir = astropy.coordinates.cartesian_to_spherical(self.xdm,self.ydm,self.zdm)
        self.lat_dir = self.sp_coords_dir[1].deg  # deg
        self.lon_dir = self.sp_coords_dir[2].deg  # deg

        # Get incident angle:
        n = np.array([0,0,-1])
        self.incident_angle = self.angle(v,n)*(180/np.pi) # degrees

        return

    def dot_product(self, v1, v2):
        
        """Calculates dot product of two vectors, v1 and v2.
         
        Parameters
        ----------
        v1 : array
            First vector.
        v2 : array
            Second vector.

        Note
        ----
        Input vector arrays must have shape
        (rows, cols) = (N,3),
        where cols is x,y,z coordinates,
        and rows is number of vectors.

        Returns
        -------
        dp : float
            Dot product of v1 and v2. 
        """
      
        # For array with multiple vectors:
        if (v1.ndim == 2) | (v2.ndim == 2):
            dp = np.sum(v1*v2,axis=1)
        
        # For single vector array:
        if (v1.ndim == 1) & (v2.ndim == 1):
            dp = np.sum(v1*v2,axis=0)

        return dp
    
    def length(self, v):
        
        """Calculates length of vector.
        
        Parameters
        ----------
        v : array
            Input vector.

        Returns
        -------
        float
            Vector length. 
        """
        
        return np.sqrt(self.dot_product(v,v))
    
    def angle(self, v1, v2):
        
        """Calculates angle between two vectors, in radians.

        Parameters
        ----------
        v1 : array
            First vector. 
        v2 : array
            Second vector.
        
        Note
        ----
        Input vector arrays must have shape 
        (rows, cols) = (N,3),
        where cols is x,y,z coordinates,
        and rows is number of vectors. 
        
        Returns
        -------
        angle : float
            Angle in radians. 
        """

        arg = self.dot_product(v1,v2)/(self.length(v1)*self.length(v2))
        
        # Need to round to limited decimal places, 
        # otherwise, rounding errors may give arg>1, 
        # which is undefined in arccos. 
        arg = np.round(arg,4)
        angle = np.arccos(arg)
        
        return angle

    def bin_sim(self, elow=10, ehigh=10000, num_ebins=24,\
            rlow=1e-7, rhigh=1000, num_rbins=120, 
            starting_ph_output='starting_photons.hdf5',
            measured_ph_output='measured_photons.hdf5'):

        """Bins the simulations, and writes output files 
        for both starting and measured photons. 
           
        Events are weigthed by the ratio of the cosine of incident 
        angle to cosine of measured angle. This is a geometric 
        correction factor to account for the effect of the projected area.
        
        Parameters
        ---------
        elow : float, optional
            Lower energy bound in keV (defualt is 10 keV). 
        ehigh : float, optional 
            Upper energy bound in keV (default is 10000 keV).
        num_ebins : int, optional 
            Number of energy bins to use (default is 24). Only log 
            binning for now. 
        rlow : float, optional 
            Lower radial bound in km (default is 1e-7). 
        rhigh : float, optional
            Upper radial bound in km (default is 1000).
        num_rbins : int, optional
            Number of radial bins to use (default is 120).
        starting_ph_output : str, optional
            Name of output file for starting photons binned histogram 
            (default is 'starting_photons.hdf5'). 
        measured_ph_output : str, optional
            Name of output file for measured photons binned histogram
            (default is 'measured_photons.hdf5'). 
        """
        
        # Define energy bin edges:
        self.energy_bin_edges = np.logspace(np.log10(elow), np.log10(ehigh), num_ebins)
        
        # Define radial bin edges:
        self.radial_bins = np.logspace(np.log10(rlow), np.log10(rhigh), num_rbins)
        
        # Define incident angle bin edges:
        self.incident_ang_bins = np.linspace(0,100,40)
        
        # Define xi and yi bin edges:
        minx = np.amin(self.xi)
        maxx = np.amax(self.xi)
        self.xyi_bins = np.linspace(minx,maxx,100)
        
        # Define xm and ym bin edges:
        minx = 0.1 # can't use 0 for log spacing
        maxx = np.amax(self.xm)
        self.xym_bins = np.logspace(np.log10(minx),np.log10(maxx),100)
   
        # Define longitude bin edges:
        self.lat_bins = np.linspace(-90,90,45)

        # Define latitude bin edges:
        self.lon_bins = np.linspace(0,360,90)

        # Make histogram for starting photons:
        self.starting_photons = Histogram([self.energy_bin_edges, self.radial_bins, \
                self.xyi_bins, self.xyi_bins], \
                labels=["Ei [keV]", "ri [km]", "xi [cm]", "yi [cm]"])
        self.starting_photons.fill(self.ei, self.ri, self.xi, self.yi)
        self.starting_photons.write(starting_ph_output, overwrite=True)

        # Make un-weighted histogram,
        # used for comparison.
        self.measured_photons_baseline = Histogram([self.energy_bin_edges, self.radial_bins, \
                self.xym_bins, self.xym_bins, self.incident_ang_bins],\
                labels=["Em [keV]", "rm [km]", "xm [cm]", "ym [cm]", "theta_prime [deg]"])
        self.measured_photons_baseline.fill(self.em, self.rm, self.xm, self.ym, self.incident_angle)
        self.measured_photons_baseline.write(measured_ph_output, overwrite=True)
        
        # Make weighted histogram.
        self.measured_photons = Histogram([self.energy_bin_edges, self.radial_bins, \
                self.xym_bins, self.xym_bins, self.incident_ang_bins],\
                labels=["Em [keV]", "rm [km]", "xm [cm]", "ym [cm]", "theta_prime [deg]"])
        self.measured_photons.fill(self.em, self.rm, self.xm, self.ym, self.incident_angle, 
                weight=np.cos(np.deg2rad(self.theta))/np.cos(np.deg2rad(self.incident_angle)))
        self.measured_photons.write(measured_ph_output, overwrite=True)

        # Get binning info:
        self.get_binning_info()

        return

    def get_binning_info(self):

        """Get info from histogram."""

        self.energy_bin_centers = self.starting_photons.axes["Ei [keV]"].centers
        self.energy_bin_widths = self.starting_photons.axes["Ei [keV]"].widths
        self.radial_bin_centers = self.measured_photons.axes["rm [km]"].centers
        self.radial_bin_widths = self.measured_photons.axes["rm [km]"].widths

        # Get mean energy:
        emean_list = []
        for i in range(0,len(self.energy_bin_edges)-1):
            emean = np.sqrt(self.energy_bin_edges[i]*self.energy_bin_edges[i+1])
            emean_list.append(emean)
        self.emean = np.array(emean_list)

        # Projected histograms:
        self.ei_hist = self.starting_photons.project("Ei [keV]").contents
        self.ei_array = np.array(self.ei_hist)
        self.em_hist = self.measured_photons.project("Em [keV]").contents
        self.em_array = np.array(self.em_hist)
        self.rm_hist = self.measured_photons.project("rm [km]").contents
        self.rm_array = np.array(self.rm_hist)

        return

    def load_response(self, starting_ph_file="starting_photons.hdf5",\
            measured_ph_file="measured_photons.hdf5"):

        """Load response files for starting and measured photons.

        Parameters
        ----------
        starting_ph_file : str, optional 
            Binned histogram for staring photons (default is 'starting_photons.hdf5').
        measured_ph_file : str, optional 
            Binned histogram for measured photons (default is 'measured_photons.hdf5').
        
        Note
        ----
        Default photon files are the output from the bin_sim method.
        """
        
        self.starting_photons = Histogram.open(starting_ph_file)
        self.measured_photons = Histogram.open(measured_ph_file)

        return

    def make_scattering_plots(self, starting_pos=True, measured_pos=True, \
            spec_i=True, radial_dist=True, theta_prime = True, \
            theta_prime_em = True, rad_em = True, rad_ei = True, show_baseline = True):

        """Visualize the photon scattering.
        
        Parameters
        ----------
        starting_pos : bool, optional 
            Scatter plot showing starting postions of all photons. 
        measured_pos : bool, optional 
            Scatter plot showing measured postions of all photons.  
        spec_i : bool, optional 
            Starting and measured photon spectrum. 
        radial_dist : bool, optional 
            Radial distribution of measured photons. 
        theta_prime : bool, optional 
            Incident angle distribution.
        theta_prime_em : bool, optional
            Incident angle versus measured energy.
        rad_em : bool, optional
            Radial distribution versus measured energy.
        rad_ei : bool, optional
            Radial distribution versus initial energy.
        show_baseline : bool, optional
            Option to show angular comparison to un-weighted histogram. 
        """

        # Define condition for unscattered photons:
        theta_low = self.theta - 0.2
        theta_high = self.theta + 0.2
        condition = (self.incident_angle > theta_low) & (self.incident_angle < theta_high)

        # Starting position:
        if starting_pos == True:
            plt.scatter(self.xi,self.yi,color="darkorange")
            plt.xlabel("x [cm]")
            plt.ylabel("y [cm]")
            plt.savefig("dist_initial.png")
            plt.show()
            plt.close()

        # Measured position:
        if measured_pos == True:
            plt.scatter(self.xm, self.ym, color="cornflowerblue", label="measured (all)")
            plt.scatter(self.xi, self.yi, color="darkorange", label="starting")
            plt.scatter(self.xm[condition], self.ym[condition], color="black", label="measured (beam)")
            plt.xlabel("x [cm]")
            plt.ylabel("y [cm]")
            plt.savefig("dist_measured.png")
            plt.show()
            plt.close()

        # Starting photon spectrum (dN/dE):
        if spec_i == True:
            
            # Plot initial photons:
            yi = self.ei_array/self.energy_bin_widths
            yerr = np.sqrt(self.ei_array)/self.energy_bin_widths
            plt.loglog(self.emean, yi,ls="-", marker="o", color="black",label="Initial Photons")
            plt.errorbar(self.emean, yi, yerr=yerr, ls="-", marker="o", color="black")

            # Plot measured photon:
            ym = self.em_array/self.energy_bin_widths
            yerr = np.sqrt(self.em_array)/self.energy_bin_widths
            plt.loglog(self.emean, ym,ls="-", marker="s", color="cornflowerblue")
            plt.errorbar(self.emean, ym, yerr=yerr, ls="-", marker="s", color="cornflowerblue", label="Measured Photons")

            plt.ylim(np.amax(yi)*0.1,np.amax(yi)*10)
            plt.xlabel("Energy [keV]")
            plt.ylabel("dN/dE [ph/keV]")
            plt.grid(ls="--",color="grey",alpha=0.3)
            plt.legend(frameon=False)
            plt.savefig("ei_photon_dist.png")    
            plt.show()
            plt.close()

            # Print number of starting and measured photons as sanity check:
            print()
            print("Number of starting photons: " + str(self.ei_array.sum()))
            print("Number of measured photons: " + str(self.em_array.sum()))
            print()
        
        # Radial distribution of measured photons:
        if radial_dist == True:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            area = 2*np.pi*self.radial_bins[:-1]*self.radial_bin_widths
            total = np.sum(self.rm_array)
            ax1.loglog(self.radial_bin_centers, self.rm_array/(area), color="cornflowerblue")
            ax2.loglog(self.radial_bin_centers, self.rm_array, ls="--", color="darkorange", zorder=0)
            ax1.set_ylabel("counts/area", color="cornflowerblue")
            ax2.set_ylabel("counts", color="darkorange")
            ax1.set_xlabel("rm [km]")
            plt.grid(ls=":",color="grey",alpha=0.3)
            plt.savefig("rdist.png")    
            plt.show()
            plt.close()

        # Distribution of theta prime for measured photons:
        if theta_prime == True:
            if show_baseline == True:
                dist = self.measured_photons_baseline.project("theta_prime [deg]").contents
                plt.plot(self.incident_ang_bins[1:],dist,color="blue",label="true")
            dist = self.measured_photons.project("theta_prime [deg]").contents
            plt.plot(self.incident_ang_bins[1:],dist,color="red",label=r"scaled by cos($\theta_i$)/cos($\theta_m$)")
            plt.yscale("log")
            plt.ylabel("counts")
            plt.xlabel(r"$\theta$")
            plt.xlim(0,100)
            plt.grid(ls="--",color="grey",alpha=0.3)
            plt.legend(loc=1)
            plt.savefig("theta_prime_dist.png")    
            plt.show()
            plt.close()

        # Theta prime distribution versus measured energy:
        if theta_prime_em == True:
            self.measured_photons.project(["Em [keV]","theta_prime [deg]"]).plot(norm=mpl.colors.LogNorm())
            plt.xscale("log")
            plt.xlim(1e2,1e4)
            plt.ylim(0,100)
            plt.savefig("thetaprime_energy_m_dist.png")
            plt.show()
            plt.close()

        # Radial distribution versus measured energy:
        if rad_em == True:
            self.measured_photons.project(["Em [keV]","rm [km]"]).plot(norm=mpl.colors.LogNorm())
            plt.xscale("log")
            plt.yscale("log")
            plt.savefig("radial_energy_m_dist.png")
            plt.show()
            plt.close()

        # Radial distribution versus initial energy:
        if rad_ei == True:
            transmitted_index = np.isin(self.idi,self.idm)
            transmitted_events = Histogram([self.energy_bin_edges,self.radial_bins],labels=["ei [keV]","rm [km]"])
            transmitted_events.fill(self.ei[transmitted_index],self.rm)
            transmitted_events.project(["ei [keV]","rm [km]"]).plot(norm=mpl.colors.LogNorm())
            plt.xscale("log")
            plt.yscale("log")
            plt.savefig("radial_energy_i_dist.png")
            plt.show()
            plt.close()

        return

    def calc_tp(self, show_plot=True):

        """Calculate the transmission probability. 
        
        This gives the probability that a photon of a given energy 
        will pass the atmosphere without being scattered, and thus cross 
        the detecting volume. The transmitted photons do not
        scatter, and thus there is no corresponding energy 
        dispersion. However, the transmission probability does not 
        account for scattered photons which enter the detecting volume 
        from off-axis angles. These photons will produce a significant 
        energy dispersion.

        Parameters
        ----------
        show_plot : bool, optional
            Option to plot transmission probability and compare to 
            original calculation.
        
        Notes
        -----
        The transmission probability is a function of energy 
        and off-axis angle of the source.

        The transmission probability can also be calculated using
        the get_total_edisp_matrix method, which is preferred. 
        """
        
        # Only use photons that do not scatter.
        theta_low = self.theta - 0.2
        theta_high = self.theta + 0.2
        condition = (self.incident_angle > theta_low) & (self.incident_angle < theta_high)
        idm_watch = self.idm[condition]
        transmitted_index = np.isin(self.idi,idm_watch)

        # Make array of tranmissted events:
        transmitted_events = Histogram([self.energy_bin_edges],labels=["et [keV]"])
        transmitted_events.fill(self.ei[transmitted_index])
        tranmitted_array = transmitted_events.contents
        
        # Calculate tranmission probability:
        self.tp = tranmitted_array/self.ei_array

        # Save TP to file:
        d = {"energy[keV]":self.emean,"TP":self.tp}
        df = pd.DataFrame(data = d, columns = ["energy[keV]","TP"])
        df.to_csv("tp_beam.dat",sep="\t",index=False)

        if show_plot == True:
            self.plot_tp()

        return
        
    def plot_tp(self):

        """Plot the transmission probability."""

        # Get original TP:
        e_official, tp_official = self.get_tp_from_file(self.emean)

        # Plot new TP:
        plt.semilogx(self.emean, self.tp, ls="--", label="my pipeline")
        plt.semilogx(e_official, tp_official,label="official")
        plt.xlabel("Ei [keV]")
        plt.ylabel("Transmission Probability")
        plt.legend(loc=2,frameon=False)
        plt.savefig("transmission_probability.png")
        plt.show()
        plt.close()

        return

    def get_tp_from_file(self):

        """Read original transmission probability from numpy array (for 33 km).
        This has been used for past COSI studies, including DC1, and it's
        included here for a sanity check. 
        
        Returns
        -------
        tp_energy, tp_array : list, array
            list of energies and array of TP values, respectively.
        """

        # Transmission probability:
        tp_file = os.path.join(self.test_dir,"TP_33km.npy")
        tp_array = np.load(tp_file)
        theta_list = [0.0,5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,\
                45.0,50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,90.0,90.0001,180.0]
        tp_energy = [50.0,60.0,80.0,100.0,150.0,200.0,300.0,400.0,500.0,\
                600.0,800.0,1000.0,1022.0,1250.0,1500.0,2000.0,2044.0,3000.0,\
                4000.0,5000.0,6000.0,7000.0,8000.0,9000.0,10000.0]

        # Save TP official to file:
        d = {"energy[keV]":tp_energy,"TP":tp_array[0]}
        df = pd.DataFrame(data = d, columns = ["energy[keV]","TP"])
        df.to_csv("tp_official.dat",sep="\t",index=False)

        # Get index for theta of off-axis angle:
        theta_list = np.array(theta_list)
        try:
            return_index = np.where(theta_list == self.theta)[0][0]
        except:
            print("WARNING: Incident angle is not in test list. Using 0 degrees.")
            return_index = 0
        
        return tp_energy, tp_array[return_index]

    def get_total_edisp_matrix(self, show_sanity_checks=False, make_plots=True):

        """Get the energy dispersion matrix. 
        
        The total energy dispersion is the sum of the beam photons 
        (which don't scatter) and the scattered photons. Here I calculate 
        all three: beam, scattered, and total. Likewise, I calculate the 
        transmission fraction for all three. 

        Parameters
        ----------
        show_sanity_checks : bool, optional 
            Print a comparison of total energy dispersion to summed 
            energy dispersion (beam + scattered), and also for transmission 
            probability, to verify that they are the same (default is False). 
        make_plots : bool, optional 
            Show plots (default is True). 
        """
        
        # Define condition for beam and scattered component:
        theta_low = self.theta - 0.2
        theta_high = self.theta + 0.2
        condition = (self.incident_angle > theta_low) & (self.incident_angle < theta_high)
        
        # Make beam edisp array:
        idm_watch = self.idm[condition]
        em_watch = self.em[condition]
        ia_watch = self.incident_angle[condition]
        self.edisp_array_beam, \
        self.normed_edisp_array_beam,\
        self.tp_beam = self.make_edisp_matrix(idm_watch, em_watch, ia_watch)

        # Save TP Beam to file:
        d = {"energy[keV]":self.emean,"TP":self.tp_beam}
        df = pd.DataFrame(data = d, columns = ["energy[keV]","TP"])
        df.to_csv("tp_beam.dat",sep="\t",index=False)

        # Make scattered edisp array:
        new_condition = ~condition & (self.incident_angle<88)
        idm_watch = self.idm[new_condition]
        em_watch = self.em[new_condition]
        ia_watch = self.incident_angle[new_condition]
        self.edisp_array_scattered, \
        self.normed_edisp_array_scattered,\
        self.tp_scattered = self.make_edisp_matrix(idm_watch, em_watch, ia_watch)

        # Make total edisp array:
        condition = self.incident_angle<88
        idm_watch = self.idm[condition]
        em_watch = self.em[condition]
        ia_watch = self.incident_angle[condition]
        self.edisp_array_total, \
        self.normed_edisp_array_total,\
        self.tp_total = self.make_edisp_matrix(idm_watch, em_watch, ia_watch, write_hist=True)

        # Calculate total as sum of beam and scattered (sanity check):
        self.normed_edisp_array_summed = \
        self.normed_edisp_array_beam + self.normed_edisp_array_scattered
        diff = self.normed_edisp_array_summed - self.normed_edisp_array_total
        if show_sanity_checks == True:
            print()
            print("difference b/n total and summed edisp arrays:")
            print(diff)
            print()
            self.tp_summed = self.tp_beam + self.tp_scattered
            diff = self.tp_summed - self.tp_total
            print()
            print("difference b/n total and summed TP arrays:")
            print(diff)
            print()
 
        if make_plots == True:

            # Plot edisp:
            self.plot_edisp_matrix(self.normed_edisp_array_beam, "edisp_matrix_beam.png")
            self.plot_edisp_matrix(self.normed_edisp_array_scattered, "edisp_matrix_scattered.png")
            self.plot_edisp_matrix(self.normed_edisp_array_total, "edisp_matrix_total.png")
       
            # Plot transmission probability:
            self.plot_tp_from_edisp()

        return

    def make_edisp_matrix(self, idm_watch, em_watch, ia_watch, write_hist=False):

        """Make energy dispersion matrix.

        Parameters
        ----------
        idm_watch : ArrayLike 
            IDs of measured photons for watched region.
        em_watch : ArrayLike 
            Energy histogram of photons for watched region.
        ia_watch : ArrayLike
            Angle histogram of photons for watched region. 
        write_hist : bool
            Option to save histogram to hdf5.
        """

        # Make edisp array:
        transmitted_index = np.isin(self.idi, idm_watch)
        transmitted_events = Histogram([self.energy_bin_edges, self.energy_bin_edges, self.incident_ang_bins],\
                labels=["Ei [keV]", "Em [keV]", "theta_prime [deg]"])
        transmitted_events.fill(self.ei[transmitted_index], em_watch, ia_watch)
         
        # Save response matrix:
        if write_hist == True:
            transmitted_events.write('atm_response.hdf5', overwrite=True)
        
        # Project onto em, ei:
        self.edisp_array = np.array(transmitted_events.project(["Em [keV]", "Ei [keV]"]))
        
        # Normalized edisp array:
        norm = self.ei_array[None,:]
        self.normed_edisp_array = self.edisp_array / norm

        # Transmission probability:
        # axis=0 implies projection onto Ei, 
        # which gives the probability that a photon started at Ei will
        # reach the detecting area (regardless of its measured energy).
        # axis=1 implies projection onto Em,
        # which gives the probability that a photon will be measured 
        # at Em (regardless of its initial energy).
        # However, projection onto Em depends on the initial spectrum. 
        self.tp = self.normed_edisp_array.sum(axis=0)

        return self.edisp_array, self.normed_edisp_array, self.tp

    def plot_edisp_matrix(self, matrix, savefile):

        """Plot energy dispersion matrix."""

        fig = plt.figure(figsize=(7.5,6))
        ax = plt.gca()
        img = ax.pcolormesh(self.energy_bin_edges, self.energy_bin_edges, matrix, cmap="viridis")
        ax.set_xscale('log')
        ax.set_yscale('log')
        cbar = plt.colorbar(img,fraction=0.045)
        cbar.set_label("Ratio", fontsize=14)
        plt.xlabel("Ei [keV]", fontsize=14)
        plt.ylabel("Em [keV]", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(savefile,dpi=600)
        plt.show()
        plt.close()

        return

    def plot_tp_from_edisp(self,tp_file=None):

        """Plot the transmission probability.
        
        Parameters
        ----------
        tp_file : str, optional
            Option to overlay TP from analytical calculation in plot. 
            Specify file name from TPCalc class. Default is None. 
        """
        
        # Get original TP 
        # Sanity check for now. Can probably remove soon. 
        #e_official, tp_official = self.get_tp_from_file()
        #plt.semilogx(e_official, tp_official, marker="", ls="-", color="black", label="original")

        # Plot TP:
        plt.semilogx(self.emean, self.tp_total, marker="o", ls="--", label="Total")
        plt.semilogx(self.emean, self.tp_beam, marker="s", ls="-", label="Transmitted")
        plt.semilogx(self.emean, self.tp_scattered, marker="^", ls=":", label="Scattered")
        
        # Plot TP from analytical calculation:
        if tp_file != None:
            df = pd.read_csv(tp_file,delim_whitespace=True)
            plt.semilogx(df["energy[MeV]"]*1000, df["TP"], marker="", ls="-", color="grey", label="Analytical")
        
        plt.xlabel("Ei [keV]", fontsize=14)
        plt.ylabel("Detection Fraction", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(10,1e4)
        plt.legend(loc=2,ncol=1,frameon=False,fontsize=12)
        plt.grid(ls=":",color="grey",alpha=0.4)
        plt.savefig("transmission_probability_from_edisp.pdf")
        plt.show()
        plt.close()

        return

    def PL(self, E, p):

        """General PL spectral model, normalized to 1e-3 at 100 keV. 

        Parameters
        ----------
        E : float 
            Energy in keV.
        p : float 
            Power of dN/dE. 
        
        Returns
        -------
        dn/de : float
            Value at given energy. 
        """
        
        return (1e-3)*(E/100.0)**(-1*p)
    
    def PL_interp(self, p):
    
        """Returns interpolated PL function for energies between 
        100 keV to 10 MeV. 

        Paramters
        ---------
        p : float
            Power of dN/dE 

        Returns
        -------
        pl_interp : scipy:interpolate:interp1d
            Interpolated function. 
        """

        # Energy array b/n 100 keV to 10 MeV.
        energy = np.logspace(np.log10(1e2),np.log10(1e4),20)
        
        # Define PL spectrum in dN/dE:
        pl_interp =  interp1d(energy, self.PL(energy,p), kind='linear', fill_value="extrapolate")

        return pl_interp

    def ff_correction(self, model_flux, name, show_plots=True):

        """Calculate atmospheric correction factor by forward folding the 
        atmospheric energy dispersion with the model counts. 

        The energy dispersion matrices must first be generated via
        the 'get_total_edisp_matrix' method. 

        Parameters
        ----------
        model_flux : scipy:interpolate:interp1d
            Interp1d object giving the model flux as a function of energy.
        name : str 
            Name of saved files.
        show_plots : bool 
            Option to plot correction factor and ratio.
        """

        # Get integrated counts for each energy bin:
        N_list = []
        for i in range(0,len(self.energy_bin_edges)-1):
            elow = self.energy_bin_edges[i]
            ehigh = self.energy_bin_edges[i+1]
            int_flux = integrate.quad(model_flux,elow,ehigh)
            N_list.append(int_flux[0])
        N_list = np.array(N_list)

        # Matrix multiplication to get predicted counts (for total, beam, and scattered):
        p_total = np.matmul(self.normed_edisp_array_total,N_list)
        p_beam = np.matmul(self.normed_edisp_array_beam,N_list)
        p_scattered = np.matmul(self.normed_edisp_array_scattered,N_list)

        # Correction factor:
        c_total = p_total/N_list
        c_beam = p_beam/N_list
        
        # Correction factor ratio:
        c_ratio = c_total/c_beam

        # Save FF correction to file:
        d = {"energy[keV]":self.emean,"c_total":c_total,"c_beam":c_beam,"c_ratio": c_ratio}
        df = pd.DataFrame(data = d, columns = ["energy[keV]","c_total","c_beam","c_ratio"])
        df.to_csv("ff_correction_factor_%s.dat" %name, sep="\t", index=False)
        
        # Calculate scattering fraction that enters detecting area:
        s_frac = p_scattered / p_total
        # Save to file:
        d = {"energy[keV]":self.emean,"s_frac":s_frac}
        df = pd.DataFrame(data = d, columns = ["energy[keV]","s_frac"])
        df.to_csv("s_frac.dat",sep="\t",index=False)

        if show_plots == True:

            # Plot correction factor:
            plt.figure(figsize=(7.5,5.5))
            plt.semilogx(self.emean, c_total, marker="o", lw=2, label="transmitted + scattered")
            plt.semilogx(self.emean, c_beam, marker="s", lw=2, label="transmitted")
            plt.semilogx(self.emean, self.tp_beam, ls="--", lw=2, label="TP")
            plt.xlabel("Energy [keV]", fontsize=14)
            plt.ylabel("FF Correction Factor", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(loc=2,frameon=False,fontsize=12)
            plt.grid(ls=":",color="grey",alpha=0.5)
            plt.savefig("ff_correction_%s.png" %name)
            plt.show()
            plt.close()

            # Plot correction factor ratio:
            plt.figure(figsize=(7.5,5.5))
            plt.semilogx(self.emean[2:], c_ratio[2:], marker="o")
            plt.xlabel("Energy [keV]", fontsize=14)
            plt.ylabel("FF Correction Factor Ratio", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlim(1e1,1e4)
            plt.grid(ls=":",color="grey",alpha=0.5)
            plt.savefig("ff_correction_ratio_%s.png" %name)
            plt.show()
            plt.close()

        return 

