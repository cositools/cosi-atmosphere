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
from mhealpy import HealpixMap, HealpixBase
from numpy.linalg import norm
import h5py

class ProcessSpherical(ProcessSims.Process):

    """Analyze atmophere simulations from spherical mass model.
        
    Parameters
    ----------
    r_alt : float 
        Altitude of observations in km. 
    all_events_file : str, optional 
        Event file with all thrown events (default is output 
        from ParseSims method). 
    measured_events_file : str, optional 
        Event file with all measured events (default is output 
        from ParseSims method).  
    """

    def __init__(self, r_alt, all_events_file="all_thrown_events.dat", \
            measured_events_file="event_list.dat"):
     
        # Get test directory:
        path_prefix = os.path.split(ProcessSims.__file__)[0]
        self.test_dir = os.path.join(path_prefix,"data_files")

        # Define Earth radius:
        self.r_earth = 6.378e8 * (1e-5) # km

        # Define altitude of observations:
        self.r_alt = r_alt # km

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
        self.zm = np.array(df["zm[cm]"]) # measured z
        self.xdm = np.array(df["xdm[cm]"]) # measured x direction
        self.ydm = np.array(df["ydm[cm]"]) # measured y direction
        self.zdm = np.array(df["zdm[cm]"]) # measured z direction
 
        #  Define vector array for initial direction:
        keep_index = np.isin(self.idi,self.idm)
        vi = np.array([self.xdi,self.ydi,self.zdi]).T

        # Define vector array for measured direction:
        vm = np.array([self.xdm,self.ydm,self.zdm]).T
   
        # Spherical coordinates for initial position:
        self.sp_coords_ri = astropy.coordinates.cartesian_to_spherical(self.xi,self.yi,self.zi)
        self.ri = self.sp_coords_ri[0] * (1e-5) # km
        self.lat_ri = self.sp_coords_ri[1].deg # deg
        self.lon_ri = self.sp_coords_ri[2].deg # deg

        # Spherical coordinates for initial direction:
        self.sp_coords_di = astropy.coordinates.cartesian_to_spherical(self.xdi,self.ydi,self.zdi)
        self.lat_di = self.sp_coords_di[1].deg  # deg
        self.lon_di = self.sp_coords_di[2].deg  # deg

        # Spherical coordinates for measured position:
        self.sp_coords_rm = astropy.coordinates.cartesian_to_spherical(self.xm,self.ym,self.zm)
        self.rm = self.sp_coords_rm[0] * (1e-5) # km
        self.lat_rm = self.sp_coords_rm[1].deg # deg
        self.lon_rm = self.sp_coords_rm[2].deg # deg

        # Spherical coordinates for measured direction:
        self.sp_coords_dm = astropy.coordinates.cartesian_to_spherical(self.xdm,self.ydm,self.zdm)
        self.lat_dm = self.sp_coords_dm[1].deg  # deg
        self.lon_dm = self.sp_coords_dm[2].deg # deg

        # Define surface normal for measured photons:
        nm = -1*np.array([self.xm,self.ym,self.zm]).T

        # Get points of intersection:
        o = np.array([self.xi,self.yi,self.zi]).T # cm
        u = np.array([self.xdi,self.ydi,self.zdi]).T # cm
        r = (self.r_earth + self.r_alt) * 1e5 # cm
        intersecting_points = self.find_projection(o,u,r)
        
        # Define surface normal for initial photons:
        ni = -1*intersecting_points
        ni[keep_index] = nm
 
        # Get incident angle for measured photons:
        self.incident_angle_m = self.angle(vm,nm)*(180/np.pi) # degrees

        # Get incident angle for initial photons:
        self.incident_angle_i = self.angle(vi,ni)*(180/np.pi) # degrees

        print()
        print("Total number of initial events: " + str(len(self.idi[:])))
        print("Total number of unmeasured events: " + str(len(self.idi[~keep_index])))
        print("WARNING: Not all incident angles are defined.")
        print("Number of undefined incident angles: " + str(len(self.incident_angle_i[np.isnan(self.incident_angle_i)])))
        print()

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

    def line_point(self,o,u,d):

        """Finds  point on a line segment.

        Parameters
        ----------
        o : array-like 
            Point on line. In our case it's the starting photon position.
        u : array-like
            Direction of line segment. In our case it's the starting photon direction.
        d: array-like 
            Distance from specified point (o) in the specified direction (u).
        
        Returns
        -------
        x : array-like
            Point on line. 
        """
    
        # Standard equation of a line
        x = o + (d*u.T).T

        return x

    def find_projection(self,o,u,r):
    
        """Finds projection of line segment onto sphere.
        
        Everything should be done in Cartesian coordinates.
        The closest point of intersection defines the normal vector to 
        the surface, relative to the initial direction.
        The true incident angle (before scattering) is the angle 
        between these two vectors.

        Parameters
        ----------
        o : array-like 
            Point on line. In our case it's the starting photon position.
        u : array_like
            Direction of line segment. In our case it's the starting photon direction.
        r : float
            Radius of watched sphere. Default is r_earth + 33.1 km. 

        Returns
        -------
        intersection : array
            Point of intersection

        Note
        ----
        All inputs must have self-consistent units.  
        """

        # Solve for d.
        # This is the solution of a quadratic equation, so we want the 
        # shortest distance, which is the first intercept. 
        dp = self.dot_product(u,o)
        sqrt = np.sqrt( dp**2 - norm(u,axis=1)**2 * (norm(o,axis=1)**2 - r**2) )
        print()
        print("Finding intersection...")
        print("Number of photons with no solution: " + str(len(sqrt[np.isnan(sqrt)])))
        print()
        num_plus = -1* dp + sqrt
        num_min = -1* dp - sqrt
        denom = norm(u,axis=1)**2
        sol1 = num_plus/denom
        sol2 = num_min/denom
        min_d = np.minimum(sol1,sol2)

        # Get intersecting point:
        intersection = self.line_point(o,u,min_d)

        return intersection

    def bin_sim(self, name, elow=10, ehigh=10000, num_ebins=17, \
            anglow=0, anghigh=100, num_angbins=26,  nside=16, \
            scheme='ring', weighted=True):

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
        nside : int, optional 
            nside for healpix binning (defualt is 16).
        scheme : str, optional 
            Scheme for healpix binning (ring or nested). Default is ring.
        weighted : bool, optional
            Wether or nor to use a weighted histogram (default is True).

        Note
        ----
        Response file contains 5 different histograms, each stored as 
        its own group. The group names are:
        1. primary_rsp
        2. starting_photons
        3. starting_pe, ons_rsp
        4. measured_photons
        5. measured_photons_rsp
        """
        
        # Define energy bin edges: 
        self.energy_bin_edges = np.logspace(np.log10(elow), np.log10(ehigh), num_ebins)
        
        # Define radial bin edges (ri is normalized relative to Earth's surface):
        rlow = self.r_earth 
        rhigh = self.r_earth + 4000
        num_rbins = 4000
        self.radial_bins = np.linspace(rlow, rhigh, num_rbins)
        
        # Define incident angle bin edges (for initial and measured):
        # Using 4 deg resolution for now.
        self.incident_ang_bins = np.linspace(anglow,anghigh,num_angbins)
        print()
        print("Theta bins:")
        print(self.incident_ang_bins)

        # Healpix parameters:
        print()
        print("Using NSIDE %s" %str(nside))
        print("Approximate resolution at NSIDE {} is {:.2} deg".format(nside, hp.nside2resol(nside, arcmin=True) / 60))
        print()
        npix = hp.nside2npix(nside)
        self.ang_bin_edges = np.arange(0,npix+1,1)
        
        # Bin initial longitude and latitude position with healpix:
        self.m_ri = HealpixMap(nside = nside, scheme = scheme, dtype = int)
        self.ang_pixs_ri = self.m_ri.ang2pix(self.lon_ri,self.lat_ri,lonlat=True)
        
        # Bin initial longitude and latitude direction with healpix:
        self.m_di = HealpixMap(nside = nside, scheme = scheme, dtype = int)
        self.ang_pixs_di = self.m_di.ang2pix(self.lon_di,self.lat_di,lonlat=True)

        # Bin measured longitude and latitude position with healpix:
        self.m_rm = HealpixMap(nside = nside, scheme = scheme, dtype = int)
        self.ang_pixs_rm = self.m_rm.ang2pix(self.lon_rm,self.lat_rm,lonlat=True)
        
        # Bin measured longitude and latitude direction with healpix:
        self.m_dm = HealpixMap(nside = nside, scheme = scheme, dtype = int)
        self.ang_pixs_dm = self.m_dm.ang2pix(self.lon_dm,self.lat_dm,lonlat=True)
    
        # Make histogram for starting photons:
        self.starting_photons = Histogram([self.energy_bin_edges, self.radial_bins, self.ang_bin_edges, self.ang_bin_edges], \
                labels=["Ei [keV]", "ri [km]", "ang_ri_pixs", "ang_di_pixs"], sparse=True)
        self.starting_photons.fill(self.ei, self.ri, self.ang_pixs_ri, self.ang_pixs_di)

        # Make histogram for starting photons response:
        keep_index = np.isin(self.idi,self.idm)
        self.starting_photons_rsp = Histogram([self.energy_bin_edges, self.incident_ang_bins],\
                labels=["Ei [keV]", "theta_i [deg]"], sparse=True)
        self.starting_photons_rsp.fill(self.ei, self.incident_angle_i)

        # Make histogram for measured photons:
        self.measured_photons = Histogram([self.energy_bin_edges, self.radial_bins, self.ang_bin_edges, self.ang_bin_edges],\
                labels=["Em [keV]", "rm [km]", "ang_rm_pixs", "ang_dm_pixs" ], sparse=True)
        self.measured_photons.fill(self.em, self.rm, self.ang_pixs_rm, self.ang_pixs_dm)

        # Make histogram for measured photons response:
        self.measured_photons_rsp = Histogram([self.energy_bin_edges, self.incident_ang_bins],\
                labels=["Em [keV]", "theta_m [deg]"], sparse=True)
        self.measured_photons_rsp.fill(self.em, self.incident_angle_m)

        # Make primary response atmospheric response matric:
       
        # Get index for photons that reach the watched volume:
        keep_index = np.isin(self.idi,self.idm)

        print("Number of simulated events: " + str(len(self.idi)))
        print("Number of events that reached watched volume: " + str(len(self.idm)))
    
        self.primary_rsp = Histogram([self.energy_bin_edges, self.energy_bin_edges,\
                self.incident_ang_bins, self.incident_ang_bins],\
                labels=["Ei [keV]", "Em [keV]", "theta_i [deg]", "theta_m [deg]"], sparse=True)
        if weighted == True:
            self.primary_rsp.fill(self.ei[keep_index], self.em, \
                self.incident_angle_i[keep_index], self.incident_angle_m, \
                weight=np.cos(np.deg2rad(self.incident_angle_i[keep_index]))/np.cos(np.deg2rad(self.incident_angle_m)))
        if weighted == False:
            self.primary_rsp.fill(self.ei[keep_index], self.em, \
                self.incident_angle_i[keep_index], self.incident_angle_m)

        # Write response to file:
        savefile = name 
        self.starting_photons.write(savefile, name='starting_photons', overwrite=True)
        self.starting_photons_rsp.write(savefile, name ='starting_photons_rsp', overwrite=True)
        self.measured_photons.write(savefile, name='measured_photons', overwrite=True)
        self.measured_photons_rsp.write(savefile, name='measured_photons_rsp', overwrite=True)
        self.primary_rsp.write(savefile, name='primary_rsp', overwrite=True)
        
        # Get binning info:
        self.get_binning_info()

        return

    def get_binning_info(self):

        """Get info from histogram."""

        # Energy info:
        self.energy_bin_centers = self.primary_rsp.axes["Ei [keV]"].centers
        self.energy_bin_widths = self.primary_rsp.axes["Ei [keV]"].widths
        self.energy_bin_edges = self.primary_rsp.axes["Ei [keV]"].edges

        # Get mean energy:
        emean_list = []
        for i in range(0,len(self.energy_bin_edges)-1):
            emean = np.sqrt(self.energy_bin_edges[i]*self.energy_bin_edges[i+1])
            emean_list.append(emean)
        self.emean = np.array(emean_list)
        
        # Radius info:
        self.radial_bins = self.starting_photons.axes["ri [km]"].edges
        self.radial_bin_centers = self.starting_photons.axes["ri [km]"].centers
        self.radial_bin_widths = self.starting_photons.axes["ri [km]"].widths

        # Incident angle bins:
        self.incident_ang_centers = self.primary_rsp.axes["theta_i [deg]"].centers
        self.incident_ang_bins = self.primary_rsp.axes["theta_i [deg]"].edges

        # Projected histograms:
        self.ei_hist = self.starting_photons.project("Ei [keV]").contents.todense()
        self.ei_array = np.array(self.ei_hist)
        self.em_hist = self.measured_photons.project("Em [keV]").contents.todense()
        self.em_array = np.array(self.em_hist)
        self.ri_hist = self.starting_photons.project("ri [km]").contents.todense()
        self.ri_array = np.array(self.ri_hist)
        self.rm_hist = self.measured_photons.project("rm [km]").contents.todense()
        self.rm_array = np.array(self.rm_hist)
        
        return

    def make_scattering_plots(self, pos_init=True, pos_meas=True, pos_proj=True,\
            pos_ang_ri=True, pos_ang_di=True, ri=True,\
            pos_ang_rm=True, pos_ang_dm=True, rm=True,\
            theta_dist=True, ang_dist=None, spec=True, rsp_file=None):
        
        """Visualize the simulated photons. 

        Parameters
        ----------
        pos_init : bool, optional 
            3d scatter plot of initial postions
        pos_meas : bool, optional 
            3d scatter plot of measured positions
        pos_proj : bool, optional 
            2d projection onto xy-axis of initial positions
        pos_ang_ri : bool, optional 
            healpix map of initial positions
        pos_ang_di : bool, optional 
            healpix map of initial directions
        ri : bool, optional
            initial radius (relative to both Earth center and surrounding sphere disk)
        pos_ang_rm : bool, optional 
            healpix map of measured positions
        pos_ang_dm : bool, optional 
            healpix map of measured directions
        rm : bool, optional 
            measured radius with respect to Earth center 
        theta_dist : bool, optional 
            distributions of incident angle (initial and measured) 
        ang_dist : list, optional 
            Distribution of measured anlges for a given initial angle.
            Takes list with incident angle in degrees and energy in keV. 
            Default is None. 
        spec : bool, optional
            Spectrum of initial and measured photons (dN/dE). 
        rsp_file : str, optional
            Name of response file to use.
        """
       
        # Make sure response is loaded:
        if rsp_file != None:
            self.load_response(rsp_file)
        try:
            self.incident_ang_bins
        except:
            print("ERROR: Need to load response file.")
            sys.exit()
        
        # Initial photons 3d position: 
        if pos_init == True:
            ax = plt.axes(projection ="3d")
            sp = ax.scatter3D(self.xi,self.yi,self.zi,c=self.lat_ri,\
                alpha=1,cmap="viridis",depthshade=True)
            plt.colorbar(sp,shrink=0.5,label="Latitude",pad=0.15)
            ax.set_xlabel('x [cm]')
            ax.set_ylabel('y [cm]')
            ax.set_zlabel('z [cm]')
            plt.title("Initial Photon Position")
            plt.savefig("initial_photon_position.png")
            plt.show()
            plt.close()
 
        # Measured photons 3d position:
        if pos_meas == True:
            ax = plt.axes(projection ="3d")
            sp = ax.scatter3D(self.xm,self.ym,self.zm,c=self.lat_rm,\
                alpha=1,cmap="viridis",depthshade=True)
            plt.colorbar(sp,shrink=0.5,label="Latitude",pad=0.15)
            ax.set_xlabel('x [cm]')
            ax.set_ylabel('y [cm]')
            ax.set_zlabel('z [cm]')
            plt.title("Measured Photon Position")
            plt.savefig("measured_photon_position.png")
            plt.show()
            plt.close()

        # Measured and initial photons 2d position:
        if pos_proj == True:
            ax = plt.axes()
            ax.scatter(self.xi,self.yi,color="cornflowerblue",label="initial")
            ax.scatter(self.xm,self.ym,color="darkorange",label="measured")
            ax.set_xlabel('x [cm]')
            ax.set_ylabel('y [cm]')
            circ = plt.Circle((0,0),6.378e8,zorder=10,color="cyan",
                    fill=False,label="Earth radius")
            ax.set_aspect(1)
            ax.add_artist(circ)
            plt.title("Photon Position xy Projection")
            plt.legend(frameon=False,loc=2,ncol=1)
            plt.savefig("measured_photon_xy_projection.png")
            plt.show()
            plt.close()

        # Plot initial angular position:
        if pos_ang_ri == True:
            h = self.starting_photons.project("ang_ri_pixs")
            m = HealpixMap(base = HealpixBase(npix = h.nbins), data = h.contents.todense())
            plot,ax = m.plot('mollview')
            ax.coords.grid(True, color='black', ls='dotted')
            ax.get_figure().set_figwidth(4)
            ax.get_figure().set_figheight(3)
            plt.title("Initial Position")
            plt.savefig("initial_position_healpix.png",bbox_inches='tight')
            plt.show()
            plt.close()

        # Plot initial angular direction:
        if pos_ang_di == True:
            h = self.starting_photons.project("ang_di_pixs")
            m = HealpixMap(base = HealpixBase(npix = h.nbins), data = h.contents.todense())
            plot,ax = m.plot('mollview')
            ax.coords.grid(True, color='black', ls='dotted')
            ax.get_figure().set_figwidth(4)
            ax.get_figure().set_figheight(3)
            plt.title("Initial Direction")
            plt.savefig("initial_direction_healpix.png",bbox_inches='tight')
            plt.show()
            plt.close()

        # Distribution of radius for initial photons:
        if ri == True:
           
            # First calculate relative to surrounding sphere disk:

            # Define rdisk of surrounding sphere disk:
            rsphere = self.r_earth + 200
            rdisk = np.sqrt(self.radial_bins**2 - rsphere**2) 

            # Get rdisk width:
            rdisk_width = []
            rdisk_mean = []
            for i in range(0,len(rdisk)-1):
                rdisk_width.append(rdisk[i+1] - rdisk[i])
            rdisk_width = np.array(rdisk_width)
          
            # Calculate the differential area:
            # Note: I use the full expression, which includes the
            # squared differential term. This is often ignored, 
            # but is important to include for r near 0. 
            area = 2*np.pi*rdisk[:-1]*rdisk_width + np.pi*(rdisk_width**2)
            
            # Calculate rate:
            rate = self.ri_array/(area)
            
            # Get mean rate from disk:
            good_index = (~np.isnan(rdisk[:-1])) & (rdisk[:-1] < rsphere)
            good_rate = rate[good_index]
            self.f_0 = np.mean(good_rate) # mean rate: ph/km2
            
            # Plot:
            plt.plot(rdisk[:-1], rate, color="cornflowerblue")
            plt.axvline(x=self.r_earth, color="darkorange", linestyle=":", label="r_earth")
            plt.axvline(x=self.r_earth+200, color="green", linestyle=":", label="r_earth+200")
            plt.ylabel("counts/area [$\mathrm{ph \ km^{-2}}$]", fontsize=14)
            plt.xlabel("r_disk [km]", fontsize=14)
            plt.grid(ls="--",color="grey",alpha=0.3)
            plt.legend(frameon=True,loc=2)
            plt.ylim(0.05,0.1)
            plt.xlim(0,rsphere*1.2)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.savefig("ri_dist_disk_center.pdf")    
            plt.show()
            plt.close()
            
            # Plot relative to Earth center:
            area = 2*np.pi*(self.radial_bins[:-1])*self.radial_bin_widths
            rate = self.ri_array/(area)
            
            # Plot:
            plt.plot(self.radial_bin_centers, rate, color="cornflowerblue")
            plt.axvline(x=self.r_earth, color="darkorange", linestyle=":", label="r_earth")
            plt.axvline(x=self.r_earth+200, color="green", linestyle=":", label="r_earth+200")
            plt.ylabel("counts/area [$\mathrm{ph \ km^{-2}}$]", fontsize=14)
            plt.xlabel("ri [km]", fontsize=14)
            plt.grid(ls="--",color="grey",alpha=0.3)
            plt.legend(frameon=True,loc=1)
            plt.xlim(6e3,1e4)
            plt.ylim(0,1.4*np.amax(self.ri_array/(area)))
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.savefig("ri_dist_earth_center.pdf")    
            plt.show()
            plt.close()

        # Plot measured angular position:
        if pos_ang_rm == True:
            h = self.measured_photons.project("ang_rm_pixs")
            m = HealpixMap(base = HealpixBase(npix = h.nbins), data = h.contents.todense())
            plot,ax = m.plot('mollview')
            ax.coords.grid(True, color='black', ls='dotted')
            ax.get_figure().set_figwidth(4)
            ax.get_figure().set_figheight(3)
            plt.title("Measured Position")
            plt.savefig("measured_position_healpix.png",bbox_inches='tight')
            plt.show()
            plt.close()

        # Plot measured angular direction:
        if pos_ang_dm == True:
            h = self.measured_photons.project("ang_dm_pixs")
            m = HealpixMap(base = HealpixBase(npix = h.nbins), data = h.contents.todense())
            plot,ax = m.plot('mollview')
            ax.coords.grid(True, color='black', ls='dotted')
            ax.get_figure().set_figwidth(4)
            ax.get_figure().set_figheight(3)
            plt.title("Measured Direction")
            plt.savefig("measured_direction_healpix.png",bbox_inches='tight')
            plt.show()
            plt.close()

        # Distribution of radius for measured photons:
        if rm == True:
            plt.plot(self.radial_bin_centers, self.rm_array, color="cornflowerblue")
            plt.axvline(x=self.r_earth, color="darkorange", linestyle=":", label="r_earth")
            plt.axvline(x=self.r_earth+200, color="green", linestyle=":", label="r_earth+200")
            plt.ylabel("counts [$\mathrm{ph}$]", fontsize=14)
            plt.xlabel("rm [km]", fontsize=14)
            plt.grid(ls="--",color="grey",alpha=0.3)
            plt.legend(frameon=True,loc=1)
            plt.xlim(6.1e3,6.9e3)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.savefig("rm_dist.pdf")    
            plt.show()
            plt.close()

        # Plot theta_distributions:
        if theta_dist == True:

            # Average flux needs to be calculated from ri.
            # Make sure that this has been done:
            try: 
                self.f_0
            except: 
                print("Average flux is not calculated.")
                print("You must set ri = True")
                sys.exit()

            # Solution for the flux of a constant field
            # through a hemisphere. 
            def flux(theta):
                
                R = self.r_earth + self.r_alt # km
                f_0 = self.f_0 # mean rate from disk in ph/km^2
                c = 2*np.pi*(R**2)*f_0
               
                print()
                print("calculation of uniform flux through hemisphere:")
                print("radius of hemisphere [km]: " + str(R))
                print("mean rate from surrounding sphere disk [ph/km^2]: " + str(f_0))
                print()

                return c*np.sin(theta)*np.cos(theta)
            
            analytical_soln = flux(self.incident_ang_bins*(np.pi/180))
            
            # Need to compare the analytical solution to counts/angle in steradians:
            d_theta = 4 * (np.pi/180)

            # Plot distributions:
            theta_i = np.array(self.starting_photons_rsp.project("theta_i [deg]").contents.todense())
            theta_m = np.array(self.measured_photons_rsp.project("theta_m [deg]").contents.todense())
            theta_m_weighted = np.array(self.primary_rsp.project("theta_m [deg]").contents.todense())
            plt.stairs(theta_i/d_theta,self.incident_ang_bins,label="initial",ls="-",lw=2)
            plt.stairs(theta_m/d_theta,self.incident_ang_bins,label="measured",ls="--",lw=2)
            plt.stairs(theta_m_weighted/d_theta,self.incident_ang_bins,label="measured (weighted)",ls="--",lw=2)
            plt.plot(self.incident_ang_bins,analytical_soln,label="analytical",ls=":",lw=3,color="grey",zorder=0,alpha=0.8)
            plt.yscale("log")
            plt.ylabel(r"$\Delta \Phi / \Delta \theta$  [$\mathrm{ph \ rad^{-1}}$]", fontsize=14)
            plt.xlabel(r"$\theta \ [\circ]$", fontsize=14)
            plt.ylim(1e4,1e8)
            plt.xlim(0,100)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(ls="--",color="grey",alpha=0.3)
            plt.legend(loc=1,ncol=2,frameon=False,fontsize=12)
            plt.savefig("theta_distributions.pdf")
            plt.show()
            plt.close()

            # Plot fraction:
            ang_bins = np.array(self.incident_ang_bins)

            # Note: Need to avoid overflow bin, so we take len(ang_bins) - 1:
            frac = (theta_m[0:len(ang_bins)-1]/theta_i[0:len(ang_bins)-1])
            plt.plot(ang_bins[0:len(ang_bins)-1],frac,ls="--",marker="o",color="black")
            plt.ylabel(r"$N_m / N_i$", fontsize=14)
            plt.xlabel(r"$\theta \ [\circ]$", fontsize=14)
            plt.xlim(0,100)
            plt.ylim(0,1)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(ls="--",color="grey",alpha=0.3)
            plt.savefig("theta_i_m_residual.pdf")
            plt.show()
            plt.close()
    
        if ang_dist != None:
         
            # Get angular bin:
            ang_bin = self.get_theta_bin(ang_dist[0])
            print("Using incident angle of %s deg" %str(ang_dist[0]))
            print("Corresponding to angular bin %s" %str(ang_bin))

            # Get energy bin:
            energy_bin = self.get_energy_bin(ang_dist[1])
            print("Using energy of %s keV" %str(ang_dist[1]))
            print("Corresponding to energy bin %s" %str(energy_bin))

            # Plot weighted histogram:
            dist = self.primary_rsp.slice[{"theta_i [deg]":ang_bin,"Ei [keV]":energy_bin}].project(["theta_m [deg]"]).contents.todense()
            plt.semilogy(self.incident_ang_centers,dist,marker='o',ls="-",label=r"weighted by cos($\theta_i$)/cos($\theta_m$)")
           
            # Plot un-weighted histogram:
            ang_i = self.incident_ang_centers[ang_bin] * (np.pi/180)
            new = (1.0/(np.cos(ang_i)/np.cos((np.pi/180.0)*self.incident_ang_centers)))*dist
            plt.semilogy(self.incident_ang_centers,new,marker='o',ls="-.", color="red",label=r"un-weighted")

            # Plot alternative weighting, which includes additional sin factor:
            # This is an alternative weighting, which I'll keep here for now. 
            #new = (np.sin(ang_i)/np.sin((np.pi/180.0)*self.incident_ang_centers))*dist
            #plt.semilogy(self.incident_ang_centers,new,marker='o',ls="--", 
            #color="cyan",label=r"weighted by cos($\theta_i$)sin($\theta_i$)/cos($\theta_m$)sin($\theta_m$)")
            
            plt.xlabel(r"$\theta_m$ [$\circ$]",fontsize=14)
            plt.ylabel("Counts",fontsize=14)
            plt.grid(ls="--",color="grey",alpha=0.4)
            plt.ylim(1,1e7)
            plt.legend(loc=1,frameon=True)
            savefile = "ang_dist_%s_%s.pdf" %(str(ang_dist[0]),str(ang_dist[1]))
            plt.savefig(savefile)
            plt.show()

        # Photon spectrum (dN/dE):
        if spec == True:

            # Starting distribution:
            print("Initial energy distribution:")
            underflow = self.starting_photons_rsp.project("Ei [keV]")[-1]
            overflow = self.starting_photons_rsp.project("Ei [keV]")[self.measured_photons_rsp.end]
            print("underflow bin: " + str(underflow))
            print("overflow bin: " + str(overflow))

            # Measured distribution:
            print("Measured energy distribution")
            underflow = self.measured_photons_rsp.project("Em [keV]")[-1]
            overflow = self.measured_photons_rsp.project("Em [keV]")[self.measured_photons_rsp.end]
            print("underflow bin: " + str(underflow))
            print("overflow bin: " + str(overflow))

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
            plt.xlim(1e1,1e4)
            plt.xlabel("Energy [keV]")
            plt.ylabel("dN/dE [ph/keV]")
            plt.grid(ls="--",color="grey",alpha=0.3)
            plt.legend(frameon=False)
            plt.savefig("energy_dist.png")    
            plt.show()
            plt.close()

            # Print number of starting and measured photons as sanity check:
            print()
            print("Number of starting photons: " + str(self.ei_array.sum()))
            print("Number of measured photons: " + str(self.em_array.sum()))
            print()
             
        return
   
    def load_response(self, name):

        """Load response file and corresponding normalization.

        Parameters
        ----------
        name, str
            Name of response file.
        """
    
        savefile = name 
        self.primary_rsp = Histogram.open(savefile, name='primary_rsp')
        self.starting_photons = Histogram.open(savefile, name='starting_photons')
        self.starting_photons_rsp = Histogram.open(savefile, name='starting_photons_rsp')
        self.measured_photons = Histogram.open(savefile, name='measured_photons')
        self.measured_photons_rsp = Histogram.open(savefile, name='measured_photons_rsp')
        self.get_binning_info()

        return

    def get_total_edisp_matrix(self, theta, make_plots=True, rsp_file=None, tp_file=None):

        """Get the energy dispersion matrix. 
        
        The total energy dispersion is the sum of the transmitted 
        photons (which don't scatter) and the scattered photons. 
        Here I calculate all three: transmitted, scattered, and total. 
        Likewise, I calculate the transmission probability for all three. 

        Parameters
        ----------
        theta : float
            Incident angle of source in degrees. 
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

        # Normalize response:
        self.normalize_response()

        # Make transmitted edisp array:
        self.normed_edisp_array_beam, \
        self.tp_beam = self.make_edisp_matrix(theta_bin,comp="transmitted")

        # Save TP Beam to file:
        d = {"energy[keV]":self.emean,"TP":self.tp_beam}
        df = pd.DataFrame(data = d, columns = ["energy[keV]","TP"])
        df.to_csv("tp_beam.dat",sep="\t",index=False)

        # Make scattered edisp array:
        self.normed_edisp_array_scattered, \
        self.tp_scattered = self.make_edisp_matrix(theta_bin,comp="scattered")

        # Make total edisp array:
        self.normed_edisp_array_total, \
        self.tp_total = self.make_edisp_matrix(theta_bin,comp="total")
 
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

    def get_theta_bin(self,theta,rsp_file=None):

        """Returns index of the theta bin.
        
        Parameters
        ----------
        theta : float
            Incident angle in degrees.
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
        for i in range(0,len(self.incident_ang_bins)):
            try:
                if (theta >= self.incident_ang_bins[i]) & (theta < self.incident_ang_bins[i+1]):
                    theta_index = i
                    break
            except: 
                print("ERROR: theta not found.")
                sys.exit()
    
        return theta_index

    def get_energy_bin(self,energy,rsp_file=None):

        """Returns index of the energy bin.
        
        Parameters
        ----------
        energy : float
            Energy in keV.
        rsp_file : str, optional
            Prefix of response file to load.

        Returns
        -------
        energy_index : int
            Index of energy bin. 
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
        for i in range(0,len(self.energy_bin_edges)):
            try:
                if (energy >= self.energy_bin_edges[i]) & (energy < self.energy_bin_edges[i+1]):
                    energy_index = i
                    break
            except: 
                print("ERROR: energy not found.")
                sys.exit()
    
        return energy_index


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
        self.primary_rsp = self.primary_rsp.todense() / N[:,None,:,None]
         
        return
    
    def make_edisp_matrix(self, theta_bin, comp="total"):

        """Make energy dispersion matrix.
        
        Parameters
        ----------
        theta_bin : int 
            index of incident angle.
        comp : str, optional
            Component to use for constructing matrix. Either 
            transmitted, scattered, or total (default is total). 
        """
    
        # Project onto em, ei:
        if comp == "transmitted":
            self.edisp_array = self.primary_rsp.slice[{"theta_i [deg]":theta_bin, 
                "theta_m [deg]":theta_bin}].project(["Em [keV]", "Ei [keV]"]).contents
            self.edisp_array = np.array(self.edisp_array) 

        if comp == "scattered":
            
            # Need to skip 90 degree bin, b/c it's undefined from scaling factor:
            deg90_bin = self.get_theta_bin(90)

            # Sum over all measured angles except source incident angle and 90 degrees:
            counter = 0
            for i in range(0,len(self.incident_ang_centers)):
                if (i != theta_bin) & (i < deg90_bin):
                    if counter == 0:
                        this_edisp_array = self.primary_rsp.slice[{"theta_i [deg]":theta_bin, 
                            "theta_m [deg]":i}].project(["Em [keV]", "Ei [keV]"]).contents
                        self.edisp_array = np.array(this_edisp_array) 
                        counter += 1
                    if counter != 0:
                        this_edisp_array = self.primary_rsp.slice[{"theta_i [deg]":theta_bin, 
                            "theta_m [deg]":i}].project(["Em [keV]", "Ei [keV]"]).contents
                        self.edisp_array += np.array(this_edisp_array) 

        if comp == "total":
            self.edisp_array = self.normed_edisp_array_beam + self.normed_edisp_array_scattered
        
        # Transmission probability:
        self.tp = self.edisp_array.sum(axis=0)
        
        return self.edisp_array, self.tp
