# Imports
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from  cosi_atmosphere.response import TPCalc

class TPCalculator():

    """Calculates transmission probability for gamma rays.
    
    Mass attenuation coeffients are from NIST XCOM:
    https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html

    Atmospheric data is fom NRLMSIS:
    https://swx-trec.com/msis
    
    A great review of the relationship between transmitted intensity, 
    linear attenuation coefficient, and mass attenuation coefficient
    can be found here:
    https://www.nde-ed.org/Physics/X-Ray/attenuationCoef.xhtml
    """

    def __init__(self):

        # Get data directory:
        path_prefix = os.path.split(TPCalc.__file__)[0]
        self.data_dir = os.path.join(path_prefix,"data_files")
    
        return
        
    def read_atm_model(self, atm_file):

        """Reads in atmospheric profile and makes interpolated function.
        
        Parameters
        ----------
        atm_file : str
            Name of atmosphere file. 
        
        Returns
        -------
        density : scipy:interpolate:interp1d
            Interpolted function of altitude (in cm) 
            vs. density (in g/cm3).
        
        Note
        ----
        Atmosphere file is output from Atmosphere class. 
        """

        # Read in atmospher profile:
        df = pd.read_csv(atm_file,delim_whitespace=True)
        self.alt = df["altitude[km]"] * (1e5) # cm
        self.rho = df["mass_density[kg/m3]"] * (1e-3) # g/cm3

        # Make interpolated function:
        self.density = interp1d(self.alt,self.rho,kind="linear")

        # Other elements for calculating relative abundance:
        self.N2 = df["N2[m-3]"]
        self.O2 = df["O2[m-3]"]
        self.O = df["O[m-3]"]
        self.He = df["He[m-3]"]
        self.H = df["H[m-3]"]
        self.Ar = df["Ar[m-3]"]
        self.N = df["N[m-3]"]
        self.AO = df["anomalous_oxygen[m-3]"]

        return self.density

    def relative_abundance(self):
        
        """Calculates relative abundance of each element of the 
        atmosphere as a function of altitude."""
         
        # Define dictionaries for each component:
        total_density = self.N2 + self.O2 + self.O + self.He + self.H \
                + self.Ar + self.N + self.AO
        Ab_N2 = {"name":"N2","ls":"-","Ab":self.N2/total_density}
        Ab_O2 = {"name":"O2","ls":"--","Ab":self.O2/total_density}
        Ab_O = {"name":"O","ls":":","Ab":self.O/total_density}
        Ab_He = {"name":"He","ls":":","Ab":self.He/total_density}
        Ab_H = {"name":"H","ls":":","Ab":self.H/total_density}
        Ab_Ar = {"name":"Ar","ls":"-.","Ab":self.Ar/total_density}
        Ab_N = {"name":"N","ls":":","Ab":self.N/total_density}
        Ab_AO = {"name":"AO","ls":":","Ab":self.AO/total_density}

        # Plot:
        name_list = [Ab_N2,Ab_O2,Ab_O,Ab_He,Ab_H,Ab_Ar,Ab_N]
        for each in name_list:
            plt.loglog(self.alt/1e5,each["Ab"],label=each["name"],ls=each["ls"])
        plt.xlabel("Altitude [km]",fontsize=12)
        plt.ylabel("Relative Abundance",fontsize=12)
        plt.legend(frameon=False,loc=1,ncol=4)
        plt.ylim(1e-7,100)
        plt.grid(ls=":",alpha=0.2,color="grey")
        plt.show()
        plt.close()

        return
        
    def read_mass_attenuation(self, nist_data="default", show_plot=True):
    
        """Reads in attenuation data from NIST XCOM.
       
        NIST XCOM data can be downloaded from: 
        https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html
    
        Data file must include all column options from XCOM, with 
        column names as follows:
        energy[MeV]
        scat_coh[cm2/g]
        scat_incoh[cm2/g]
        photo_abs[cm2/g]
        pair_prod_NF[cm2/g]
        pair_prod_EF[cm2/g]
        atten_tot_wCS[cm2/g]
        atten_tot_woCS[cm2/g]
        
        Parameters
        ----------
        nist_data : str, optional
            Name of NIST data file. Default file is calculated assuming
            an atmospheric composition of 78% N2 and 22% O2. 
        show_plot : bool
            Option to plot attenuation data (defualt is True).
        """
        
        # Use default file if none is specified:
        if nist_data == "default":
            nist_data = os.path.join(self.data_dir,"mass_attenuation_coefficients_N2-78_O2-22.dat")

        # Read NIST XCOM data:
        df = pd.read_csv(nist_data, delim_whitespace=True)
        self.energy = df["energy[MeV]"]
        self.scat_coh = df["scat_coh[cm2/g]"]
        self.scat_incoh = df["scat_incoh[cm2/g]"]
        self.photo_abs = df["photo_abs[cm2/g]"]
        self.pair_prod_NF = df["pair_prod_NF[cm2/g]"]
        self.pair_prod_EF = df["pair_prod_EF[cm2/g]"]
        self.atten_tot_wCS = df["atten_tot_wCS[cm2/g]"]
        self.atten_tot_woCS = df["atten_tot_woCS[cm2/g]"]

        if show_plot == True:
            self.plot_attenuation()

        return

    def plot_attenuation(self):
    
        """Plots attenuation data from NIST XCOM."""

        fig = plt.figure(figsize=(7,5))

        plt.loglog(self.energy, self.scat_coh,
                label="coherent scattering",ls="--")
        plt.loglog(self.energy, self.scat_incoh,
                label="incoherent scattering",ls="--")
        plt.loglog(self.energy, self.photo_abs,
                label="photo absorption",ls="-.")
        plt.loglog(self.energy, self.pair_prod_NF,
                label="pair production (NF)", ls=":")
        plt.loglog(self.energy, self.pair_prod_EF,
                label="pair production (EF)", ls=":")
        plt.loglog(self.energy, self.atten_tot_wCS, 
                label="Total attenuation (wCS)", color="black", ls="-")

        plt.xlabel("Energy [MeV]", fontsize=12)
        plt.ylabel(r"Mass Attenuation Coefficient [$\mathrm{cm^2 \ g^{-1}}$]", fontsize=12)
        plt.grid(ls=":",alpha=0.2, color="grey")
        plt.legend(frameon=False,loc=1,ncol=1)
        plt.ylim(1e-7,1e4)
        plt.show()
        plt.close()

        return

    def coslaw(self, A, theta, x):

        """Uses law of cosines to determine the altitude for any 
        distance along the line of sight, relative to Earth's surface, 
        for a given altitude of observation and off-axis angle of source. 
       
        Parameters
        ----------
        A : float
            Altitude of observations in cm.
        theta : float
            Off-axis angle of source in degrees. 
        x : float
            distance along line-of-sight in cm.

        Returns
        -------
        r : float
            Side length of triangle in cm (opposite the angle pi - theta). 
        """

        R_E = 6378 * 1e5 # cm
        z = R_E + A
        alpha = np.pi - math.radians(theta)
    
        return np.sqrt(z**2 + x**2 - 2*x*z*np.cos(alpha)) - R_E

    def los_integral(self, A, theta, dx=1e4, alt_max=200e5):
    
        """Estimates line of sight integral with respect to radius of
        spherical reference frame.
    
        Parameters
        ----------
        A : float
            Altitude of observations in cm.
        theta : float
            Off-axis angle of source in degrees.
        dx : float, optional
            x step size in cm (default is 1e4).
        alt_max : float, optional 
            Top of atmosphere in cm (default is 200e5 cm).

        Returns
        -------
        integral : float
            Integral along line of sight, which as units of g/cm2
        """
    
        x = -1*dx
        integral = 0
        r = 0
        while r<(alt_max-dx):
            x += dx
            r=self.coslaw(A,theta,x)
            integral += self.density(r)*dx

        return integral

    def calc_tp(self, A, theta, output_name=None, dx=1e4, alt_max=200, show_plot=True):

        """Calculates transmission probability.
        
        Parameters
        ----------
        A : float
            Altitude of observations in km.
        theta : float
            Off-axis angle of source in degrees.
        output_name : str, optional
            Name of output file (default is None, in which case no 
            output file is written).
        dx : float, optional
            x step size in cm (default is 1e4, i.e. 100 m).
        alt_max : float, optional 
            Top of atmosphere in km (default is 200).
        show_plot : bool, optional
            Option to plot transmission probability (default is True).

        Returns
        -------
        tp : array
            Transmission probability as a function of altitude. 
        """
       
        # Altitudes need to be in cm for calculation:
        A *= 1e5 
        alt_max *= 1e5
        
        los_integral = self.los_integral(A, theta, dx=dx, alt_max=alt_max)
        self.tp = np.exp(-1*self.atten_tot_wCS*los_integral)
       
        # Write to file:
        if output_name != None:
            d = {"energy[MeV]":self.energy, "TP":self.tp}
            df = pd.DataFrame(data = d, columns = ["energy[MeV]","TP"])
            df.to_csv(output_name,sep="\t",index=False)
        
        # Plot:
        if show_plot == True:
            self.plot_tp()

        return self.tp

    def plot_tp(self):

        """Plots transmission probability."""

        plt.semilogx(self.energy,self.tp)
    
        plt.xlabel("Energy [MeV]", fontsize=12)
        plt.ylabel("Transmission Probability", fontsize=12)
        plt.ylim(0,1)
        plt.xlim(0.01,10)
        plt.grid(ls=":",alpha=0.2, color="grey")
        plt.show()
        plt.close()

        return

    def cosima_tp_file(self, A, output_name, theta_list="default", \
            energy_list="default", show_plot=True):

        """Makes input transmission probability file for MEGAlib's cosima.
        
        Parameters
        ----------
        A : float
            Altitude of observations in km.
        output_name : str
            Name output files. Output will be saved as both .npy file 
            as well as .dat file for cosima input.
        theta_list : list, optional
            List of fff-axis angles to include in degrees. Default is 0-90 degrees
            with 5 degree spacing. TP is set to zero beyond 90 degrees.
        energy_list : list, optional
            List of energy values to use in keV. Default is 50 keV - 10 MeV,
            using log spacing. 
        show_plot : bool, optional
            Option to plot 2D TP array. 
        """

        # Make sure atmosphere and attenuation has been loaded:
        try: 
            self.alt
        except:
            print("ERROR: Must specify atmosphere model.")
            sys.exit()

        try:
            self.energy
        except:
            print("ERROR: Must specify attenuation coefficients.")
            sys.exit()

        # Calculate theta max for given altitude:
        R_E = 6378 # km
        arg = R_E / (R_E + A)
        theta_max = math.pi - math.asin(arg)
        print("Max theta for given altitude due to Earth occultation [deg]: %s" %str(math.degrees(theta_max)))

        # Define theta list:
        if theta_list == "default":
            theta_list = np.arange(0,95,5).tolist()
        
        # Define energy list:
        if energy_list == "default":
            energy_list = np.logspace(np.log10(50),4).tolist()

        # Calc TP for each theta:
        tp_list = []
        for each in theta_list:
            this_tp = self.calc_tp(A,each,show_plot=False)
            tp_func = interp1d(np.array(self.energy)*1000,this_tp,kind="linear")
            tp_list.append(tp_func(energy_list))

        # Append zero values:
        theta_list.append(90.001)
        theta_list.append(180)
        tp_list.append([0]*len(energy_list))
        tp_list.append([0]*len(energy_list))

        tp_list = np.array(tp_list)
        
        # Save output to numpy array:
        np.save(output_name,tp_list)

        # Save output for cosima:
        
        # Open file for writing:
        f = open("%s.dat" %output_name, "w")
        f.write("IP LIN\n\n")
        
        # Theta string:
        f.write("# Theta axis in degrees:\n")
        theta_str = "XA "
        for each in theta_list:
            theta_str += str(each) + " "
        f.write(theta_str + "\n")
        
        # Energy string:
        f.write("# Energy axis in keV:\n")
        energy_str = "YA "
        for each in energy_list:
            energy_str += str(each) + " "
        f.write(energy_str + "\n\n")

        for i in range(0,len(theta_list)):
            for j in range(0,len(energy_list)):
                f.write("AP %s %s %s\n" %(i,j,tp_list[i,j]))
        f.write("EN")
        f.close()

        # Plot:
        fig = plt.figure(figsize=(8,8))
        ax = plt.gca()

        img = ax.pcolormesh(energy_list,theta_list,tp_list,cmap="viridis",vmin=0,vmax=1)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        cbar = plt.colorbar(img,fraction=0.045)
        cbar.set_label("Probability",size=16,labelpad=12)
        cbar.ax.tick_params(labelsize=12)

        plt.ylabel('Zenith Angle [$\circ$]',fontsize=18)
        plt.xlabel('Energy [keV]',fontsize=18)
        plt.title('Transmission Probability (%s km)' %str(A), fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        ax.tick_params(axis='both',which='major',length=9)
        ax.tick_params(axis='both',which='minor',length=5)

        plt.xscale("log")
        plt.savefig("%s.png" %output_name,bbox_inches='tight')
        if show_plot==True:
            plt.show()
        plt.close()

        return
