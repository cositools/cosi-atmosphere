# Imports:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from atmospheric_gammas.response import MassModels
import os,sys 

class MakeMassModels:

    def __init__(self, atmosphere_file, kwargs={}):

        """
        Generates a mass model based on input atmospheric data.

        Inputs:
        
        atmosphere_file: input file describing the atmosphere, 
        calculated with Atmospheric_profile class, based on NRLMSIS. 
        
        kwargs: pass any kwargs to pandas read_csv method. 
        """
        
        # Get test directory:
        path_prefix = os.path.split(MassModels.__file__)[0]
        self.test_dir = os.path.join(path_prefix,"test_files")
        
        df = pd.read_csv(atmosphere_file, delim_whitespace=True, **kwargs)

        self.height = np.array(df["altitude[km]"])
        self.density = np.array(df["mass_density[kg/m3]"]) * (1e-3) # g/cm3 (required for geomega) 
        self.H = np.array(df["H[m-3]"]) 
        self.He = np.array(df["He[m-3]"])
        self.N = np.array(df["N[m-3]"]) + 2*np.array(df["N2[m-3]"]) 
        self.O = np.array(df["O[m-3]"]) + 2*np.array(df["O2[m-3]"])
        self.Ar = np.array(df["Ar[m-3]"]) 

        # Normalize so that we can convert to ints (as required by geomega):
        # Also scale by 1e6 in order to convert to int.
        # Note: converting to int actually uses the floor. 
        # Use this to compare to AZ, but improve later. 
        atm_tot = self.H + self.He + self.N + self.O + self.Ar
        self.H_normed = (1e6*(self.H/atm_tot)).astype("int") 
        self.He_normed = (1e6*(self.He/atm_tot)).astype("int")
        self.N_normed = (1e6*(self.N/atm_tot)).astype("int")
        self.O_normed = (1e6*(self.O/atm_tot)).astype("int")
        self.Ar_normed = (1e6*(self.Ar/atm_tot)).astype("int")

        return 
   
    def get_cart_vectors(self, angle, altitude):
        
        """
        Get x position and direction vectors for off-axis beam.
        
        Inputs:
        angle: incident angle of source in degrees. 
        altitude: altitude of detecting plane in km.
        """

        conv = math.pi/180.0
        angle = angle * conv # radians

        y = 200 - altitude # km 
        x = y*math.tan(angle)  # km
        x = x * 1e5 # cm
        nx = -1*math.sin(angle)
        nz = -1*math.cos(angle)

        print()
        print("x [cm]: " + str(x))
        print("nx [cm]: " + str(nx))
        print("nz [cm]: " + str(nz))
        print()

        return

    def plot_atmosphere(self):

        """
        Plot atmosphere model. Also compare normalized atmospheric 
        components to original calculation from AZ.
        """

        # plot elements:
        fig,ax1 = plt.subplots(figsize=(9,6))
        ax2 = ax1.twinx()
        ax1.loglog(self.height,self.H,label="H")
        ax1.loglog(self.height,self.He,label="He")
        ax1.loglog(self.height,self.N,label="N (N + N2)")
        ax1.loglog(self.height,self.O,label="O (O + O2)")
        ax1.loglog(self.height,self.Ar,label="Ar")
        ax1.set_ylabel("Number Density [$\mathrm{m^{-3}}$]",fontsize=12)
        ax1.set_ylim(ymin=1e11,ymax=1e26)
        ax1.set_xlabel("Altitude [km]", fontsize=12)
        ax1.legend(loc=3,frameon=False)
        ax1.grid(ls=":",color="grey")
        ax1.tick_params(axis="both",labelsize=12)

        ax2.loglog(self.height,self.density,ls="--",lw=3,color="black",label="mass density")
        ax2.set_ylabel("Mass Density [$\mathrm{g \ cm^{-3}}$]",fontsize=12)
        ax2.legend(loc=1,frameon=False)
        ax2.tick_params(axis="y",labelsize=12)

        ax1.set_xlabel("Altitude [km]", fontsize=12)
        plt.savefig("particle_profile.pdf")
        plt.show()
        plt.close()

        # plot elements normed:
        fig,ax1 = plt.subplots(figsize=(9,6))
        ax1.loglog(self.height,self.He_normed,label="He")
        ax1.loglog(self.height,self.N_normed,label="N (N + N2)")
        ax1.loglog(self.height,self.O_normed,label="O (O + O2)")
        ax1.loglog(self.height,self.Ar_normed,label="Ar")
        ax1.set_ylabel("ComponentsByAtoms (normalized)",fontsize=12)
        ax1.set_xlabel("Altitude [km]", fontsize=12)
        ax1.legend(loc=3,frameon=False)
        ax1.grid(ls=":",color="grey")
        ax1.tick_params(axis="both",labelsize=12)

        # For comparing to AZ original,
        # from MEGAlib/resource/examples/advanced/Atmosphere:
        geo_original = os.path.join(self.test_dir,"Atmosphere.BestCrabDay.geo")
        g = open(geo_original,"r")
        H_list = []
        He_list = []
        N_list = []
        O_list = []
        Ar_list = []
        height_list = np.arange(0,200,1)
        lines = g.readlines()
        for line in lines:
            split = line.split()
            if "H" in split:
                H_list.append(int(split[2]))
            if "He" in split:
                He_list.append(int(split[2]))
            if "N" in split:
                N_list.append(int(split[2]))
            if "O" in split:
                O_list.append(int(split[2]))
            if "Ar" in split:
                Ar_list.append(int(split[2]))
        
        plt.loglog(height_list,He_list,ls="",marker="o",zorder=0)
        plt.loglog(height_list,N_list,ls="",marker="o",zorder=0)
        plt.loglog(height_list,O_list,ls="",marker="o",zorder=0)
        plt.loglog(height_list,Ar_list,ls="",marker="o",zorder=0)

        ax1.set_xlabel("Altitude [km]", fontsize=12)
        plt.savefig("particle_profile_normed.pdf")
        plt.show()
        plt.close()

        return

    def rectangular_model(self,watch_height):
        
        """
        Original rectangular mass model by AZ, 
        from MEGAlib/resource/examples/advanced/Atmosphere. 
        This is also similar to what was done by Alex Lowell. 
        The model consists of recangular atmoshperic slabs. 
        There is a very tiny surrounding sphere placed at the
        top of the atmoshpere, although this is not used for a 
        unifrom beam source. The watched volume is the entire 
        slab starting at the specified watch_height. The source 
        is a narrow beam with radius = 1cm. 

        Inputs:
        watch_height: altitude for watched volume in km. 
            The actual volume will be a box starting at the 
            specified value, with a height equal to the spacing
            specified in the atmospheric model. 
        """

        # Get half-height from atmospheric profile.
        # Note: Atmospheric profile must start from 0 km. 
        max_height = np.amax(self.height)
        n_values = len(self.height) - 1 # -1 since altitude starts at zero
        half_height = 0.5 * (max_height/n_values) * 1e5 # cm 

        # Get index for watched volume:
        watch_index = np.argwhere(self.height==watch_height)[0]
        watch_index = int(watch_index)

        print("Using half-height [cm]: " + str(half_height))
        print("Watch index: " + str(watch_index))

        # Make mass model:
        f = open("atmosphere.geo","w")

        # General:
        f.write("# Atmosphere model\n\n")
        f.write("Name AtmoshpereModel\n\n")

        # Surrounding sphere:
        f.write("# Surrounding sphere:\n")
        f.write("SurroundingSphere 0.1 0 0 20000000.0 0.1\n\n")

        # World volume:
        f.write("Volume World\n")
        f.write("World.Material Vacuum\n")
        f.write("World.Shape BOX 10240000000.000000 10240000000.000000 10240000000.000000\n")
        f.write("World.Visibility 1\n")
        f.write("World.Position 0 0 0\n")
        f.write("World.Mother 0\n\n")

        # Materials:
        f.write("Include $(MEGALIB)/resource/examples/geomega/materials/Materials.geo\n\n")

        # write atmoshpere slices:
        for i in range(0,len(self.H_normed)-1):

            # Material:
            # Note: using density at i-1: 
            # This should be fine as long as the atmospheric model
            # has small altitude bins.
            f.write("Material MaterialSlice_%s_%s\n" %(str(i),str(i+1)))
            f.write("MaterialSlice_%s_%s.Density %s\n" %(str(i),str(i+1),str(self.density[i])))
            if self.H_normed[i] != 0:
                f.write("MaterialSlice_%s_%s.ComponentByAtoms H %s\n" %(str(i),str(i+1),str(self.H_normed[i])))
            f.write("MaterialSlice_%s_%s.ComponentByAtoms He %s\n" %(str(i),str(i+1),str(self.He_normed[i])))
            f.write("MaterialSlice_%s_%s.ComponentByAtoms N %s\n" %(str(i),str(i+1),str(self.N_normed[i])))
            f.write("MaterialSlice_%s_%s.ComponentByAtoms O %s\n" %(str(i),str(i+1),str(self.O_normed[i])))
            f.write("MaterialSlice_%s_%s.ComponentByAtoms Ar %s\n\n" %(str(i),str(i+1),str(self.Ar_normed[i])))

            # Volume (Box):
            z_slab = 2*(i)*half_height + half_height
            f.write("Volume VolumeSlice_%s_%s\n" %(str(i),str(i+1)))
            f.write("VolumeSlice_%s_%s.Material MaterialSlice_%s_%s\n" %(str(i),str(i+1),str(i),str(i+1)))
            f.write("VolumeSlice_%s_%s.Shape BOX 51200000.000000 51200000.000000 %s\n" %(str(i),str(i+1),str(half_height)))
            f.write("VolumeSlice_%s_%s.Visibility  1\n" %(str(i),str(i+1)))
            f.write("VolumeSlice_%s_%s.Position 0 0 %s\n" %(str(i),str(i+1),str(z_slab)))
            f.write("VolumeSlice_%s_%s.Mother World\n\n" %(str(i),str(i+1)))

        # Write TestVolume:
        f.write("Volume TestVolume\n")
        f.write("TestVolume.Material MaterialSlice_%s_%s\n" %(str(watch_index),str(watch_index+1)))
        f.write("TestVolume.Shape BOX 51200000.000000 51200000.000000 %s\n" %str(half_height))
        f.write("TestVolume.Visibility 1\n")
        f.write("TestVolume.Position 0 0 0\n")
        f.write("TestVolume.Mother VolumeSlice_%s_%s\n" %(str(watch_index),str(watch_index+1)))
         
        f.close()

        return

    def spherical_model(self, watch_alt):

        """
        Model consists of concentric spherical shells, 
        enclosed by a large surrounding sphere. 
        
        Inputs:
        watch_alt: altitude for watched volume in km. 
            The actual volume will be a shell with the inner radius
            starting at the specified value, and a height equal to the spacing
            specified in the atmospheric model. 
        """

        # Make mass model:
        f = open("atmosphere.geo","w")

        # General:
        f.write("# Atmosphere model\n\n")
        f.write("Name AtmoshpereModel\n\n")

        # Surrounding sphere:
        # Radius set to 200 km + r_earth
        f.write("# Surrounding sphere:\n")
        f.write("SurroundingSphere 657800000.0 0 0 0 657800000.0\n")
        f.write("ShowSurroundingSphere true\n\n")

        # World volume:
        f.write("Volume World\n")
        f.write("World.Material Vacuum\n")
        f.write("World.Shape BOX 10240000000.000000 10240000000.000000 10240000000.000000\n")
        f.write("World.Visibility 1\n")
        f.write("World.Position 0 0 0\n")
        f.write("World.Mother 0\n\n")

        # Materials:
        f.write("Include $(MEGALIB)/resource/examples/geomega/materials/Materials.geo\n\n")

        # Get shell thickness from atmospheric profile.
        # Note: Atmospheric profile must start from 0 km. 
        max_height = np.amax(self.height)
        n_values = len(self.height) - 1 # -1 since altitude starts at zero
        shell_thickness = (max_height/n_values) * 1e5 # cm 

        # Make sure watched altitude is in height list:
        if np.isin(watch_alt,self.height) == False:
            print()
            print("ERROR: Watched altitude must be defined in atmospheric model!")
            print()
            sys.exit()
            
        # Get index for watched volume:
        watch_index = np.argwhere(self.height==watch_alt)[0]
        watch_index = int(watch_index)

        # Must define curvature relative to Earth's surface [cm]:
        r_earth = 6.378e8 

        # For watched area:
        watch_inner = r_earth + (watch_alt * 1e5) # cm
        watch_outer = watch_inner + shell_thickness

        print("Using shell thickness [cm]: " + str(shell_thickness))
        print("Watch index: " + str(watch_index))

        # write atmoshpere shells:
        for i in range(0,len(self.H_normed)-1):

            # Define inner and outer radius of shells.
            r1 = r_earth + shell_thickness*i # inner radius [cm]
            r2 = r1 + shell_thickness # outer radius [cm]

            # Material:
            # Note: using density at i, which should be ok
            # as long as the shell thickness is small.
            f.write("Material MaterialSlice_%s_%s\n" %(str(i),str(i+1)))
            f.write("MaterialSlice_%s_%s.Density %s\n" %(str(i),str(i+1),str(self.density[i])))
            if self.H_normed[i] != 0:
                f.write("MaterialSlice_%s_%s.ComponentByAtoms H %s\n" %(str(i),str(i+1),str(self.H_normed[i])))
            f.write("MaterialSlice_%s_%s.ComponentByAtoms He %s\n" %(str(i),str(i+1),str(self.He_normed[i])))
            f.write("MaterialSlice_%s_%s.ComponentByAtoms N %s\n" %(str(i),str(i+1),str(self.N_normed[i])))
            f.write("MaterialSlice_%s_%s.ComponentByAtoms O %s\n" %(str(i),str(i+1),str(self.O_normed[i])))
            f.write("MaterialSlice_%s_%s.ComponentByAtoms Ar %s\n\n" %(str(i),str(i+1),str(self.Ar_normed[i])))

            # Volume (Box):
            this_int = random.randint(0,10) # set random color per shell for visualization
            f.write("Volume VolumeSlice_%s_%s\n" %(str(i),str(i+1)))
            f.write("VolumeSlice_%s_%s.Material MaterialSlice_%s_%s\n" %(str(i),str(i+1),str(i),str(i+1)))
            f.write("VolumeSlice_%s_%s.Shape SPHERE %s %s\n" %(str(i),str(i+1),str(r1),str(r2)))
            f.write("VolumeSlice_%s_%s.Visibility  0\n" %(str(i),str(i+1)))
            f.write("VolumeSlice_%s_%s.Position 0 0 0\n" %(str(i),str(i+1)))
            f.write("VolumeSlice_%s_%s.Color %s\n" %(str(i),str(i+1),str(this_int)))
            f.write("VolumeSlice_%s_%s.Mother World\n\n" %(str(i),str(i+1)))

        # Write TestSphere:
        f.write("Volume TestSphere\n")
        f.write("TestSphere.Material MaterialSlice_%s_%s\n" %(str(watch_index),str(watch_index+1)))
        f.write("TestSphere.Shape Sphere %s %s\n" %(str(watch_inner),str(watch_outer)))
        f.write("TestSphere.Visibility 1\n")
        f.write("TestSphere.Position 0 0 0\n")
        f.write("TestSphere.Color 2\n")
        f.write("TestSphere.Mother VolumeSlice_%s_%s\n" %(str(watch_index),str(watch_index+1)))
    
        f.close()

        return

