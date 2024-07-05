# Imports
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from histpy import Histogram
import gc 
from astropy.convolution import convolve, Gaussian2DKernel
import gzip

class Simulate:
 
    def run_sim(self, source_file, seed=0, verbosity=1):

        """Run atmospheric sims.
        
        Parameters
        ----------
        source_file : str
            Cosima source file.
        seed : int, optional
            Option to pass seed to cosima (default is no seed). 
        verbosity : int, optional
            Output level of cosima (default is 1).
        """

        if seed != 0:
            os.system("cosima -v %s -s %s %s" %(verbosity,seed,source_file))

        if seed == 0:
            os.system("cosima -v %s %s" %(verbosity,source_file))

        return

    def parse_sim_file(self, sim_file, unique=True):

        """Parse sim file with StoreSimulationInfo = init-only. 

        Parameters
        ----------
        sim_file : str 
            Cosima sim file. 
        unique : bool 
            A photon may cross the detecting volume numerous times.
            To only count the first pass, set unique=True (defualt). 
            If false, will count all passes. 
        """

        # initiate lists:
        id_list_all = []
        id_list = []
        ei_list = []
        em_list = []
        xi = []
        yi = []
        zi = []
        xm = []
        ym = []
        zm = []
        xdi = []
        ydi = []
        zdi = []
        xdm = []
        ydm = []
        zdm = []

        if sim_file.endswith(".gz") :
            f = gzip.open(sim_file.strip(),"rt")
        
        else: 
            f = open(sim_file,"r")

        i = 0
        while True:
     
            this_line = f.readline().strip().split()
            i = i + 1

            if this_line:
        
                if "ID" in this_line:
            
                    this_id = int(this_line[1])
                    time_line = f.readline().strip().split()
                    this_time = float(time_line[1])

                    init_line = f.readline().strip().split(";")
            
                    # Save initial info of all thrown photons:
                    id_list_all.append(int(this_id))
                    ei_list.append(float(init_line[22]))
                    xi.append(float(init_line[4]))
                    yi.append(float(init_line[5]))
                    zi.append(float(init_line[6]))
                    xdi.append(float(init_line[16]))
                    ydi.append(float(init_line[17]))
                    zdi.append(float(init_line[18]))

                    # Get info for events that pass watched volume:
                    get_events = True
                    while get_events:

                        next_line = f.readline().strip().split(";")
               
                        if "IA ENTR" in next_line[0]:
                
                            id_list.append(int(this_id))
                            em_list.append(float(next_line[14]))
                            xm.append(float(next_line[4]))
                            ym.append(float(next_line[5]))
                            zm.append(float(next_line[6]))
                            xdm.append(float(next_line[8]))
                            ydm.append(float(next_line[9]))
                            zdm.append(float(next_line[10]))

                            # Option to only consider first pass for each photon:
                            if unique == True:
                                get_events = False

                        # Break if next photon event is reached:
                        if "SE" in next_line[0]:
                            get_events = False
            
                        # Need to break at end of file:
                        if "TS" in next_line[0]:
                            break

            # if line is empty end of file is reached
            if not this_line: 
                if i > 100:
                    break

        # Write data for all thrown events:
        d = {"id":id_list_all,"ei[keV]":ei_list,"xi[cm]":xi,"yi[cm]":yi,"zi[cm]":zi,\
                "xdi[cm]":xdi, "ydi[cm]":ydi, "zdi[cm]":zdi}
        df = pd.DataFrame(data=d)
        df.to_csv("all_thrown_events.dat",float_format='%10.9e',index=False,sep="\t",\
                columns=["id","ei[keV]","xi[cm]","yi[cm]","zi[cm]",\
                "xdi[cm]","ydi[cm]","zdi[cm]"])

        # Write event list for photons passing watched volume:
        d = {"id":id_list, "em[keV]":em_list, \
                "xm[cm]":xm, "ym[cm]": ym, "zm[cm]":zm,\
                "xdm[cm]":xdm, "ydm[cm]":ydm, "zdm[cm]": zdm}
        df = pd.DataFrame(data=d)
        df.to_csv("event_list.dat",float_format='%10.9e',index=False,sep="\t",\
                columns=["id","em[keV]","xm[cm]","ym[cm]","zm[cm]",
                    "xdm[cm]","ydm[cm]","zdm[cm]"])

        return

    def parse_sim_file_all_info(self, sim_file, unique=True):

        """Parse sim file with StoreSimulationInfo = all. This also 
        returns the sequence of interactions for each measured photon. 

        Parameters
        ----------
        sim_file : str 
            Cosima sim file. 
        unique : bool 
            A photon may cross the detecting volume numerous times.
            To only count the first pass, set unique=True (defualt). 
            If false, will count all passes. 
        
        Note
        ----
        This method can be used if you are interested in the sequence 
        of interactions for each of the photons. It comes at the cost
        of a larger sim file, since everything needs to be recorded. The
        sequence of interactions is stored as a string in the parsed event 
        file, encoded as follows:

        S: start
        C: Compton event
        B: Bremsstrahlung event
        A: Photo electric absorption
        R: Rayliegh scattering 
        P: Pair production
        X: Pair annihilation
        I: Entered watched volume

        In the default mode, we only consider the first time a photon
        crosses the watched volume. If considering multiple crossings, 
        the photon properties are recorded separately for each crossing.
        This includes the event sequence, which gives all interactions
        leading up to the time the photon entered the watched volume. 
        
        For some interactions such as PAIR creation, two IA entries are 
        generated, representing the electron and the positron. Thus, the
        sequence "PP" corresponds to one pair event. 
        """

        # initiate lists:
        id_list_all = []
        id_list = []
        ei_list = []
        em_list = []
        xi = []
        yi = []
        zi = []
        xm = []
        ym = []
        zm = []
        xdi = []
        ydi = []
        zdi = []
        xdm = []
        ydm = []
        zdm = []
        seq = []

        if sim_file.endswith(".gz") :
            f = gzip.open(sim_file,"rt")
        
        else: 
            f = open(sim_file,"r")

        i = 0
        while True:
     
            this_line = f.readline().strip().split()
            i = i + 1

            if this_line:
        
                if "ID" in this_line:
            
                    this_id = int(this_line[1])

                    # Get info for events that pass watched volume:
                    get_events = True
                    while get_events:

                        next_line = f.readline().strip().split(";")
               
                        if "IA INIT" in next_line[0]:

                            # Save initial info of all thrown photons:
                            id_list_all.append(int(this_id))
                            ei_list.append(float(next_line[22]))
                            xi.append(float(next_line[4]))
                            yi.append(float(next_line[5]))
                            zi.append(float(next_line[6]))
                            xdi.append(float(next_line[16]))
                            ydi.append(float(next_line[17]))
                            zdi.append(float(next_line[18]))
                
                            # Reset sequence string for this photon:
                            this_seq = "S" # "Start"

                        # Record each interaction for each photon:
                        if "IA COMP" in next_line[0]:
                            this_seq += "C"
                        if "IA BREM" in next_line[0]:
                            this_seq += "B"
                        if "IA PHOT" in next_line[0]:
                            this_seq += "A"
                        if "IA RAYL" in next_line[0]:
                            this_seq += "R"
                        if "IA PAIR" in next_line[0]:
                            this_seq += "P"
                        if "IA ANNI" in next_line[0]:
                            this_seq += "X"

                        if "IA ENTR" in next_line[0]:
               
                            this_seq += "I"
                            id_list.append(int(this_id))
                            em_list.append(float(next_line[14]))
                            xm.append(float(next_line[4]))
                            ym.append(float(next_line[5]))
                            zm.append(float(next_line[6]))
                            xdm.append(float(next_line[8]))
                            ydm.append(float(next_line[9]))
                            zdm.append(float(next_line[10]))
                            seq.append(this_seq)
                            
                            # Reset sequence:
                            this_seq = "S"
                
                            # Option to only consider first pass for each photon:
                            if unique == True:
                                get_events = False

                        # Break if next photon event is reached:
                        if "SE" in next_line[0]:
                            get_events = False
            
                        # Need to break at end of file:
                        if "TS" in next_line[0]:
                            break

            # if line is empty end of file is reached
            if not this_line: 
                if i > 100:
                    break

        # Write data for all thrown events:
        d = {"id":id_list_all,"ei[keV]":ei_list,"xi[cm]":xi,"yi[cm]":yi,"zi[cm]":zi,\
                "xdi[cm]":xdi, "ydi[cm]":ydi, "zdi[cm]":zdi}
        df = pd.DataFrame(data=d)
        df.to_csv("all_thrown_events.dat",float_format='%10.9e',index=False,sep="\t",\
                columns=["id","ei[keV]","xi[cm]","yi[cm]","zi[cm]",\
                "xdi[cm]","ydi[cm]","zdi[cm]"])

        # Write event list for photons passing watched volume:
        d = {"id":id_list, "em[keV]":em_list, \
                "xm[cm]":xm, "ym[cm]": ym, "zm[cm]":zm,\
                "xdm[cm]":xdm, "ydm[cm]":ydm, "zdm[cm]": zdm, "sequence":seq}
        df = pd.DataFrame(data=d)
        df.to_csv("event_list.dat",float_format='%10.9e',index=False,sep="\t",\
                columns=["id","em[keV]","xm[cm]","ym[cm]","zm[cm]",
                    "xdm[cm]","ydm[cm]","zdm[cm]","sequence"])

        return

    def plot_sequence(self, event_file, elow=10, ehigh=10000, num_ebins=30,
                      show_plots=True):

        """Plots ditributions for all photon interactions.

        Parameters
        ----------
        event_file : str
            Name of event file with measured photons. Must be parsed 
            from parse_sim_file_all_info.  
        elow : float, optional
            Lower energy bound in keV (defualt is 100 keV). 
        ehigh : float, optional 
            Upper energy bound in keV (default is 10000 keV.
        num_ebins : int, optional 
            Number of energy bins to use (default is 30). Only log binning for now. 
        show_plots : bool, optional
            Whether or not to show plots (default is True).
        """
        
        # Load data frame:
        df = pd.read_csv(event_file, delim_whitespace=True)
        em = df["em[keV]"]
        seq = np.array(df["sequence"]).astype("str")
        
        # Get total number of interactions:
        n_tot = np.array(df["sequence"].str.len())

        # Get index of scattered photons, which
        # will have more than two interactions:
        scatter_index = n_tot>2
        seq = seq[scatter_index]
        em = em[scatter_index]
        n_tot_scat = n_tot[scatter_index] - 2 
     
        # Get number of interactions for each event type:
        n_c = np.char.count(seq,"C")
        n_b = np.char.count(seq,"B")
        n_a = np.char.count(seq,"A")
        n_r = np.char.count(seq,"R")
        n_p = np.char.count(seq,"P") 
        n_x = np.char.count(seq,"X")

        # Need to divide pair events by 2, since each event has 2 entries:
        n_p = n_p / 2.0
        n_x = n_x / 2.0

        # Define energy bin edges: 
        energy_bin_edges = np.logspace(np.log10(elow), np.log10(ehigh), num_ebins) 

        # Define bin edges for number of interactions:
        count_edges = np.arange(50)
   
        # Define plot dicts:
        tot = {"array":n_tot_scat,"title":"Total"}
        compton = {"array":n_c,"title":"Compton Scattering"}
        brem = {"array":n_b,"title":"Bremsstrahlung"}
        phot = {"array":n_a,"title":"Photo Absorption"}
        rayl = {"array":n_r,"title":"Rayliegh Scattering"}
        pair = {"array":n_p,"title":"Pair Production"}
        anni = {"array":n_x,"title":"Pair Annihilation"}

        plot_list = [tot,compton,brem,phot,rayl,pair,anni]

        # Make plots:
        for each in plot_list:

            # Make hist:
            hist = Histogram([energy_bin_edges,count_edges], labels=["Em [keV]", "interactions"])
            hist.fill(em,each["array"])
        
            # Get normalization factor from total number of 
            # photons in each energy bin:
            if each["title"] == "Total":
                N = hist.project(["Em [keV]"]).contents
                N[N==0] = 1e-50
            
            # Normalize hist:
            hist = hist/N[:,None]
           
            # Plot histogram:
            hist_array = np.array(hist.contents)

            # Smooth image with Gaussian kernel:
            sig_y = 1e-40 # No smoothing along y axis.
            sig_x=0.5
            gauss_kernel = Gaussian2DKernel(sig_y,y_stddev=sig_x)
            filtered_arr = convolve(hist_array, gauss_kernel, boundary='extend')

            # Setup figure:
            fig = plt.figure(figsize=(6.7,8.55))
            gs = fig.add_gridspec(2, hspace=0, height_ratios=[1,3], width_ratios=[1.5])
            axs = gs.subplots(sharex=True, sharey=False)

            # Plot 2d hist:
            img = axs[1].pcolormesh(energy_bin_edges,count_edges, filtered_arr.T, cmap="magma",vmax=0.7)
           
            # Plot max fraction:
            first = np.argmax(filtered_arr,axis=1)
            plt.plot(hist.axes["Em [keV]"].centers,first,color="cyan",lw=2,ls="--",label="max fraction")
            leg = plt.legend(loc=1,frameon=False,fontsize=14)
            for text in leg.get_texts():
                text.set_color("cyan")

            axs[1].set_xscale('log')
            cbar = plt.colorbar(img,fraction=0.045,pad=0.15,location="bottom")
            cbar.set_label("Fraction", fontsize=16)
            cbar.ax.tick_params(labelsize=14)
            plt.xlabel("$\mathrm{E_m}$ [keV]", fontsize=16)
            plt.ylabel("Number of Interactions", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.ylim(0,15)
            
            # Plot projection onto measured energy axis:
            ehist = hist.slice[{"interactions":slice(1,hist.axes["interactions"].nbins)}].project(["Em [keV]"]).contents
            axs[0].stairs(ehist,edges=energy_bin_edges,linewidth=2)
            axs[0].set_xscale('log')
            axs[0].set_title(each["title"], fontsize=16)
            axs[0].set_yticks([0,0.25,0.5,0.75,1.0])
            axs[0].tick_params(axis='y', labelsize=12)
            axs[0].grid(ls=":",color="grey",alpha=0.7,which="both")
            savefile = each["title"].replace(" ","") + ".png"
            plt.tight_layout()
            plt.savefig(savefile)
            if show_plots == True:
                plt.show()
            plt.close()

        return

    def combine_sims(self, file_list, nsim, output_name):

        """Combine event files from multiple simulations.

        Parameters
        ----------
        file_list : list . 
            List of file names to combine. The files must be the output
            from either the parse_sim_file or parse_sim_file_all_info 
            methods. 
        nsim : int 
            Number of simulated photons for each file. Must be the 
            same for all files. 
        output_name : str
            Prefix of output dat file. 
        """

        for i in range(0,len(file_list)):

            print("combining file %s" %str(i))

            this_df = pd.read_csv(file_list[i],delim_whitespace=True)
            
            if i == 0:
                new_df = this_df
                df_columns = this_df.columns.tolist()

            if i > 0:
                this_df["id"] = np.array(this_df["id"]) + i*nsim
                new_df = pd.concat([new_df,this_df],ignore_index=True)
        
            # Remove from memory:
            del this_df
            gc.collect()

        # Write new data frame:
        new_df.to_csv("%s.dat.gz" %output_name,float_format='%10.9e',index=False,sep="\t",\
            columns=df_columns)

        return
