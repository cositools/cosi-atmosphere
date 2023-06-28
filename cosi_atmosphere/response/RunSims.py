# Imports
import pandas as pd
import os

class Simulate:
 
    def run_sim(self, source_file, seed=0):

        """
        Run atmospheric sims.
        
        inputs:
        
        source_file: Cosima source file.
        
        seed[int]: Option to pass seed to cosima. 
        Default is no seed. 
        """

        if seed != 0:
            os.system("cosima -s %s %s" %(seed,source_file))

        if seed == 0:
            os.system("cosima %s" %(source_file))

        return

    def parse_sim_file(self, sim_file, unique=True):

        """
        Parse sim file. 

        Inputs:
        
        sim_file: cosima sim file. 
        
        unique: A photon may cross the detecting volume numerous times.
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

        f = open(sim_file,"r")

        i = 0
        while True:
     
            this_line = f.readline().strip().split()
            i = i + 1
    
            print()
            print("reading line %s" %str(i))
            print()

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
