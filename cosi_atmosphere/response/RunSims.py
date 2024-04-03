# Imports
import pandas as pd
import os
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

    def parse_sim_file(self, sim_file,alt,unique=True,nbSimFile=1,nbSimEventperFile=10e4):

        """Parse sim file. 

        Parameters
        ----------
        sim_file : str 
            Cosima sim file or list of Cosima sim files 
        unique : bool 
            A photon may cross the detecting volume numerous times.
            To only count the first pass, set unique=True (default). 
            If false, will count all passes.
        nbSimEventperFile : int
            number of events simulated per files if there are more than one file. default is 10e4 (1000 files).     
	
        nbSimFile : int
           number of sim files. default is 1.
        alt : float
           altitude used for the mass model 
    
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

        
 
        #get all the file names
        listfile = []
        if nbSimFile > 1 :
            with open(sim_file,"r") as file :
                listfile = file.readlines()
        else :
            listfile.append(sim_file)
             
       
        for nb in range(nbSimFile):     
            
            filename = listfile[nb]
            print(f"reading {filename.strip()}")           

            #check if it is compressed file or not            
            if filename.strip().endswith(".gz") :
                f = gzip.open(filename.strip(),"r")

            else :
                f = open(filename.strip(),"r")

            
            #loop on the lines of the file
            for lines in f :
     
                this_line = lines.decode("utf-8").split()
                init_line = lines.decode("utf-8").split(";")

                #skip empty line
                if len(this_line)==0:
                    continue
                
                
                #print(this_line)
                #print(init_line)

        
                if this_line[0]=="ID" :
            
                    this_id = int(this_line[1])+nb*nbSimEventperFile
                        

                elif "IA INIT" in init_line[0]:
                    # Save initial info of all thrown photons:
                    id_list_all.append(int(this_id)+nb*nbSimEventperFile)
                    ei_list.append(float(init_line[22]))
                    xi.append(float(init_line[4]))
                    yi.append(float(init_line[5]))
                    zi.append(float(init_line[6]))
                    xdi.append(float(init_line[16]))
                    ydi.append(float(init_line[17]))
                    zdi.append(float(init_line[18]))

                    # Get info for events that pass watched volume:
                    get_events = True
                    
                

               
                elif "IA ENTR" in init_line[0] and get_events:
                
                        id_list.append(int(this_id))
                        em_list.append(float(init_line[14]))
                        xm.append(float(init_line[4]))
                        ym.append(float(init_line[5]))
                        zm.append(float(init_line[6]))
                        xdm.append(float(init_line[8]))
                        ydm.append(float(init_line[9]))
                        zdm.append(float(init_line[10]))

                        # Option to only consider first pass for each photon:
                        if unique == True:
                            get_events = False

                #Break if next photon event is reached:
                elif this_line[0]=="SE":
                    get_events = False
            
            f.close()        

        # Write data for all thrown events:
        d = {"id":id_list_all,"ei[keV]":ei_list,"xi[cm]":xi,"yi[cm]":yi,"zi[cm]":zi,\
                "xdi[cm]":xdi, "ydi[cm]":ydi, "zdi[cm]":zdi}
        df = pd.DataFrame(data=d)
        df.to_csv(f"all_thrown_events_{alt}km.dat",float_format='%10.9e',index=False,sep="\t",\
                columns=["id","ei[keV]","xi[cm]","yi[cm]","zi[cm]",\
                "xdi[cm]","ydi[cm]","zdi[cm]"])

        # Write event list for photons passing watched volume:
        d = {"id":id_list, "em[keV]":em_list, \
                "xm[cm]":xm, "ym[cm]": ym, "zm[cm]":zm,\
                "xdm[cm]":xdm, "ydm[cm]":ydm, "zdm[cm]": zdm}
        df = pd.DataFrame(data=d)
        df.to_csv(f"event_list_{alt}km.dat",float_format='%10.9e',index=False,sep="\t",\
                columns=["id","em[keV]","xm[cm]","ym[cm]","zm[cm]",
                    "xdm[cm]","ydm[cm]","zdm[cm]"])

        return
