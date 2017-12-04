###################################################################
###                       import section                        ###
###################################################################

from pathlib import Path
from getting_Best_indexes_from_graph import getting_best_indexes_from_graph
from PlotGraphFromCSV import plot
import os
import glob


# |-----------------------------------------------------------------------------------------------------------------|
#                                                                                                                   |
#                       Function Definition                                                                         |
#                                                                                                                   |
# |-----------------------------------------------------------------------------------------------------------------|

def finding_src_dst(source_path, graph=0, optimized_index=0):

    """The function takes source path like  C:/Users/User/Google Drive/DATA/850nm_new and gets the source path of csv folders
      and directory path where the graphs will be plotted automatically. The function also gets two arguments graph and optimized_index
      if graph variable is 1 then it creates Graphs for each row according to csv files and if optimized_index variable is 1 then
      it returns the best two indexes of graph """

    source_path = Path(source_path) # source path contain the relative path

    dir = [x for x in source_path.iterdir() if x.is_dir()] # the dir varible contains all the subdirectory of the source folder
                                                               # dir path = WindowsPath('C:/Users/User/Google Drive/DATA/850nm_new/fingertip_vedio_with_mobile_flash_but_without_led_flash')

    for sub_dir in dir:            # sub_dir contains a value from list dir. subdir =  WindowsPath('C:/Users/User/Google Drive/DATA/850nm_new/fingertip_vedio_with_mobile_flash_but_without_led_flash')
        sub_sub_dir = [x for x in sub_dir.iterdir() if x.is_dir()]     # sub_sub_dir contains the whole csv files path
        if len(sub_sub_dir)!=0:    # For filtering out the directories containing only csv files
            for csv_dir in sub_sub_dir:     # Accessing Each csv folder like 101_Dulia_Begum_F55_GL=24.5
                for file in csv_dir.glob('*_blue.csv'):  # For Accessing blue.csv files
                    if graph == 1:
                        plot(file, os.path.join(csv_dir,"blue"))
                    if optimized_index == 1:
                        getting_best_indexes_from_graph(file,csv_dir,blue=1)
                for file in csv_dir.glob('*_red.csv'):   # For Accessing blue.csv files
                    if graph == 1:
                        plot(file, os.path.join(csv_dir, "red"))
                    if optimized_index == 1:
                        getting_best_indexes_from_graph(file, csv_dir,red=1)
                for file in csv_dir.glob('*_green.csv'):    # For Accessing blue.csv files
                    if graph == 1:
                        plot(file, os.path.join(csv_dir, "green"))
                    if optimized_index == 1:
                        getting_best_indexes_from_graph(file, csv_dir,green=1)



source_path = "C:/Users/User/Google Drive/DATA/850nm_new"
finding_src_dst(source_path,graph=1,optimized_index=0)