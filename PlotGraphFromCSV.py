###################################################################
###                       import section                        ###
###################################################################


import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# |-----------------------------------------------------------------------------------------------------------------|
#                                                                                                                   |
#                       Function Definition                                                                         |
#                                                                                                                   |
# |-----------------------------------------------------------------------------------------------------------------|




def plot(source, dest):
    """ The function takes the source path and destination path. From source path it takes csv files and
        creates plot with matplotlib and save all figures it to destination path."""
    path_name = source   # path_name variable which takes source path
    df = pd.read_csv(path_name)  # From source path file which is csv it creates dataframe

    x_axis = [x+1 for x in range(df.shape[1])]    # For creating graph it needs x range which will be created by x_axis list [0 1 ..... 315]
    for i in range(df.shape[0]):
        y_axis = df.loc[i,:].values     # y_axis variable takes values from csv file
        plt.plot(x_axis,y_axis)
        path_name = os.path.join(dest,"{}".format(i))   # Path name takes destination folder path
        plt.savefig(path_name)
        plt.clf()
