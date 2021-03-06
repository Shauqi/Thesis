#---------------------  Oct 27, 2017 6:32 PM   ---------------------
###################################################################
###                       import section                        ###
###################################################################

import numpy as np
import cv2
from pprint import pprint
import csv
import pandas as pd
import numpy as np
import glob
import time
import matplotlib.pyplot as plt
import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import norm
import matplotlib.mlab as mlab
###################################################################
###                       Init section                          ###
###################################################################


#---------------------To run the code ----------------Shift + Control + R

#---------------------To debug the code ----------------Shift + Control + D



def oneDArray(x): return list(itertools.chain(*x))

from extractStoreDataFromVideo import extractStoreDataFromVideo

from generate_MxN_Blockwise_Average_Intensities import generate_MxN_Blockwise_Average_Intensities


# |-----------------------------------------------------------------------------------------------------------------|
#                                                                                                                   |
#                       extractStoreDataFromVideo                                                                   |
#                                                                                                                   |
# |-----------------------------------------------------------------------------------------------------------------|


#sourceDir='/Users/kamrul/Local_HDD/Research/KUET-Glucose Research/kuet-glucose-data/940nm/fingertip_video_with_led_flash_but_without_mobile_flash'
#destDir='/Users/kamrul/Local_HDD/Research/KUET-Glucose Research/kuet-glucose-data/940nm/fingertip_video_with_led_flash_but_without_mobile_flash_csv'
#extractStoreDataFromVideo (sourceDir, destDir)

# |-----------------------------------------------------------------------------------------------------------------|
#                                                                                                                   |
#                       generate_MxN_Blockwise_Average_Intensities                                                  |
#                                                                                                                   |
# |-----------------------------------------------------------------------------------------------------------------|

sourceDir='C:/Users/User/Google Drive/DATA/850nm_new/fingertip_video_without_mobile_and_led_flash'
destDir='C:/Users/User/Google Drive/DATA/850nm_new/fingertip_video_without_mobile_and_led_flash_csv'

mRow=10
nCol=10

generate_MxN_Blockwise_Average_Intensities(sourceDir, destDir, mRow, nCol)


# |-----------------------------------------------------------------------------------------------------------------|
#                                                                                                                   |
#            generate_plot_from_data_and_find_best_indices_which_have similarity to sine                            |
#                                                                                                                   |
# |-----------------------------------------------------------------------------------------------------------------|

from Finding_Source_and_Dest import finding_src_dst

source_path = "C:/Users/User/Google Drive/DATA/850nm_new"
finding_src_dst(source_path, optimized_index=1, graph=0)  # if you want to plot graph then pass graph = 1
                                                          # if you want to get best indices then pass optimized_index = 1