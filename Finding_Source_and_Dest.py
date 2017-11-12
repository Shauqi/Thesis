###################################################################
###                       import section                        ###
###################################################################

from pathlib import Path
import os
import glob


# |-----------------------------------------------------------------------------------------------------------------|
#                                                                                                                   |
#                       Function Definition                                                                         |
#                                                                                                                   |
# |-----------------------------------------------------------------------------------------------------------------|

def finding_src_dst(source_path):

    """The function takes source path like  C:/Users/User/Google Drive/DATA/850nm_new and gets the source path of csv folders
      and directory path where the graphs will be plotted automatically """

    source_path = Path(source_path) # source path contain the relative path

    sub_dir = [x for x in source_path.iterdir() if x.is_dir()] # the sub_dir varible contains all the subdirectory of the source folder
                                                               # sub_dir path = WindowsPath('C:/Users/User/Google Drive/DATA/850nm_new/fingertip_vedio_with_mobile_flash_but_without_led_flash')



    print(sub_dir)





source_path = "C:/Users/User/Google Drive/DATA/850nm_new"
finding_src_dst(source_path)