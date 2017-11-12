'''
nVideo= Numebr of Video files
nFrame= Number of frames in a video file
vList = name of the video list in a directory
vDirName= Video file name with dir
vName= Video file name
pMatrix= Pixel matrix of an image frame

dirName= Any Specific Directory name
vidDir= Video directory name
vidName= Video file name
imgDir= Image directory name

stFrame= Starting frame number
endFrame= Ending frame number
totalFrame = Total frame number

redPixel= red pixel matrix of a frame
greenPixel= Green pixel matrix of a frame
bluePixel= Blue pixel matrix of a frame

tempMatrix= Temporary matrix to hold dataset

histValue= Histogram value of the given matrix
redHist= Histogram value genereated from red pixels only
greenHist= Histogram value genereated from green pixels only
blueHist= Histogram value genereated from blue pixels only
'''

# ---------------------  Oct 27, 2017 6:32 PM   ---------------------
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


# ---------------------To run the code ----------------Shift + Control + R

# ---------------------To debug the code ----------------Shift + Control + D



def oneDArray(x): return list(itertools.chain(*x))

from generate_MxN_Blockwise_Average_Intensities import generate_MxN_Blockwise_Average_Intensities

# |-----------------------------------------------------------------------------------------------------------------|
#                                                                                                                   |
#                       extractStoreDataFromVideo                                                                   |
#                                                                                                                   |
# |-----------------------------------------------------------------------------------------------------------------|


#  sourceDir='/Users/kamrul/Local_HDD/Research/KUET-Glucose Research/kuet-glucose-data/940nm/fingertip_video_with_led_flash_but_without_mobile_flash'
# destDir='/Users/kamrul/Local_HDD/Research/KUET-Glucose Research/kuet-glucose-data/940nm/fingertip_video_with_led_flash_but_without_mobile_flash_csv'

# extractStoreDataFromVideo (sourceDir, destDir)

# |-----------------------------------------------------------------------------------------------------------------|
#                                                                                                                   |
#                       generate_MxN_Blockwise_Average_Intensities                                                  |
#                                                                                                                   |
# |-----------------------------------------------------------------------------------------------------------------|

'''
Program Name: extractStoreDataFromVideo 
Input: Directory of the video file as source dir, and the output CSV files dir as destDir
Output: Make a directory with the same name of video file in the same directory where the video file is stored. Then keep all the frames with frame number. The frame name should be the video file name.
Using the same name, make a CSV file for each red, green and blue pixels. Here each file would be ended with _red, _green, and _blue.
'''



###################################################################
###                       import section                        ###
###################################################################

import numpy as np
import cv2
import os
import pandas as pd
import glob


# |-----------------------------------------------------------------------------------------------------------------|
#  In this function, we gave input of the video files directory name as source directory. The source directory
# stores all the fingertip video files. The destination directory will be the storage of each video file's CSV file
# directory. Each folder is creadted with the same name of video file name. Each directory named with the video file name
# will store the red, green and blue pixels as a 2-D matrix in csv file for each frame.
# |-----------------------------------------------------------------------------------------------------------------|



def extractStoreDataFromVideo(sourceDir, destDir):
    # Here the name of the full file name of a video is making to grab the video file
    # |-------------------------------------------------------------------------------------------------------------|
    # |Here we are generating the full information of the frame using this command                                  |
    # |https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get   |
    # |-------------------------------------------------------------------------------------------------------------|

    # To get all the video name that is ending by .mp4
    sourceDir = sourceDir + '/*.mp4'

    # Get all of the video files name
    vList = glob.glob(sourceDir)

    # vidDir = vidDir + '/'+ vidName


    for i in range(len(vList)):
        vDirName = vList[i]
        head, tail = os.path.split(vDirName)
        print
        tail

        # To make a directory as the video file name we use this command
        # test_video.mp4 video file name makes the following directory
        # /Users/kamrul/Local_HDD/Research/Python Code/Python Code Purdue University/KUET Data Analysis/test_video

        destDirFinal = destDir + '/' + tail[:len(tail) - 4]
        # print destDir

        if not os.path.exists(destDirFinal):
            os.makedirs(destDirFinal)

        cap = cv2.VideoCapture(vDirName)

        if not cap.isOpened():
            print("Could not open :", vDirName)
            return

        while not cap.isOpened():
            cap = cv2.VideoCapture(vDirName)
            cv2.waitKey(1000)
            print("Wait for the header")

        while True:
            flag, frame = cap.read()

            if flag:

                # ----------------------------------------------------------------------------------------------------|
                # Here we are getting the information of the frame that is under process
                # ----------------------------------------------------------------------------------------------------|
                # nFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # frame_fps = cap.get(cv2.CAP_PROP_FPS)
                # print 'frame_width='+str(frame_width) + 'frame_height =' + str(frame_height)


                pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(pos_frame)

                # We can also show the fram when The frame is ready and already captured
                # cv2.imshow('video', frame)

                # Store the whole RGB pixel information in the pMatrix
                pMatrix = frame

                # Collecting the RGB data from each frame
                redPixel = pMatrix[:, :, 2]  # Storing the red pixels
                greenPixel = pMatrix[:, :, 1]  # Storing the green pixels
                bluePixel = pMatrix[:, :, 0]  # Storing the blue pixels

                # ----------------------------------------------------------------------------------------------------|
                # ------ Here we are saving the matrix in CSV format ---------
                # Red, green and blue pixels are stored differently in different CSV file for each frame with
                # frame number
                # ----------------------------------------------------------------------------------------------------|

                fileName1 = destDirFinal + '/' + tail[:len(tail) - 4] + '_' + 'F=' + str(pos_frame) + '_red.csv'
                fileName2 = destDirFinal + '/' + tail[:len(tail) - 4] + '_' + 'F=' + str(pos_frame) + '_green.csv'
                fileName3 = destDirFinal + '/' + tail[:len(tail) - 4] + '_' + 'F=' + str(pos_frame) + '_blue.csv'

                df = pd.DataFrame(redPixel)
                df.to_csv(fileName1)

                df = pd.DataFrame(greenPixel)
                df.to_csv(fileName2)

                df = pd.DataFrame(bluePixel)
                df.to_csv(fileName3)

            else:
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                print("Frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)

            if cv2.waitKey(10) == 27:
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break

            # ----------------------------------------------------------------------------------------------------|
            # Here, I keep the code for future use if we need to know about the frames other information
            # ----------------------------------------------------------------------------------------------------|

            '''
            frame_mode = int(cap.get(cv2.CAP_PROP_MODE  )) # Backend-specific value indicating the current capture mode
            frame_brightness = int(cap.get(cv2.CAP_PROP_BRIGHTNESS )) # Brightness of the image (only for cameras).
            frame_contrast = int(cap.get(cv2.CAP_PROP_CONTRAST )) # Contrast of the image (only for cameras).
            frame_saturation = int(cap.get(cv2.CAP_PROP_SATURATION )) # Saturation of the image (only for cameras).
            fraem_hue = int(cap.get(cv2.CAP_PROP_HUE )) # Hue of the image (only for cameras).
            frame_gain = int(cap.get(cv2.CAP_PROP_GAIN )) # Gain of the image (only for cameras).
            frame_exposure = int(cap.get(cv2.CAP_PROP_EXPOSURE))  # Exposure (only for cameras).
            frame_rgb = int(cap.get(cv2.CAP_PROP_CONVERT_RGB))  # Boolean flags indicating whether images should be converted to RGB.
            frame_wBalanceU = int(cap.get(cv2.CAP_PROP_WHITE_BALANCE_U )) # The U value of the whitebalance setting (note: only supported by DC1394 v 2.x backend currently)
            frame_wBalanceV = int(cap.get(cv2.CAP_PROP_WHITE_BALANCE_V))  # The V value of the whitebalance setting (note: only supported by DC1394 v 2.x backend currently)
 
            frame_isoSpeed = int(cap.get(cv2.CAP_PROP_ISO_SPEED))  # The ISO speed of the camera (note: only supported by DC1394 v 2.x backend currently)
            frame_bufferSize = int(cap.get(cv2.CAP_PROP_BUFFERSIZE))  # Amount of frames stored in internal buffer memory (note: only supported by DC1394 v 2.x backend currently)
            '''

    cap.release()
    cv2.destroyAllWindows()





