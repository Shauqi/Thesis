# |-----------------------------------------------------------------------------------------------------------------|
#                                                                                                                   |
#                       Function Definition                                                                         |
#                                                                                                                   |
# |-----------------------------------------------------------------------------------------------------------------|

# The image frame is divided into M x N = 3 x 4 blockwise and we will keep the average intensities of each block
# of pixel data since the whole RGB dataset takes a lot of spaces and took more computing time

#                   |-------|------|------|------|
#                   |  1    |   2  |  3   |  4   |
#                   |-------|------|------|------|
#                   |  5    |   6  |  7   |  8   |
#                   |-------|------|------|------|
#                   |  9    |  10  |  11  |  12  |
#                   |-------|------|------|------|

# We store the averaged intensities as a CSV file. For each red, green and blue pixels we will make different CSV file
# from each video file.

# So if the block is taken as 3 x 3 then we will get 9 averaged pixels from 9 different locations.

# We will store 9 pixels in 9 rows of the matrix. Then we get multiple averaged pixels from different frames of the
# video. If the video has 300 frames then the red pixels matrix will be 9 x 300 in size.

# For green pixels, the matrix size will also be 9 x 300 and so on

#|---------------------------------------------------------------------------------------------------------------------|
#                                                                                                                      |
#                                       Import section                                                                 |
#                                                                                                                      |
#|---------------------------------------------------------------------------------------------------------------------|

import numpy as np
import cv2
import os
import pandas as pd
import glob



def generate_MxN_Blockwise_Average_Intensities(sourceDir, destDir, mRow, nCol):

    # sourceDir = Where the video files are stored

    # destDir = Where we have to store the CSV files

    # Here the name of the full file name of a video is making to grab the video file

    # To get all the video name that is ending by .mp4
    sourceDir = sourceDir + '/*.mp4'

    # Get all of the video files name
    vList = glob.glob(sourceDir)


    # For each video in the list
    for i in range(len(vList)):

        vDirName = vList[i]
        head, tail = os.path.split(vDirName)  # The real path name will be splitted in to dir name and file name
        print(tail)

        #|--------------------------------------------------------------------------------------------------------------|
        # To make a directory as the video file name we use this command                                                |
        # test_video.mp4 video file name makes the following directory                                                  |
        # /Users/kamrul/Local_HDD/Research/Python Code/Python Code Purdue University/KUET Data Analysis/test_video      |
        #|--------------------------------------------------------------------------------------------------------------|


        destDirFinal = destDir + '/' + tail[:len(tail) - 4]

        if not os.path.exists(destDirFinal):
            os.makedirs(destDirFinal)

        # Here we ar ecapturing the video data using cv2.VideoCapture ()
        cap = cv2.VideoCapture(vDirName)


        if not cap.isOpened():  # If the video is not opern then it will pring a msg and return
            print("Could not open :", vDirName)
            return

        while not cap.isOpened():
            cap = cv2.VideoCapture(vDirName)
            cv2.waitKey(1000)
            print("Wait for the header")

        #redAllBlockAvg, greenAllBlockAvg, blueAllBlockAvg will store all frames blockwise avg pixels
        # This variables are storing all of the BLOCKs avg values
        redAllBlockAvg = [[0 for x in range(1)] for y in range(mRow*nCol)]
        greenAllBlockAvg = [[0 for x in range(1)] for y in range(mRow * nCol)]
        blueAllBlockAvg = [[0 for x in range(1)] for y in range(mRow * nCol)]


        while True:

            # One frame is read
            flag, frame = cap.read()

            if flag:

                # |-------------------------------------------------------------------------------------------------------------|
                # |Here we are generating the other inner information of the frame using this command                           |
                # |https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get   |
                '''
                # --------------------------------------------------------------------------------------------------------------|
                # Here, I keep the code for future use if we need to know about the frames other information
                # --------------------------------------------------------------------------------------------------------------|

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
                # |-------------------------------------------------------------------------------------------------------------|

                # --------------------------------------------------------------------------------------------------------------|
                # Here we are getting the information of the frame that is under process                                        |
                # --------------------------------------------------------------------------------------------------------------|
                nFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

                # The frame is taken as MxN size  M=num of row and N=num of col
                rBlock = mRow  # How many blocks are considered row wise
                cBlock = nCol  # How many blocks are considered column wise


                # Based on number of block, we will defince the width and height of the frame
                block_width = int(frame_width / cBlock)
                block_height = int(frame_height / rBlock)

                # (x1,y1)
                #  |-------------|----------------|---------------|
                #  |             |                |               |
                #  |             |                |               |
                #  |             |                |               |
                #  |             |                |               |
                #  |-------------|----------------|---------------|
                #             (x2,y2)



                x1 = 0
                y1 = 0
                x2 = x1 + block_width
                y2 = y1 + block_height

                #             (x1,y1)
                #  |-------------|----------------|---------------|
                #  |             |                |               |
                #  |             |                |               |
                #  |             |                |               |
                #  |             |                |               |
                #  |-------------|----------------|---------------|
                #                              (x2,y2)



                # avgRedFrame, avgGreenFrame, avgBlueFrame will store only a single frame blockwise avg pixels
                avgRedFrame = []
                avgGreenFrame = []
                avgBlueFrame = []


                for rowIndx in range(rBlock):

                    for colIndx in range(cBlock):

                        redBloc = redPixel[y1:y2, x1:x2]
                        greenBloc = greenPixel[y1:y2, x1:x2]
                        blueBloc = bluePixel[y1:y2, x1:x2]

                        # Averaging the whole block of red pixel intensities, then adding in the array
                        avg1 = redBloc.mean()
                        avgRedFrame.append(avg1)

                        # Averaging the whole block of green pixel intensities, then adding in the array
                        avg2 = greenBloc.mean()
                        avgGreenFrame.append(avg2)

                        # Averaging the whole block of blue pixel intensities, then adding in the array
                        avg3 = blueBloc.mean()
                        avgBlueFrame.append(avg3)

                        # Change the width of x1
                        x1 = x2
                        # Increase the width of x1
                        if colIndx == cBlock - 2:  # If the last block of the col is found
                            x2 = frame_width  # take the rest of the pixels
                        else:
                            x2 = x1 + block_width

                    # Change the height of y1
                    y1 = y2

                    # Increase the height of y2
                    if rowIndx == rBlock - 2:  # If the last y-block is found
                        y2 = frame_height  # take the rest of the pixels
                    else:
                        y2 = y1 + block_height

                    # Start from begining of x1=0
                    x1 = 0

                    # Increase the width of x2
                    if colIndx == cBlock - 2:
                        x2 = frame_width
                    else:
                        x2 = x1 + block_width

                if pos_frame==1:
                    redAllBlockAvg = avgRedFrame
                    greenAllBlockAvg = avgGreenFrame
                    blueAllBlockAvg = avgBlueFrame

                else:
                    redAllBlockAvg=np.column_stack((redAllBlockAvg, avgRedFrame))
                    greenAllBlockAvg = np.column_stack((greenAllBlockAvg, avgGreenFrame))
                    blueAllBlockAvg = np.column_stack((blueAllBlockAvg, avgBlueFrame))

                    '''
                    We are storing the average intensity of a certain block for each frame in a video
                    So first row contains the First BLOC of each frames avg values
                    So total 10x10=100 blocks infroamtion are storing in this table.
                    We are generating ONE this kind of CSV file from ONE video frame
                      
                    <---------------------- Frame (F) number ---------------------------------------------------------------->
                      F=1    F=2     F=3                                                                                F=324
                    |------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
                    |  b1  |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
                    |------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
                    |  b2  |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
                    |------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
                    |  b3  |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
                    |------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
                    |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
                    |------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
                    |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
                    |------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
                    |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
                    |------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
                    |b100  |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
                    |------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
                    
                    frame= 1 to total number of frame of a video
                    b1, b2, ....b100 = block number of a frame
                    
                    
                    '''
                avgRedFrame=[]
                avgGreenFrame = []
                avgBlueFrame = []


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

        # --------------------------------------------------------------------------------------------------------------|
        # Since we store three RGB blockwise averaged csv files per video, we take this code here and we will
        # save after each video processing is done
        # --------------------------------------------------------------------------------------------------------------|

        fileName1 = destDirFinal + '/' + tail[:len(tail) - 4] + '_' +  str(mRow) + 'x'+str(mRow) + '_red.csv'
        fileName2 = destDirFinal + '/' + tail[:len(tail) - 4] + '_' +  str(mRow) + 'x'+str(mRow) + '_green.csv'
        fileName3 = destDirFinal + '/' + tail[:len(tail) - 4] + '_' +  str(mRow) + 'x'+str(mRow) + '_blue.csv'

        # --------------------------------------------------------------------------------------------------------------|
        # Here we are saving the blockwise averaged matrix in CSV format ---------                                      |
        # Red, green and blue pixels are stored differently in different CSV file for each video with                   |
        # block size number                                                                                             |
        # --------------------------------------------------------------------------------------------------------------|

        df = pd.DataFrame(redAllBlockAvg)
        df.to_csv(fileName1)

        df = pd.DataFrame(greenAllBlockAvg)
        df.to_csv(fileName2)

        df = pd.DataFrame(blueAllBlockAvg)
        df.to_csv(fileName3)



    cap.release()
    cv2.destroyAllWindows()