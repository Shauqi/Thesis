'''
# Simple code for seeing the plot of sine curve

sample = 10   # x range
x = [x/4 for x in range(sample*4)] # x = [0.0, 0.25, ..... , 9.75] where length of x is 40 
y = np.sin(x)       # y holds sin values for x values
plt.plot(x, y)      # For plotting
plt.xlabel('sample(n)')
plt.ylabel('voltage(V)')
plt.show()

'''

###################################################################
###                       import section                        ###
###################################################################


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from similarity import Similarity


# |-----------------------------------------------------------------------------------------------------------------|
#                                                                                                                   |
#                       Function Definition                                                                         |
#                                                                                                                   |
# |-----------------------------------------------------------------------------------------------------------------|



def normalize(list):
    '''This function makes the values in range of -1 to 1'''
    mini = min(list)
    maxi = max(list)
    for i in range(len(list)):
        list[i] = (((list[i] - mini) * 2) / (maxi - mini)) - 1      # Making each element Normalized
    return list



def get_best_indices(list,sin_val):
    ''' The function takes on single row and finds out the best indexes according to similarity distance. The similarity values used are
        Euclidean distance, Manhattan distance, Minkowski distance, Cosine distance and Jaccard distance.
        It returns a dictionary of list'''


    ### local optima saves a dictionary where dictionary is like { distance_type: [best_distance_value, best_lowest_index, best_upper_index] }
    local_optima = {"Euclidean":[9999999999,9999999,99999999], "Manhattan": [9999999999,9999999,99999999],
                    "Minkowski": [9999999999,9999999,99999999], "Cosine": [9999999999,9999999,99999999],
                    "Jaccard": [9999999999,9999999,99999999]}

    measures = Similarity()     ### Calling Similarity class
    size = len(sin_val)         ### size of sine value list which is 40

    for i in range(len(list)-size):

        ### Euclidean Portion
        val = measures.euclidean_distance(list[i:i+size],sin_val)
        if val <= local_optima["Euclidean"][0]:
            local_optima["Euclidean"] = [val,i,i+size]

        ### Manhattan Portion
        val = measures.manhattan_distance(list[i:i+size],sin_val)
        if val <= local_optima["Manhattan"][0]:
            local_optima["Manhattan"] = [val,i,i+size]

        ### Minkowski Portion
        val = measures.minkowski_distance(list[i:i+size],sin_val,3)
        if val <= local_optima["Minkowski"][0]:
            local_optima["Minkowski"] = [val, i, i + size]

        ### Cosine Portion
        val = measures.cosine_similarity(list[i:i + size], sin_val)
        if val <= local_optima["Cosine"][0]:
            local_optima["Cosine"] = [val, i, i + size]

        ### Jaccard Portion
        val = measures.jaccard_similarity(list[i:i + size], sin_val)
        if val <= local_optima["Jaccard"][0]:
            local_optima["Jaccard"] = [val, i, i + size]

    return local_optima


def get_global_optima(list, sin_val):
    global_optima = []
    best_row = {"Euclidean": -1, "Manhattan": -1, "Minkowski": -1, "Cosine": -1, "Jaccard": -1}
    best_val = {"Euclidean":[9999999999,9999999,99999999], "Manhattan": [9999999999,9999999,99999999],
                    "Minkowski": [9999999999,9999999,99999999], "Cosine": [9999999999,9999999,99999999],
                    "Jaccard": [9999999999,9999999,99999999]}


    for i in range(len(list)):
        sample = normalize(list[i])
        local_values = get_best_indices(sample, sin_val)

        if local_values["Euclidean"][0] <= best_val["Euclidean"][0]:
            best_val["Euclidean"] = local_values["Euclidean"]
            best_row["Euclidean"] = i

        if local_values["Manhattan"][0] <= best_val["Manhattan"][0]:
            best_val["Manhattan"] = local_values["Manhattan"]
            best_row["Manhattan"] = i

        if local_values["Minkowski"][0] <= best_val["Minkowski"][0]:
            best_val["Minkowski"] = local_values["Minkowski"]
            best_row["Minkowski"] = i

        if local_values["Cosine"][0] <= best_val["Cosine"][0]:
            best_val["Cosine"] = local_values["Cosine"]
            best_row["Cosine"] = i

        if local_values["Jaccard"][0] <= best_val["Jaccard"][0]:
            best_val["Jaccard"] = local_values["Jaccard"]
            best_row["Jaccard"] = i

        global_optima.append(local_values)

    return global_optima , best_row ,best_val


def get_sine():
    sample = 10  # x range
    x_range_for_sin = [x / 4 for x in range(sample * 4)]  # x = [0.0, 0.25, ..... , 9.75] where length of x is 40
    sin_val = np.sin(x_range_for_sin)  # y holds sin values for x values
    return sin_val


# |-----------------------------------------------------------------------------------------------------------------|
#                                                                                                                   |
#                              Main Section                                                                         |
#                                                                                                                   |
# |-----------------------------------------------------------------------------------------------------------------|



def getting_best_indexes_from_graph(src_path, dst_path, blue = 0, red = 0, green = 0):

    '''
    The function takes
        source path = C:/Users/User/Google Drive/DATA/850nm_new/fingertip_vedio_with_mobile_flash_but_without_led_flash_csv/1_Rehana_Karim_F49_GL=26.7/1_Rehana_Karim_F49_GL=26.7_10x10_blue.csv
        destination path = C:/Users/User/Google Drive/DATA/850nm_new/fingertip_vedio_with_mobile_flash_but_without_led_flash_csv/1_Rehana_Karim_F49_GL=26.7
        if blue is selected then it means the csv file contains blue data and creates text files against blue. Similar works for red and green one.
    '''

    df = pd.read_csv(src_path)          # It creates Data Frame. The frame contains csv file like 1_Rehana_Karim_F49_GL=26.7_10x10_blue.csv
    rows = df.loc[:].values
    sin_val = get_sine()
    glob_optima, best_row ,best_val = get_global_optima(rows, sin_val)

    if blue:
        ## This portion creates a text file containing best indeices of blue csv file
        dst_path = os.path.join(dst_path,"blue_optimized_index.txt")    # Path for destination
        blue_file = open(dst_path,"w+")
        blue_file.write("Local Best:\n\n")

        for i in glob_optima:
            blue_file.write(str(i))
            blue_file.write("\n\n")

        blue_file.write("The Best Row Index:\n\n")
        blue_file.write(str(best_row))
        blue_file.write("\n\n")
        blue_file.write("Best Value")
        blue_file.write(str(best_val))
        blue_file.close()

    elif red:
        ## This portion creates a text file containing best indeices of red csv file
        dst_path = os.path.join(dst_path, "red_optimized_index.txt")    # Path for destination
        red_file = open(dst_path, "w+")
        red_file.write("Local Best:\n\n")
        for i in glob_optima:
            red_file.write(str(i))
            red_file.write("\n\n")

        red_file.write("The Best Row Index:\n\n")
        red_file.write(str(best_row))
        red_file.write("\n\n")
        red_file.write("Best Value")
        red_file.write(str(best_val))
        red_file.close()

    elif green:
        ## This portion creates a text file containing best indeices of red csv file
        dst_path = os.path.join(dst_path, "green_optimized_index.txt")    # Path for destination
        green_file = open(dst_path, "w+")
        green_file.write("Local Best:\n\n")
        for i in glob_optima:
            green_file.write(str(i))
            green_file.write("\n\n")

        green_file.write("The Best Row Index:\n\n")
        green_file.write(str(best_row))
        green_file.write("\n\n")
        green_file.write("Best Value")
        green_file.write(str(best_val))
        green_file.close()

    else:
        print("Color not selected")

