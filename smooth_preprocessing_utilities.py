# smoothing and removing peaks from dataset proccess for better training

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from typing import List


# complexity = ["NL", "NLE", "McCC"]
# cohesion = ["LCOM5"]
# inheritance = ["DIT", "NOC", "NOP", "NOD", "NOA"]
# coupling = ["CBO", "NOI", "NII", "RFC"]


# used files (with dropped classes, also used for non dropped classes)
# grails/data_dictionary_antlr_antlr4_completed_preprocessed_progress_3_non_straight
# aws/data_dictionary_aws_sdk_java_completed_preprocessed_progress_3_non_straight
# antlr/data_dictionary_antlr_antlr4_completed_preprocessed_progress_3_non_straight

# projectPath = "/home/kosmas/Desktop/thesis/thesis_meta/"

# csvPath = "gen_data/antlr/data_dictionary_antlr_antlr4_completed_preprocessed_progress_3_non_straight"
# data = pd.read_csv(projectPath + csvPath + ".csv")



# Function to convert data to metric list
def metric_data_to_list(data, metric):
    metric_list = []
    for i in range(len(data)):
        temp_list = list(map(float, data[metric][i][1:-1].split(', ') ))
        metric_list.append(temp_list)
    return metric_list



# Function to smooth data for better training
def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


# function remove peaks when data is too high or too low from previous weeks based on percentage 
# aslo apply smoothing to dataset
def remove_peaks_from_dataset(dataset, component, category, smooth_weight, peak_duration, perc):

    weigted_cat = []
    weights = [1 for i in range(len(category))]
    for metric in category:
        metric_name = metric + "_list"
        metric_list = metric_data_to_list(data, metric_name)
        metric_array = np.array(metric_list[component])
        metric_array = metric_array * float(weights[category.index(metric)])
        weigted_metric_list = metric_array.tolist()
        weigted_cat.append(weigted_metric_list)

    dataset = np.array(weigted_cat)
    real_dataset = np.mean(dataset, axis=0)


    smooth_dataset = np.mean(dataset, axis=0)

    # use this to remove peaks from dataset
    # window related method
    for k in range(peak_duration):
        duration = k+1
        for i in range(1, len(smooth_dataset) - 3*duration):

            # calculate average of first weeks
            first_avg = 0
            for j in range(i, i + duration):
                first_avg += smooth_dataset[j]
            first_avg = first_avg / duration

            # calculate average of middle weeks
            middle_avg = 0
            for j in range(i+duration, i + 2*duration):
                middle_avg += smooth_dataset[j]
            middle_avg = middle_avg / duration

            # calculate average of last weeks
            last_avg = 0
            for j in range(i+2*duration, i + 3*duration):
                last_avg += smooth_dataset[j]
            last_avg = last_avg / duration


            # if middle is too high or too low, remove peak
            if (middle_avg > first_avg * perc or middle_avg < first_avg / perc) and (middle_avg > last_avg * perc or middle_avg < last_avg / perc):
                # print("removed peak at " + str(i+duration) + " with duration: " + str(duration)) 
                smooth_dataset[i+duration] = (first_avg + last_avg) / 2
        

    # smooth dataset
    smooth_dataset = smooth(smooth_dataset, smooth_weight)
    smooth_dataset = np.array(smooth_dataset, dtype=np.float32)

    return real_dataset, smooth_dataset

# find best smooth weight from dataset according to RMSE
def plot_smooth_analysis(data, component, category):

    peak_duration = 8         # random first value for number of weeks
    perc = 1.2                # random first value for percentage of peak

    smooth_weight_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    colors = ["red", "blue", "green", "yellow", "orange"]
    # plt.figure(figsize=(30,16))
    # plt.title("Smooth weight analysis for component " + str(component) + " with category " + str(category))
    best_rmse = 100000
    for smooth_weight in smooth_weight_list:
        dataset, smooth_dataset = remove_peaks_from_dataset(data, component, category, smooth_weight, peak_duration, perc)

        # plt.plot(smooth_dataset, label="weight " + str(smooth_weight), color=colors[smooth_weight_list.index(smooth_weight)])

        # calculate RMSE between smoothed and original dataset
        rmse = np.sqrt(mean_squared_error(dataset, smooth_dataset))
        # print("RMSE for smooth weight " + str(smooth_weight) + " is " + str(rmse))

        if rmse < best_rmse:
            best_rmse = rmse
            best_smooth_weight = smooth_weight
    
    # plt.plot(dataset, label="dataset", color="brown")  
    # plt.legend()
    # plt.show()

    return best_smooth_weight, best_rmse

# find best peak duration for removing peaks from dataset according to RMSE
def plot_duration_analysis(data, component, category, best_smooth_weight):

    perc = 1.2                #  random first value for percentage of peak
    smooth_weight= best_smooth_weight

    peak_duration_list = [1, 2, 4, 8, 10]
    colors = ["red", "blue", "green", "yellow", "orange"]
    # plt.figure(figsize=(30,16))
    # plt.title("Peak smooth analysis for component " + str(component) + " with category " + str(category) + " and smooth weight " + str(smooth_weight))
    best_rmse = 100000
    for peak_duration in peak_duration_list:
        dataset, smooth_dataset = remove_peaks_from_dataset(data, component, category, smooth_weight, peak_duration, perc)

        # plt.plot(smooth_dataset, label="duration " + str(peak_duration), color=colors[peak_duration_list.index(peak_duration)])
        
        # calculate RMSE between smoothed and original dataset
        rmse = np.sqrt(mean_squared_error(dataset, smooth_dataset))
        # print("RMSE for duration " + str(peak_duration) + " is " + str(rmse))

        if rmse < best_rmse:
            best_rmse = rmse
            best_peak_duration = peak_duration
    
    # plt.plot(dataset, label="dataset", color="brown")  
    # plt.legend()
    # plt.show()

    return best_peak_duration, best_rmse

# find best percentage of peak to remove from dataset according to RMSE
def plot_peak_analysis(data, component, category, best_peak_duration, best_smooth_weight):

    peak_duration = best_peak_duration      
    smooth_weight= best_smooth_weight

    percentage_list = [2, 1.5, 1.2, 1.1, 1.05]
    colors = ["red", "blue", "green", "yellow", "orange"]
    # plt.figure(figsize=(30,16))
    # plt.title("Peak smooth analysis for component " + str(component) + " with category " + str(category) + " and smooth weight " + str(smooth_weight))
    best_rmse = 100000
    for perc in percentage_list:
        dataset, smooth_dataset = remove_peaks_from_dataset(data, component, category, smooth_weight, peak_duration, perc)

        # plt.plot(smooth_dataset, label="perc " + str(perc), color=colors[percentage_list.index(perc)])

        # calculate RMSE between smoothed and original dataset
        rmse = np.sqrt(mean_squared_error(dataset, smooth_dataset))
        # print("RMSE for perc " + str(perc) + " is " + str(rmse))

        if rmse < best_rmse:
            best_rmse = rmse
            best_perc = perc
    
    # plt.plot(dataset, label="dataset", color="brown")  
    # plt.legend()
    # plt.show()

    return best_perc, best_rmse

# find best parameters for smoothing dataset according to RMSE
def find_best_parameters(data, component, category):

    best_smooth_weight, best_rmse = plot_smooth_analysis(data, component, category)
    best_peak_duration, best_rmse = plot_duration_analysis(data, component, category, best_smooth_weight)
    best_perc, best_rmse = plot_peak_analysis(data, component, category, best_peak_duration, best_smooth_weight)

    return best_smooth_weight, best_peak_duration, best_perc


# plot best parameters for smoothing dataset 
def plot_best_smooth_params(data, component, category, category_name, best_peak_duration, best_smooth_weight, best_perc):

    peak_duration = best_peak_duration      
    smooth_weight= best_smooth_weight
    perc = best_perc

    plt.figure(figsize=(30,16))
    # plt.title("Best smooth analysis for component " + str(component) + " with category " + str(category_name))

    dataset, smooth_dataset = remove_peaks_from_dataset(data, component, category, smooth_weight, peak_duration, perc)

    plt.plot([], [], ' ', label= "smooth_weight = " + str(smooth_weight))
    plt.plot([], [], ' ', label= "peak_duration = " + str(peak_duration))
    plt.plot([], [], ' ', label= "perc = " + str(perc))


    # plt.plot(smooth_dataset, label="smoothed dataset", color="blue")
    plt.plot(dataset, label = "dataset", color="brown")
    plt.plot(smooth_dataset, label="smoothed dataset", color="blue")
    plt.xlabel("Weeks")
    plt.ylabel(category_name)
    plt.legend()
    plt_folder = "REPORT/"
    plt_name = plt_folder + "best_analysis_" + str(component) + "_" + str(category_name) + "_" + str(best_smooth_weight) + ".png"
    plt.savefig(plt_name)
    # plt.show()

    return smooth_dataset



# # random example test
# category_names = ["Cohesion", "Complexity", "Inheritance", "Coupling"]
# categories = [cohesion, complexity, inheritance, coupling]

# random_components = [18, 21, 192, 224, 249, 131, 182, 137, 200, 212, 301]

# for component in random_components:

#     category = random.choice(categories)
#     category_name = category_names[categories.index(category)]

#     best_smooth_weight, best_peak_duration, best_perc = find_best_parameters(data, component, category)
#     print("Component " + str(component) + " with category " + str(category_name) + " has the following best parameters for smooth process: ")
#     print("Best smooth weight is " + str(best_smooth_weight))
#     print("Best peak duration is " + str(best_peak_duration))
#     print("Best perc is " + str(best_perc))

#     new_dataset = plot_best_smooth_params(data, component, category, category_name, best_peak_duration, best_smooth_weight, best_perc)



