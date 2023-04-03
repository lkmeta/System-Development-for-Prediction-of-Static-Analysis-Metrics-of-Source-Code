# # PREPROCESSING part 2
# # 1. check for outliers                           # couldn't find any outliers to remove
# # 2. check for big gaps (time)                    # filled all the gaps according to 4th rule
# # 3. start everything from 1w                     # done
# # 4. keep only the last when missing 1-2 weeks    # done

# this file includes also the following two extra rules
# # 5. if 60% of life is missing drop class         # done
# # 6. if life_class < 25w  drop class              # done



import pandas as pd
from itertools import groupby
import numpy as np


complexity = ["NL", "NLE", "McCC"]
cohesion = ["LCOM5"]
inheritance = ["DIT", "NOC", "NOP", "NOD", "NOA"]
coupling = ["CBO", "NOI", "NII", "RFC"]
metricsToKeep = [metric + "_list" for metric in  complexity + cohesion + inheritance + coupling]
columnsToKeep = ["Name", "LongName", "Repo", "RandomIDs", "Weeks", "StartAt", "DropAt", "McCC_list"] + metricsToKeep

# used files (with dropped classes, also used for non dropped classes)
# grails/data_dictionary_grails_core_completed_preprocessed
# aws/data_dictionary_aws_sdk_java_completed_preprocessed
# antlr/data_dictionary_antlr_antlr4_completed_preprocessed


projectPath = "/home/kosmas/Desktop/thesis/thesis_meta/"

csvPath = "gen_data/antlr/data_dictionary_antlr_antlr4_completed_preprocessed"
data = pd.read_csv(projectPath + csvPath + ".csv")

def missing_elements(L):
    start, end = L[0], L[-1]
    return sorted(set(range(start, end + 1)).difference(L))

# weeks + metrics_values (17 X y) 
# where y = number of weeks and 17 = number of metrics + weeks
component_data = []

# weeks + metrics_values for each component (non_dropped) X (17 X y)
total_data = []

# names from components kept
total_data_names = []

# for i in range(5):
for i in range(len(data)):

    component_data = []

    flag_5 = False
    flag_6 = False

    for name, values in data.iteritems():
        if name == "LongName":
            # print('{name}: {value}'.format(name=name, value=values[i]))
            total_data_names.append(values[i])
            continue

        # print('{name}: {value}'.format(name=name, value=values[i]))

        idx_week_list = list()

        if name == "Weeks":
            temp_list = list(map(int, values[i][1:-1].split(', ') ))

            # 3. Start everyting from 1w
            temp_list = [x - temp_list[0] + 1 for x in temp_list]
            # print("New temp_list: ", temp_list)

            # 5. Check if 50% of life is missing, in case yes then drop comp
            # keep 60% for now bcs samples are not so many
            if len(temp_list) < int(0.4 * temp_list[-1]):
                flag_5 = True
                # print("New 5 temp_list: ", temp_list)
                # print("len: ", len(temp_list))
                # print("len: ", int(0.4 * temp_list[-1]))
                break

            # 6. Check if life_comp < 25w, in case yes then drop comp
            if temp_list[-1] < 25:
                flag_6 = True
                # print("New 6 temp_list: ", temp_list)
                break

            # print("Comp kept with i:", i)

            # 4. Fill gaps - Keep only the last when missing 1-2 weeks
            j = 1
            while j < temp_list[-1]:
                if temp_list[j-1] == j:
                    last_value = temp_list[j-1]
                    j += 1
                else:
                    insert_list = [last_value] * (temp_list[j-1] - j)
                    temp_list[j-1:j-1] = insert_list
                    j += len(insert_list)

            component_data.append(temp_list)
        
        if flag_5 or flag_6:
            break

        # create the list with the metrics for each week
        if name in metricsToKeep:
            metric_temp_list = list(map(float, values[i][1:-1].split(', ') ))
            values_list = list()

            # print(metric_temp_list)

            metric_counter = 0
            last_metric_value = metric_temp_list[metric_counter]
            last_idx_value = 1
            for idx_week in temp_list:
                if idx_week == last_idx_value:
                    values_list.append(last_metric_value)
                else:
                    last_idx_value = idx_week
                    metric_counter += 1
                    last_metric_value = metric_temp_list[metric_counter]
                    values_list.append(last_metric_value)
            # print(values_list)

            component_data.append(values_list)
    
    if flag_5 or flag_6:
        total_data_names.pop()
        continue

    total_data.append(component_data)


metricsToKeep.insert(0, "Weeks")
total_data_df = pd.DataFrame (total_data, columns = [metric for metric in metricsToKeep])
# total_data_df.to_csv(csvPath + "_progress_3.csv", index=False)

# Save the used names for extra bibliographic analysis
total_data_names_df = pd.DataFrame (total_data_names, columns = ["LongName"])
# total_data_names_df.to_csv(csvPath + "_names_progress_3.csv", index=False)




##########################################################################################

# # EXTRA Graphs and stats for data


# # importing package
# import matplotlib.pyplot as plt
# import numpy as np

# # range 1 len data
# comp_list = list(range(0, len(data)))

# # plot some graphs
# num_of_plots = 10
# for counter in range(180, int(len(data)/num_of_plots)):

#     plt.figure(figsize=(35, 20))
#     plt.subplots_adjust(hspace=0.4)
#     plt.suptitle("Plot 10 Dropped Classes with their metrics plot: " + str(counter+1) , fontsize=16, y=0.95)

#     for temp_counter in range(num_of_plots):
        
#         temp_class_weeks = list(map(int, data.iloc[comp_list[temp_counter+counter*num_of_plots]]['Weeks'][1:-1].split(', ')))

#         # list(map(float, data.iloc[comp_list[1866]]['McCC_list'][1:-1].split(', ')))

#         # add a new subplot iteratively
#         ax = plt.subplot(5, 2, temp_counter + 1)

#         x = np.arange(1,len(temp_class_weeks)+1,1) 

#         for i in range(1, len(metricsToKeep)):
#             # y = total_data[temp_counter+counter*num_of_plots][i]
#             y = list(map(float, data.iloc[comp_list[temp_counter+counter*num_of_plots]][metricsToKeep[i]][1:-1].split(', ')))
#             plt.plot(x, y, label = metricsToKeep[i])

#         ax.set_title("Dropped Component: " + str(temp_counter+counter*num_of_plots+1))
#         ax.set_xlabel("")


#     # plt.legend()
#     ax.legend(loc='center left', bbox_to_anchor=(1, 1))

#     # plt.show()

#     plt_name = "data_figs/all/components_plt_" + str(counter+1) + "_" + str(num_of_plots) + ".png"
#     plt.savefig(plt_name)

#     plt.close()



# # plot some graphs
# num_of_plots = 10
# for counter in range(int(len(total_data)/num_of_plots)):

#     plt.figure(figsize=(35, 20))
#     plt.subplots_adjust(hspace=0.6)
#     plt.suptitle("Plot 10 Dropped components with their metrics plt_" + str(counter+1) , fontsize=16, y=0.95)

#     for temp_counter in range(num_of_plots):

#         # add a new subplot iteratively
#         ax = plt.subplot(5, 2, temp_counter + 1)

#         x = np.arange(1,len(total_data[temp_counter+counter*num_of_plots][0])+1,1) 

#         for i in range(1, len(metricsToKeep)):
#             y = total_data[temp_counter+counter*num_of_plots][i]
#             plt.plot(x, y, label = metricsToKeep[i])

#         ax.set_title("Dropped Component: " + str(temp_counter+counter*num_of_plots+1))
#         ax.set_xlabel("")


#     # plt.legend()
#     ax.legend(loc='center left', bbox_to_anchor=(1, 1))

#     # plt.show()

#     plt_name = "data_figs/dropped/components_plt_" + str(counter+1) + ".png"
#     plt.savefig(plt_name)

#     plt.close()




# myRange = np.arange(1,1001,1) # Could be something like myRange = range(1,1000,1)
# df = pd.DataFrame({"numbers": myRange})


# # find graph with frequency of modules weeks length
# my_l = list()
# for i in range(len(total_data_df)):
#     my_l.append(len(total_data_df['Weeks'][i]))
#     # print(len(data['Weeks'][i]))

# new_y = [len(list(group)) for key, group in groupby(sorted(my_l))]
# new_x = [key for key, group in groupby(sorted(my_l))]

# from matplotlib import pyplot as plt

# plt.title('Number of components with a certain number of weeks')
# plt.xlabel('Number of weeks')
# plt.ylabel('Number of components')
# plt.plot(new_x, new_y)
# # plt.show()
# plt_name = "data_figs/weeks-components.png"
# plt.savefig(plt_name)
# plt.close()


# weeks_df = pd.DataFrame (new_x, columns = ["Weeks_Number"])
# weeks_df.insert(1, 'Components_Number', new_y)


# import seaborn
# seaborn.lmplot(y='Components_Number', x='Weeks_Number', data=weeks_df)  

# import seaborn as sns
# sns.boxplot(x=weeks_df['Weeks_Number'])
# # plt.show()
# plt_name = "data_figs/number_of_weeks.png"
# plt.savefig(plt_name)
# plt.close()



# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(weeks_df['Weeks_Number'], weeks_df['Components_Number'])
# ax.set_xlabel('Number of weeks')
# ax.set_ylabel('Number of components')
# # plt.show()
# plt_name = "data_figs/number_of_weeks_per_comp.png"
# plt.savefig(plt_name)
# plt.close()




# # # Metrics Graphs

# # weeks_list_df = total_data_df['Weeks']
# # metricsToKeep.pop(0)

# # for metric in metricsToKeep:

# #     metric_list_df = total_data_df[metric]

# #     plt.figure(figsize=(16,8))

# #     for counter in range(len(metric_list_df)):
# #         plt.plot(weeks_list_df[counter], metric_list_df[counter])

# #     plt.title(str(metric) + " graph for dropped components")
# #     plt.xlabel('Week')
# #     plt.ylabel('Value for each component')
    
# #     # plt.show()

# #     plt_name = "data_figs/metrics/" + str(metric) + "_dropped.png"
# #     plt.savefig(plt_name)

# #     plt.close()



# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(total_data_df['LOC_list'][0], total_data_df['Weeks'][0])
# ax.set_xlabel('LOC value')
# ax.set_ylabel('Weeks')
# plt.show()
# plt_name = "data_figs/number_of_weeks_per_comp.png"
# plt.savefig(plt_name)
# plt.close()




##########################################################################################

# Process to find straight and not straight lines


# # general = ["LOC", "LLOC", "NM"]
# # complexity = ["NL", "NLE", "WMC"]
# # cohesion = ["LCOM5"]
# # inheritance = ["DIT", "NOC", "NOP", "NOD", "NOA"]
# # coupling = ["CBO", "NOI", "NII", "RFC"]
# # metricsToKeep = [metric + "_list" for metric in general + complexity + cohesion + inheritance + coupling]

# csvPath = "./gen_data/data_dictionary_aws_sdk_java_completed_preprocessed_non_dropped_progress_3"
# data = pd.read_csv(csvPath + ".csv")

# def metric_data_to_list(data, metric):
#     metric_list = []
#     for i in range(len(data)):
#         temp_list = list(map(float, data[metric][i][1:-1].split(', ') ))
#         metric_list.append(temp_list)
#     return metric_list

# # Function to check if all elements in a list are equal
# def all_equal(iterable):
#     g = groupby(iterable)
#     return next(g, True) and not next(g, False)


# ##########################################################################################
# ## Find components with all straight lines in their metrics
# weeks = metric_data_to_list(data, "Weeks")
# straight_metrics = 0
# straight_components = 0
# non_straight_components = []


# for i in range(len(weeks)):
#     for metric in metricsToKeep:
#         if all_equal(metric_data_to_list(data, metric)[i]):
#             straight_metrics += 1
        

#     if straight_metrics == len(metricsToKeep):
#         straight_components += 1
#         print("Component " + str(i) + " has all straight lines in its metrics")
#     else:
#         non_straight_components.append(i)
    
#     straight_metrics = 0
    
# print("Number of components with all straight lines in their metrics: " + str(straight_components))
# print("Number of components with at least one non-straight line in their metrics: " + str(len(non_straight_components)))

# # save non-straight components to csv
# new_df = pd.DataFrame(columns=data.columns)
# for i in range(len(non_straight_components)):
#     new_df.loc[i] = data.loc[non_straight_components[i]]

# new_df.to_csv(csvPath + "_non_straight.csv", index=False)



# ##########################################################################################

# # antlr
# # Total dropped components from progress_3: 310
# # Number of components with all straight lines in their metrics: 132
# # Number of components with at least one non-straight line in their metrics: 178
# # ~43% of dropped dataset is filled completely with straight lines


# # Total non-dropped components from progress_3: 1155
# # Number of components with all straight lines in their metrics: 657
# # Number of components with at least one non-straight line in their metrics: 498
# # ~57% of non dropped dataset is filled completely with straight lines

# ##########################

# # grails
# # Total dropped components from progress_3: 1417
# # Number of components with all straight lines in their metrics: 374
# # Number of components with at least one non-straight line in their metrics: 1043
# # ~26% of dropped dataset is filled completely with straight lines


# # Total non-dropped components from progress_3: 580
# # Number of components with all straight lines in their metrics: 305
# # Number of components with at least one non-straight line in their metrics: 275
# # ~52% of non dropped dataset is filled completely with straight lines


# ##########################

# # aws
# # Total dropped components from progress_3: 2111
# # Number of components with all straight lines in their metrics: 0
# # Number of components with at least one non-straight line in their metrics: 2111
# # 0% of dropped dataset is filled completely with straight lines


# # Total non-dropped components from progress_3: 13235
# # Number of components with all straight lines in their metrics: ...
# # Number of components with at least one non-straight line in their metrics: ...
# # % of non dropped dataset is filled completely with straight lines

# ##########################################################################################


# # New graphs for non-straight components


# csvPath = "./gen_data/antlr/data_dictionary_antlr_antlr4_completed_preprocessed_progress_3_non_straight"
# data_antlr = pd.read_csv(csvPath + ".csv")


# csvPath = "./gen_data/grails/data_dictionary_grails_core_completed_preprocessed_progress_3_non_straight"
# data_grail = pd.read_csv(csvPath + ".csv")


# csvPath = "./gen_data/aws/data_dictionary_aws_sdk_java_completed_preprocessed_progress_3_non_straight"
# data_aws = pd.read_csv(csvPath + ".csv")


# csvPath = "./gen_data/data_dictionary_aws_sdk_java_completed_preprocessed_ALL_25_50_non_straight"
# data_all_25 = pd.read_csv(csvPath + ".csv")

# # concat_data = pd.concat([data_antlr, data_grail], ignore_index=True)
# concat_data = pd.concat([data_antlr, data_grail, data_aws, data_all_25], ignore_index=True)

# concat_data_names = ["antlr4", "grails-core", "aws-sdk"]

# # concat_data2 = pd.concat([concat_data1, concat_data], ignore_index=True)


# weeks = metric_data_to_list(concat_data, "Weeks")


# # find graph with frequency of classses per week 
# my_l = list()
# for i in range(len(weeks)):
#     my_l.append(len(weeks[i]))
#     # print(len(data['Weeks'][i]))


# new_y = [len(list(group)) for key, group in groupby(sorted(my_l))]
# new_x = [key for key, group in groupby(sorted(my_l))]


# # group by range of weeks 
# # 0-10, 10-20, 20-30, ...
# max_num_of_weeks = max(new_y)
# range_of_weeks = np.arange(25,400,25)


# range_of_weeks = np.append(range_of_weeks, max_num_of_weeks+25)


# plot_y = []
# plot_x = []
# for i in range(len(range_of_weeks)-1):
#     plot_x.append("(" + str(range_of_weeks[i]) + "," + str(range_of_weeks[i+1]) + ")")
#     plot_y.append(0)

# for i in range(len(weeks)):
#     for j in range(len(range_of_weeks)-1):
#         if len(weeks[i]) >= range_of_weeks[j] and len(weeks[i]) < range_of_weeks[j+1]:
#             plot_y[j] += 1
#             break


# plot_x = plot_x[:-1]
# plot_x = plot_x + ["(" + str(range_of_weeks[-2]) + "," + "inf" + ")"]

# # print(plot_x)
# # print(plot_y)


# import matplotlib.pyplot as plt

# # plot columns for each range of weeks
# plt.figure(figsize=(30,15))
# plt.bar(plot_x, plot_y)
# plt.xlabel("Number of Weeks", fontsize=16)
# plt.ylabel("Number of Classes", fontsize=16)
# # plt.show()


# plt_name = "./data_figs/number_of_weeks_per_class_after_v3.png"
# plt.savefig(plt_name)
# plt.close()



