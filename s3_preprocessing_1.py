# # PREPROCESSING part 1
# # 1. week_length > 4 weeks
# # 2. find dropped and non dropped classes


import pandas as pd
import os
import numpy as np


general = ["LOC", "LLOC", "NM"]
complexity = ["NL", "NLE", "WMC"]
cohesion = ["LCOM5"]
inheritance = ["DIT", "NOC", "NOP", "NOD", "NOA"]
coupling = ["CBO", "NOI", "NII", "RFC"]
metricsToKeep = [metric + "_list" for metric in general + complexity + cohesion + inheritance + coupling]
columnsToKeep = ["Name", "LongName", "Repo", "RandomIDs", "Weeks", "StartAt", "DropAt", "McCC_list"] + metricsToKeep

# used files
# grails/data_dictionary_grails_core_completed
# aws/data_dictionary_aws_sdk_java_completed
# antlr/data_dictionary_antlr_antlr4_completed


projectPath = "/home/kosmas/Desktop/thesis/thesis_meta/"

csvPath = "gen_data/antlr/data_dictionary_antlr_antlr4_completed"
data_dictionary = pd.read_csv(projectPath + csvPath + ".csv")
data_dictionary["RandomIDs"] = data_dictionary["RandomIDs"].apply(eval)
data_dictionary["Weeks"] = data_dictionary["Weeks"].apply(eval)
for metric in general + complexity + cohesion + inheritance + coupling:
    data_dictionary[metric + "_list"] = data_dictionary[metric + "_list"].apply(eval)
    data_dictionary[metric + "_list"] = data_dictionary[metric + "_list"].apply(lambda x: list(filter(lambda a: a != -1, x)))
data_dictionary["McCC_list"] = data_dictionary.apply(lambda row: [a if (not(np.isnan(a))) else 0 for a in (np.divide(row["WMC_list"], row["NM_list"]))], 1)

print(len(data_dictionary.index))
for metric in complexity + cohesion + inheritance + coupling:
    data_dictionary["result"] = data_dictionary.apply(lambda row: row[metric + "_list"][-1] - row[metric + "_list"][0], axis=1)
    data_dictionary["result_" + metric] = False
    data_dictionary.loc[data_dictionary["result"] > 0, "result_" + metric] = True
    data_dictionary = data_dictionary.drop(["result"], axis=1)
data_dictionary = data_dictionary[data_dictionary.apply(lambda row: any(row), axis = 1)]
print(len(data_dictionary.index))

data_dictionary["weeksLength"] = data_dictionary["Weeks"].str.len()
data_dictionary = data_dictionary.loc[data_dictionary["weeksLength"] > 4]
print(len(data_dictionary.index))

data_dictionary_dropped = data_dictionary.loc[data_dictionary["DropAt"] != -1]
print(len(data_dictionary_dropped.index))

data_dictionary_dropped = data_dictionary_dropped[columnsToKeep]
# data_dictionary_dropped.to_csv(csvPath + "_preprocessed.csv", index=False)

data_dictionary_non_dropped = data_dictionary.loc[data_dictionary["DropAt"] == -1]
print(len(data_dictionary_non_dropped.index))

data_dictionary_non_dropped = data_dictionary_non_dropped[columnsToKeep]
# data_dictionary_non_dropped.to_csv(csvPath + "_preprocessed_non_dropped.csv", index=False)
