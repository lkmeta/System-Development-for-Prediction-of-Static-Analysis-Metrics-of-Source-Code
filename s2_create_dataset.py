import pandas as pd
import os
import shutil
from dotenv import load_dotenv
load_dotenv()

resultsPath = "/home/kosmas/Desktop/thesis/AnalysisResults" # folder with all analysis results from all projects
projectPath = "/home/kosmas/Desktop/thesis/thesis_meta"

# all used metrics
general = ["LOC", "LLOC", "NM"]
complexity = ["NL", "NLE", "WMC"]
cohesion = ["LCOM5"]
inheritance = ["DIT", "NOC", "NOP", "NOD", "NOA"]
coupling = ["CBO", "NOI", "NII", "RFC"]
total_size = ["TNLA", "TNM", "TNOS", "TLLOC"]

dataset = {}

# used data_dictionary_antlr_antlr4 file
# csvPath = "data_dictionary_grails_grails-core"
csvPath = "data_dictionary_antlr_antlr4"
data_dictionary = pd.read_csv(os.path.join(projectPath, "gen_data/antlr/" + csvPath + ".csv"))
data_dictionary["RandomIDs"] = data_dictionary["RandomIDs"].apply(eval)
data_dictionary["Weeks"] = data_dictionary["Weeks"].apply(eval)
for metric in general + complexity + cohesion + inheritance + coupling + total_size:
    data_dictionary[metric + "_list"] = [[] for _ in range(len(data_dictionary))]

randomIDs = data_dictionary["RandomIDs"].tolist()
weeks = data_dictionary["Weeks"].tolist()

weeksInfo = {}
for ind1, week in enumerate(weeks):
    for ind2, weekNumber in enumerate(week):
        if (weekNumber not in weeksInfo):
            weeksInfo[weekNumber] = randomIDs[ind1][ind2]

weeks = list(weeksInfo.keys())
weeks.sort()

counter_week = 0

for week in weeks:

    if counter_week == 50:
        break
    counter_week += 1
    
    # print(week)
    folderPath = os.path.join(resultsPath, weeksInfo[week], "java")
    for fold in os.listdir(folderPath):
        for file in os.listdir(os.path.join(folderPath, fold)):
            if (file.endswith("-Class.csv")):
                weekResults = pd.read_csv(os.path.join(folderPath, fold, file))
                weekResults = weekResults[["Name", "LongName"] + general + complexity + cohesion + inheritance + coupling + total_size]
                data_dictionary = data_dictionary.drop_duplicates(subset=["Name", "LongName"])
                data_dictionary = data_dictionary.merge(weekResults, how="left", on=["Name", "LongName"])
                for metric in general + complexity + cohesion + inheritance + coupling + total_size:
                    data_dictionary[metric] = data_dictionary[metric].fillna(-1)
                    data_dictionary[metric + "_list"] = data_dictionary.apply(lambda x: x[metric + "_list"] + [x[metric]], 1)
                data_dictionary = data_dictionary.drop(general + complexity + cohesion + inheritance + coupling + total_size, axis=1)

# data_dictionary.to_csv(os.path.join("gen_data/antlr/", csvPath + "_completed.csv"), index=False)
