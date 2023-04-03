import json
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from utilities import hasMissingWeeks

# resultsPath = os.getenv("RESULTS_PATH")
# projectPath = os.getenv("PROJECT_PATH")

resultsPath = "/home/kosmas/Desktop/thesis/AnalysisResults" # folder with all analysis results from all projects
projectPath = "/home/kosmas/Desktop/thesis/thesis_meta"

with open(os.path.join(projectPath, "week_mappings.json"), "r") as file:
	week_mappings = json.load(file)

columnsToKeep = ["Name", "LongName", "Repo", "RandomIDs", "Weeks"]
data = pd.DataFrame(columns=columnsToKeep)

for i, project in enumerate(week_mappings):
	if(i in [k for k in range(62, 63)]):
		# print(i, project["repo_name"])
		weeks = []
		oldClasses = pd.DataFrame(columns=columnsToKeep)

		for key in project.keys():
			if key == "repo_name":
				continue
			week = int(key.split("week_")[1])
			weeks.append(week)

		if hasMissingWeeks(weeks):
			print("\n\n")
			print(project["repo_name"])
			exit()

		weeks.sort()
		print("Total Weeks:", len(weeks))
		dropped = 0
		lastValidWeek = -1
		for ind, week in enumerate(weeks):
			# print(week)
			randomID = project["week_" + f'{week:04d}']["randomID"]
			if os.path.exists(os.path.join(resultsPath, randomID, "java")):
				for f in os.listdir(os.path.join(resultsPath, randomID, "java")):
					for file in os.listdir(os.path.join(resultsPath, randomID, "java", f)):
						if file.endswith("-Class.csv"):
							newClasses = pd.read_csv(os.path.join(resultsPath, randomID, "java", f, file))
							newClasses.reset_index()
							oldClasses.reset_index()
							oldClasses = oldClasses.drop_duplicates(subset=["Name", "LongName"])
							if (len(oldClasses.index) > 50000):
								print(len(oldClasses.index))
								oldClasses = oldClasses.merge(newClasses, how="left", on=["Name", "LongName"], indicator=True)
							else:
								oldClasses = oldClasses.merge(newClasses, how="outer", on=["Name", "LongName"], indicator=True)
							oldClasses["Repo"] = project["repo_name"]
							oldClasses["RandomIDs"] = oldClasses["RandomIDs"].fillna("").apply(list)
							oldClasses.loc[oldClasses._merge != "left_only", "RandomIDs"] = oldClasses.loc[oldClasses._merge != "left_only", "RandomIDs"].apply(lambda x: x + [randomID], 1)
							oldClasses["Weeks"] = oldClasses["Weeks"].fillna("").apply(list)
							oldClasses.loc[oldClasses._merge != "left_only", "Weeks"] = oldClasses.loc[oldClasses._merge != "left_only", "Weeks"].apply(lambda x: x + [week], 1)
							oldClasses = oldClasses[columnsToKeep]
							lastValidWeek = week
			else:
				dropped += 1
		print(project["repo_name"], "Missing:", str(dropped))
		oldClasses["StartAt"] = oldClasses.apply(lambda row: row["Weeks"][0], 1)
		oldClasses["DropAt"] = oldClasses.apply(lambda row: row["Weeks"][-1] if (len(row["Weeks"]) > 0) else -1, 1)
		oldClasses["DropAt"] = oldClasses["DropAt"].apply(lambda x: -1 if int(x) == lastValidWeek else x)
		
		data = data.append(oldClasses, ignore_index=True)

# data.to_csv(os.path.join("gen_data/antlr/", "data_dictionary_antlr_antlr4.csv"), index=False)
