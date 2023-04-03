# System Development for Prediction of Static Analysis Metrics of Source Code


## ** Abstract **

Nowadays, the pace of technological development and the uninterrupted use of online
sources have resulted in rapid software development processes. In the numerous projects
that are constantly being implemented, what is becoming more and more evident is the
smooth development of the source code around a number of criteria that define its quality.
This problem is especially magnified when there are dilemmas of reusing snippets of code
and doubts arise about the best choice. Therefore, such issues make it necessary to assess
the quality of code fragments that are candidates for reuse based on their maintainability.
The process of evaluating the quality of the source code of a software project is a time-
consuming and costly operation, as it involves a high degree of complexity depending on the languages the project has been implemented in and its scope. At the same time, the
contribution of many developers to a project always increases the difficulties of correct
evaluation. Consequently, such issues require techniques and tools that take into account
multiple parameters in order to rigorously and reasonably assess the quality of a project
in terms of its maintainability.
One of the most prevalent techniques is the analysis of source code using static metrics
that rigorously evaluate the characteristics of the software project. This is what this thesis
is based on, as it focuses on static source code analysis and methods that will contribute
to the evaluation of software quality through them. This is done as long as there is the
availability of open source projects in repositories such as GitHub and can be used to
build tools aiming at solving the issue. The static analysis of open-source projects through
metrics is the basis of the system designed in the current thesis.
The system designed and built in this thesis is to create a reliable and functionally
useful process that is capable of predicting the future values of static analysis metrics.
This tool essentially attempts to detect patterns of behavior of static metrics for past
metrics and predict similar behavior in the future using memory. This is implemented
using LSTM and GRU networks as their architecture focuses on holding information in
memory for long periods of time. Since software production is growing at an increasingly
rapid pace, such a tool will be a key element in a smooth and well-guided source code
development path.

Louis Meta
Electrical & Computer Engineering Department,
Aristotle University of Thessaloniki, Greece
March 2023


## **Code**

### Install required libraries

`pip install -r requirements.txt`

### Process

Run each file from s1-s5.

### Context

S1: File used from Thomas Karanikiotis for dictionary creation. Based on AnalysisResults which contains all the static metrics analysis from the repos.

**Note:** AnalysisResults folder is not included in the repo, due to size limitations.

S2: File used from Thomas Karanikiotis for datasets creation. 

S3: Part 1 of preprocessing divides the classes into dropped and non-dropped ones.

S4: Part 2 of preprocessing contains all the preprocessing steps and useful graphs calculation for the thesis report.

S5: This file contains the following: 
        - all the implemented models (lstm_v1, lstm_v2, lstm_v3, gru)
        - functions that are being used for the neurons analysis for each model
        - functions that are being used for the look back (window) analysis for each model
        - functions that are being used for the final calculations and the results of the thesis



**Note:** The files s1-s4 are responsible for generating the datasets that are saved in the "gen_data" folder. This means you don't have to run these files if you only want to execute the code. To run each task from the s5 analysis, simply uncomment the appropriate section of the file and then execute the code.