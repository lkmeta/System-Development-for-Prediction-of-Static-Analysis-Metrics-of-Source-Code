# create models lstm_v1, lstm_v2, lstm_v3 and gru
# also create all functions that will be used for the analysis
# calculate rmse for each model for number of neurons in each case  


from itertools import groupby
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from itertools import groupby
from keras.callbacks import EarlyStopping
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout

complexity = ["NL", "NLE", "McCC"]
cohesion = ["LCOM5"]
inheritance = ["DIT", "NOC", "NOP", "NOA", "NOD"] 
coupling = ["CBO", "NOI", "NII", "RFC"]

metricsToKeep = [metric + "_list" for metric in complexity + cohesion + inheritance + coupling]


def metric_data_to_list(data, metric):
    metric_list = []
    for i in range(len(data)):
        temp_list = list(map(float, data[metric][i][1:-1].split(', ') ))
        metric_list.append(temp_list)
    return metric_list

# Function to check if all elements in a list are equal
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# Calculate test size
# according to common logic
# cannot predict more than 2*look_back
# this function is not being used in the final version
def calculate_test_size(len_dataset, look_back):

    pred_size = 2*look_back
    
    if len_dataset - pred_size - look_back > int(len_dataset * 0.6):
        test_size = pred_size + look_back
        train_size = len_dataset - test_size
    else:
        train_size = int(len_dataset * 0.6)
        test_size = len_dataset - train_size
        pred_size = test_size - look_back
        
    return train_size, test_size



# Function to smooth data for better training
def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


# model predict process
def model_predict(model, pred_size, input_seq, look_back, test, scaler):

    predictions = np.empty((0,1))
    temp_seq = input_seq
    input_seq = np.reshape(input_seq, (input_seq.shape[1], look_back, 1))

    for i in range(pred_size):
        new_pred = model.predict(input_seq)
        # model.reset_states()
        predictions = np.append(predictions, new_pred)
        temp_seq = np.append(temp_seq, new_pred)
        input_seq = temp_seq[-look_back:]
        input_seq = input_seq.reshape((1, look_back, 1))


    predictions = predictions.reshape(-1, 1)

    # invert predictions
    test = test.reshape(1, -1)
    test = scaler.inverse_transform(test)
    # input_seq = input_seq.reshape(1, -1)
    # input = scaler.inverse_transform(input_seq)
    final_predictions = scaler.inverse_transform(predictions)

    testScore = np.sqrt(mean_squared_error(test[0, look_back:], final_predictions[:,0]))

    print(model.name)
    print('Test Score: %.2f RMSE' % (testScore))

    return final_predictions, testScore



# lstm_v1 model with window
def lstm_v1(data, metric="LOC_list", component=0, look_back=1, train_size_default=0.6):

    # data process before modeling
    # dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process(data, metric, component, look_back)


    best_smooth_weight = 0.8
    best_peak_duration = 8
    best_perc = 1.2

    # data process before modeling
    dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process_smooth(data, metric, component, look_back, best_smooth_weight, best_peak_duration, best_perc, train_size_default)


    # create and fit the LSTM network
    batch_size = 1
    model = Sequential(name="LSTM_v1_model") #_v1_m_"+ metric + "_c_" + str(component) + "_lb_" + str(look_back))
    model.add(LSTM(16, batch_input_shape=(batch_size, look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    
    # fit model
    start = time.time()
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=1, validation_data=(testX, testY), callbacks=[es])
    end = time.time()

    # make predictions and calculate RMSE
    predictions, score = model_predict(model, test_size - look_back, testX[0], look_back, test, scaler)

    duration = (1000.0*(end-start))
    print('Fit duration: %.6f ms\n\n' % (1000.0*(end-start)))

    dataset = scaler.inverse_transform(dataset)

    # shift input for plotting
    input = scaler.inverse_transform(testX[0])
    inputPlot = np.empty_like(dataset)
    inputPlot[:, :] = np.nan
    inputPlot[-look_back-len(predictions):-len(predictions), :] = input

    # shift test predictions for plotting
    predictPlot = np.empty_like(dataset)
    predictPlot[:, :] = np.nan
    predictPlot[-len(predictions):, :] = predictions


    return model.name, dataset, inputPlot, predictPlot, score, duration



# lstm_v1 model with window and neurons as parameter
def lstm_v1_n(data, metric="LOC_list", component=0, look_back=1, train_size_default=0.6, neurons=16):

    # data process before modeling
    # dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process(data, metric, component, look_back)


    best_smooth_weight = 0.8
    best_peak_duration = 8
    best_perc = 1.2

    # data process before modeling
    dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process_smooth(data, metric, component, look_back, best_smooth_weight, best_peak_duration, best_perc, train_size_default)


    # create and fit the LSTM network
    batch_size = 1
    model = Sequential(name="LSTM_v1_model") #_v1_m_"+ metric + "_c_" + str(component) + "_lb_" + str(look_back))
    model.add(LSTM(neurons, batch_input_shape=(batch_size, look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    
    # fit model
    start = time.time()
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=1, validation_data=(testX, testY), callbacks=[es])
    end = time.time()

    # make predictions and calculate RMSE
    predictions, score = model_predict(model, test_size - look_back, testX[0], look_back, test, scaler)

    print('Fit duration: %.6f ms\n\n' % (1000.0*(end-start)))

    dataset = scaler.inverse_transform(dataset)

    # shift input for plotting
    input = scaler.inverse_transform(testX[0])
    inputPlot = np.empty_like(dataset)
    inputPlot[:, :] = np.nan
    inputPlot[-look_back-len(predictions):-len(predictions), :] = input

    # shift test predictions for plotting
    predictPlot = np.empty_like(dataset)
    predictPlot[:, :] = np.nan
    predictPlot[-len(predictions):, :] = predictions


    return model.name, dataset, inputPlot, predictPlot, score


# lstm_v2 model with memory and 1 layer
def lstm_v2(data, metric="LOC_list", component=0, look_back=1, train_size_default=0.6):

    # data process before modeling
    # dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process(data, metric, component, look_back)

    best_smooth_weight = 0.8
    best_peak_duration = 8
    best_perc = 1.2

    # data process before modeling
    dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process_smooth(data, metric, component, look_back, best_smooth_weight, best_peak_duration, best_perc, train_size_default)

    # create and fit the LSTM network
    batch_size = 1
    model = Sequential(name="LSTM_v2_model") #_v2_m_"+ metric + "_c_" + str(component) + "_lb_" + str(look_back))
    model.add(LSTM(16, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)

    # fit model
    start = time.time()
    for i in range(100):
        history = model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=1, shuffle=False, validation_data=(testX, testY), callbacks=[es])
        model.reset_states()
        if history.history['val_loss'][-1] < 0.0001:
            print("Early stopping, epoch: ", i)
            break
    end = time.time()

    # make predictions and calculate RMSE
    predictions, score = model_predict(model, test_size - look_back, testX[0], look_back, test, scaler)
    
    duration = (1000.0*(end-start))
    print('Fit duration: %.6f ms\n\n' % (1000.0*(end-start)))

    dataset = scaler.inverse_transform(dataset)

    # shift input for plotting
    input = scaler.inverse_transform(testX[0])
    inputPlot = np.empty_like(dataset)
    inputPlot[:, :] = np.nan
    inputPlot[-look_back-len(predictions):-len(predictions), :] = input

    # shift test predictions for plotting
    predictPlot = np.empty_like(dataset)
    predictPlot[:, :] = np.nan
    predictPlot[-len(predictions):, :] = predictions


    return model.name, dataset, inputPlot, predictPlot, score, duration



# lstm_v2 model with memory and 1 layer and neurons as parameter
def lstm_v2_n(data, metric="LOC_list", component=0, look_back=1, train_size_default=0.6, neurons=16):

    # data process before modeling
    # dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process(data, metric, component, look_back)

    best_smooth_weight = 0.8
    best_peak_duration = 8
    best_perc = 1.2

    # data process before modeling
    dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process_smooth(data, metric, component, look_back, best_smooth_weight, best_peak_duration, best_perc, train_size_default)

    # create and fit the LSTM network
    batch_size = 1
    model = Sequential(name="LSTM_v2_model") #_v2_m_"+ metric + "_c_" + str(component) + "_lb_" + str(look_back))
    model.add(LSTM(neurons, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)

    # fit model
    start = time.time()
    for i in range(100):
        history = model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=0, shuffle=False, validation_data=(testX, testY), callbacks=[es])
        model.reset_states()
        if history.history['val_loss'][-1] < 0.0001:
            print("Early stopping, epoch: ", i)
            break
    end = time.time()

    # make predictions and calculate RMSE
    predictions, score = model_predict(model, test_size - look_back, testX[0], look_back, test, scaler)

    print('Fit duration: %.6f ms\n\n' % (1000.0*(end-start)))

    dataset = scaler.inverse_transform(dataset)

    # shift input for plotting
    input = scaler.inverse_transform(testX[0])
    inputPlot = np.empty_like(dataset)
    inputPlot[:, :] = np.nan
    inputPlot[-look_back-len(predictions):-len(predictions), :] = input

    # shift test predictions for plotting
    predictPlot = np.empty_like(dataset)
    predictPlot[:, :] = np.nan
    predictPlot[-len(predictions):, :] = predictions


    return model.name, dataset, inputPlot, predictPlot, score



# lstm_v3 model with memory and 2 layers
def lstm_v3(data, metric="LOC_list", component=0, look_back=1, train_size_default=0.6):

    # data process before modeling
    # dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process(data, metric, component, look_back)

    best_smooth_weight = 0.8
    best_peak_duration = 8
    best_perc = 1.2

    # data process before modeling
    dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process_smooth(data, metric, component, look_back, best_smooth_weight, best_peak_duration, best_perc, train_size_default)


    # create and fit the LSTM network
    batch_size = 1
    model = Sequential(name="LSTM_v3_model") #_v3_m_"+ metric + "_c_" + str(component) + "_lb_" + str(look_back))
    model.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(16, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    # model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)

    # fit model
    start = time.time()
    for i in range(100):
        history = model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=1, shuffle=False, validation_data=(testX, testY), callbacks=[es])
        model.reset_states()
        if history.history['val_loss'][-1] < 0.0001:
            print("Early stopping, epoch: ", i)
            break
    end = time.time()

    # make predictions and calculate RMSE
    predictions, score = model_predict(model, test_size - look_back, testX[0], look_back, test, scaler)

    duration = (1000.0*(end-start))
    print('Fit duration: %.6f ms\n\n' % (1000.0*(end-start)))

    dataset = scaler.inverse_transform(dataset)

    # shift input for plotting
    input = scaler.inverse_transform(testX[0])
    inputPlot = np.empty_like(dataset)
    inputPlot[:, :] = np.nan
    inputPlot[-look_back-len(predictions):-len(predictions), :] = input

    # shift test predictions for plotting
    predictPlot = np.empty_like(dataset)
    predictPlot[:, :] = np.nan
    predictPlot[-len(predictions):, :] = predictions


    return model.name, dataset, inputPlot, predictPlot, score



# gru model
def gru(data, metric="LOC_list", component=0, look_back=1, train_size_default=0.6):

    # data process before modeling
    # dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process(data, metric, component, look_back)

    best_smooth_weight = 0.8
    best_peak_duration = 8
    best_perc = 1.2

    # data process before modeling
    dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process_smooth(data, metric, component, look_back, best_smooth_weight, best_peak_duration, best_perc, train_size_default)


    # create and fit the LSTM network
    batch_size = 1
    model = Sequential(name="GRU_model") #_m_"+ metric + "_c_" + str(component) + "_lb_" + str(look_back))
    model.add(GRU(units=16, batch_input_shape=(batch_size, look_back, 1)))
    model.add(Dense(units=1))

    # Compiling the RNN
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    
    # fit model
    start = time.time()
    model.fit(trainX, trainY, epochs=100, batch_size=1, validation_data=(testX, testY), callbacks=[es])
    end = time.time()

    # make predictions and calculate RMSE
    predictions, score = model_predict(model, test_size - look_back, testX[0], look_back, test, scaler)

    duration = (1000.0*(end-start))
    print('Fit duration: %.6f ms\n\n' % (1000.0*(end-start)))

    dataset = scaler.inverse_transform(dataset)

    # shift input for plotting
    input = scaler.inverse_transform(testX[0])
    inputPlot = np.empty_like(dataset)
    inputPlot[:, :] = np.nan
    inputPlot[-look_back-len(predictions):-len(predictions), :] = input

    # shift test predictions for plotting
    predictPlot = np.empty_like(dataset)
    predictPlot[:, :] = np.nan
    predictPlot[-len(predictions):, :] = predictions


    return model.name, dataset, inputPlot, predictPlot, score, duration



# gru model with neurons parameter
def gru_n(data, metric="LOC_list", component=0, look_back=1, train_size_default=0.6, neurons=32):

    # data process before modeling
    # dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process(data, metric, component, look_back)

    best_smooth_weight = 0.8
    best_peak_duration = 8
    best_perc = 1.2

    # data process before modeling
    dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process_smooth(data, metric, component, look_back, best_smooth_weight, best_peak_duration, best_perc, train_size_default)


    # create and fit the LSTM network
    batch_size = 1
    model = Sequential(name="GRU_model") #_m_"+ metric + "_c_" + str(component) + "_lb_" + str(look_back))
    model.add(GRU(units=neurons, batch_input_shape=(batch_size, look_back, 1)))
    model.add(Dense(units=1))

    # Compiling the RNN
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    
    # fit model
    start = time.time()
    model.fit(trainX, trainY, epochs=100, batch_size=1, validation_data=(testX, testY), callbacks=[es])
    end = time.time()

    # make predictions and calculate RMSE
    predictions, score = model_predict(model, test_size - look_back, testX[0], look_back, test, scaler)

    print('Fit duration: %.6f ms\n\n' % (1000.0*(end-start)))

    dataset = scaler.inverse_transform(dataset)

    # shift input for plotting
    input = scaler.inverse_transform(testX[0])
    inputPlot = np.empty_like(dataset)
    inputPlot[:, :] = np.nan
    inputPlot[-look_back-len(predictions):-len(predictions), :] = input

    # shift test predictions for plotting
    predictPlot = np.empty_like(dataset)
    predictPlot[:, :] = np.nan
    predictPlot[-len(predictions):, :] = predictions


    return model.name, dataset, inputPlot, predictPlot, score



# # lstm model with memory and 2 layers smooth version
# def lstm_v3_smooth(data, category, metric="LOC_list", component=0, look_back=1, neuron=16, smooth_weight=0.5, train_size_default=0.5):

#     # best_smooth_weight, best_peak_duration, best_perc = find_best_parameters(data, component, category)


#     best_smooth_weight = 0.8
#     best_peak_duration = 8
#     best_perc = 1.2

#     # data process before modeling
#     dataset, trainX, trainY, test, testX, testY, scaler, test_size = data_process_smooth(data, metric, component, look_back, best_smooth_weight, best_peak_duration, best_perc, train_size_default)

#     # create and fit the LSTM network
#     batch_size = 1
#     model = Sequential(name="LSTM_model_v3_smooth_m_"+ metric + "_c_" + str(component) + "_lb_" + str(look_back))
#     model.add(LSTM(neuron, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
#     model.add(LSTM(8, batch_input_shape=(batch_size, look_back, 1), stateful=True))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     model.summary()

#     # patient early stopping
#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)

#     # fit model
#     start = time.time()
#     for i in range(100):
#         history = model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=1, shuffle=False, validation_data=(testX, testY), callbacks=[es])
#         model.reset_states()
#         if history.history['val_loss'][-1] < 0.0001:
#             print("Early stopping, epoch: ", i)
#             break
#     end = time.time()

#     # make predictions and calculate RMSE
#     predictions, score = model_predict(model, test_size - look_back, testX[0], look_back, test, scaler)

#     print('Fit duration: %.6f ms\n\n' % (1000.0*(end-start)))

#     dataset = scaler.inverse_transform(dataset)

#     # shift input for plotting
#     input = scaler.inverse_transform(testX[0])
#     inputPlot = np.empty_like(dataset)
#     inputPlot[:, :] = np.nan
#     inputPlot[-look_back-len(predictions):-len(predictions), :] = input

#     # shift test predictions for plotting
#     predictPlot = np.empty_like(dataset)
#     predictPlot[:, :] = np.nan
#     predictPlot[-len(predictions):, :] = predictions


#     return model.name, dataset, inputPlot, predictPlot, score



# lstm_v3 model with memory and 2 layers smooth version for category data 
def lstm_v3_category(data, category_name, category, weights, component=0, look_back=1, smooth_weight=0.5, train_size_default=0.5):

    # used fixed values for best parameters
    best_smooth_weight = 0.8
    best_peak_duration = 8
    best_perc = 1.2

    # data process before modeling
    dataset, trainX, trainY, test, testX, testY, scaler, test_size = category_aggregation(data, category, component, look_back, best_smooth_weight, best_peak_duration, best_perc, weights, train_size_default)

    # create and fit the LSTM network
    batch_size = 1
    model = Sequential(name="LSTM_model_v3_smooth_c_"+ category_name + "_c_" + str(component) + "_lb_" + str(look_back))
    model.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(16, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)

    # fit model
    start = time.time()
    for i in range(100):
        history = model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=1, shuffle=False, validation_data=(testX, testY), callbacks=[es])
        model.reset_states()
        if history.history['val_loss'][-1] < 0.0001:
            print("Early stopping, epoch: ", i)
            break
    end = time.time()

    # make predictions and calculate RMSE
    predictions, score = model_predict(model, test_size - look_back, testX[0], look_back, test, scaler)

    print('Fit duration: %.6f ms\n\n' % (1000.0*(end-start)))

    dataset = scaler.inverse_transform(dataset)

    # shift input for plotting
    input = scaler.inverse_transform(testX[0])
    inputPlot = np.empty_like(dataset)
    inputPlot[:, :] = np.nan
    inputPlot[-look_back-len(predictions):-len(predictions), :] = input

    # shift test predictions for plotting
    predictPlot = np.empty_like(dataset)
    predictPlot[:, :] = np.nan
    predictPlot[-len(predictions):, :] = predictions

    return model.name, dataset, inputPlot, predictPlot, score



# lstm_v3 1 graph 2 methods for category prediction
# first method uses category data from agragated metrics
# second method uses predictions from each metric and then aggregate them
def category_graphs_1graph(data, category_name, category, weights, component=0, best_lb=1, neurons=[16], smooth_weight=0.5):
    
    dict = {}

    # Category predictions method 1
    cat_name, cat_dataset, cat_input, cat_pred, cat_score = lstm_v3_category(data, category_name, category, weights, component, best_lb, smooth_weight)

    dict[cat_name + '_method1'] = [cat_score]

    # # Metrics predictions
    # predictions = []
    # scores = []

    # # Category predictions method 2
    # for metric in category:

    #     # Find metric name
    #     metric_name = metric + "_list"

    #     # LSTM v3 model with smooth data for training
    #     v3s_name, v3s_dataset, v3s_input, v3s_pred, v3s_score = lstm_v3_smooth(data, category, metric_name, component, best_lb, neurons, smooth_weight)

    #     # weight the predictions
    #     predictions.append(v3s_pred*weights[category.index(metric)])
    #     scores.append(v3s_score)

    # predictions = np.array(predictions).mean(axis=0)

    # dict[cat_name + '_method2'] = [np.array(scores).mean()]

    # plot baseline and predictions
    plt.figure(figsize=(30,16))
    plt.plot(cat_dataset, color='brown', label="dataset")
    plt.plot(cat_input, color='blue', label="input")
    plt.plot(cat_pred, color='red', label="catMethod1Pred", marker = '*')
    # plt.plot(predictions, color='green', label="catMethod2Pred", marker = ".")
    plt.legend(loc="upper left")
    # plt.title("LSTM_v3 for and Method2 comparison for " + category_name + " component " + str(component))
    plt.xlabel("Weeks")
    plt.ylabel(category_name)

    graph_name = "lstm_v3_best_" + category_name + "_c_" + str(component)

    # plt.show()
    # plt_folder = "models/1g_category/"

    # plt_name = plt_folder + graph_name + ".png"
    # plt.savefig(plt_name)
    # plt.close()

    return graph_name, dict


# lstm_v3 1 graph 2 methods for category prediction
def category_graphs_1cat(data, category_name, category, weights, component=0, best_lb=1, smooth_weight=0.5, train_size_default=0.5):
    
    dict = {}

    # print("Category: " + category_name + " component: " + str(component) + " best_lb: " + str(best_lb) + " neurons: " + str(neurons) + " smooth_weight: " + str(smooth_weight))

    # Category predictions method 1
    cat_name, cat_dataset, cat_input, cat_pred, cat_score = lstm_v3_category(data, category_name, category, weights, component, best_lb, smooth_weight, train_size_default)


    dict[cat_name + '_method1'] = [cat_score]

    # # Metrics predictions
    # predictions = []
    # scores = []

    # # Category predictions method 2
    # for metric in category:

    #     # Find metric name
    #     metric_name = metric + "_list"

    #     # LSTM v3 model with smooth data for training
    #     v3s_name, v3s_dataset, v3s_input, v3s_pred, v3s_score = lstm_v3_smooth(data, category, metric_name, component, best_lb, neurons, smooth_weight, train_size_default)

    #     # weight the predictions
    #     predictions.append(v3s_pred*weights[category.index(metric)])
    #     scores.append(v3s_score)


    # predictions = np.array(predictions).mean(axis=0)

    # dict[cat_name + '_method2'] = [np.array(scores).mean()]

    return dict, cat_dataset, cat_input, cat_pred




# lstm_v3 for category prediction
# compared with average from category metrics predictions
def category_graphs_4graphs(data, category_names, categories, component=0, best_lb=1, smooth_weight=0.5, train_size_default=0.5, lb_size = 0.5, counter_folder = 1):
    
    # plot baseline and predictions
    plt.figure(figsize=(30,16))
    plt.subplots_adjust(hspace=0.4)
    train_size = int((train_size_default + lb_size) * 100)
    # suptitle = "Plot dropped component " + str(component) + " predictions for all the categories with train_data: 50% and lb: 20% from dataset."
    # plt.suptitle(suptitle, fontsize=15, y=0.95)
    
    final_dict = {}

    for i in range(len(categories)):
        category = categories[i]
        category_name = category_names[categories.index(category)]

        weights = [1 for i in range(len(category))]

        # Category predictions
        dict, cat_dataset, cat_input, cat_pred = category_graphs_1cat(data, category_name, category, weights, component, best_lb, smooth_weight, train_size_default)

        final_dict.update(dict)

        dict_name = category_name + "_c_" + str(component)
        final_dict[dict_name] = dict

        # add a new subplot iteratively
        ax = plt.subplot(2, 2, i+1)
        ax.plot(cat_dataset, color='brown', label="dataset")
        ax.plot(cat_input, color='blue', label="input")
        ax.plot(cat_pred, color='green', label="predictions")
        ax.set_ylabel(category_name)
        ax.set_xlabel("Weeks")
        ax.legend(loc="upper left")


    # plt_folder = "results/FINAL/v3_best/"

    name = str(counter_folder) + "_lstm_graph_" + str(component) + "_train_" + str(train_size) + "_lb_" + str(best_lb)

    # plt_name = plt_folder + name + ".png"
    # plt.savefig(plt_name)
    # plt.close()

    return name, final_dict


# plot all models in one graph to compare the results
def category_graphs_all_models(counter, data, functions, metric, component=0, best_lb=1, train_size_default=0.5, lb_size = 0.5):
    
    # plot baseline and predictions
    plt.figure(figsize=(30,16))
    plt.subplots_adjust(hspace=0.8)
    # weeks = metric_data_to_list(data, "Weeks")
    # dataset_size = len(weeks[component])
    # train_size = int(dataset_size * 0.5)
    train_size = int((train_size_default + lb_size) * 100)
    suptitle = "Plot dropped component " + str(component) + " predictions for all models with train_data:" + str(train_size) + "% from dataset and lb:" +  str(best_lb)
    plt.suptitle(suptitle, fontsize=15, y=0.95)
    
    dict = {}

    for i in range(len(functions)):
        function = functions[i]
        function_name = function.__name__

        # Category predictions
        name, dataset, inputPlot, predictPlot, score = function(data, metric, component, best_lb, train_size_default)

        dict[name] = score

        # add a new subplot iteratively
        ax = plt.subplot(2, 2, i+1)
        ax.plot(dataset, color='brown', label="dataset")
        ax.plot(inputPlot, color='blue', label="input")
        ax.plot(predictPlot, color='red', label="predictions")
        ax.set_title(name)
        ax.set_ylabel(metric)
        ax.set_xlabel("Weeks")
        ax.legend(loc="upper left")



    # plt_folder = "results/test/"

    name = str(counter) + "_class_graph_" + str(component) + "_metric_" + metric + "_train_" + str(train_size) + "_lb_" + str(best_lb)

    # plt_name = plt_folder + name + ".png"
    # plt.savefig(plt_name)
    # plt.close()

    return name, dict



# plot all cases for neurons parameter in one graph to compare the results
def category_graphs_6(counter, data, function, metric, neurons, component=0, best_lb=1, train_size_default=0.5, lb_size = 0.5):
    
    # plot baseline and predictions
    plt.figure(figsize=(30,16))
    plt.subplots_adjust(hspace=0.5)
    train_size = int((train_size_default + lb_size) * 100)
    suptitle = "Plot dropped component " + str(component) + " predictions for gru with train_data:" + str(train_size) + "% from dataset and lb:" +  str(best_lb)
    plt.suptitle(suptitle, fontsize=15, y=0.95)
    
    dict = {}

    for i in range(len(neurons)):


        # Category predictions
        name, dataset, inputPlot, predictPlot, score = function(data, metric, component, best_lb, train_size_default, neurons[i])

        dict[name + "_" + str(neurons[i])] = score

        # add a new subplot iteratively
        ax = plt.subplot(3, 2, i+1)
        ax.plot(dataset, color='brown', label="dataset")
        ax.plot(inputPlot, color='blue', label="input")
        ax.plot(predictPlot, color='red', label="predictions")
        ax.set_title(name + " with " + str(neurons[i]) + " neurons")
        # ax.set_ylabel(metric)
        ax.set_xlabel("Weeks")
        ax.legend(loc="upper left")



    # plt_folder = "results/gru/"

    name = str(counter) + "_class_graph_" + str(component) + "_metric_" + metric + "_train_" + str(train_size) + "_lb_" + str(best_lb)

    # plt_name = plt_folder + name + ".png"
    # plt.savefig(plt_name)
    # plt.close()

    return name, dict



# plot all cases for look back parameter in one graph to compare the results
def category_graphs_lb(counter, data, function, metric, component=0, lb_sizes=1, train_size_default=0.5):
    
    # plot baseline and predictions
    plt.figure(figsize=(30,16))
    plt.subplots_adjust(hspace=0.5)
    # train_size = int((train_size_default + lb_size) * 100)
    suptitle = "Plot dropped component " + str(component) + " predictions for lstm_v3 with look back analysis" 
    plt.suptitle(suptitle, fontsize=15, y=0.95)
    
    dict = {}

    for i in range(len(lb_sizes)):


        look_back_value = int(lb_sizes[i]*len(weeks[component]))

        if look_back_value == 0:
            look_back_value = 1
            

        train_size = int((train_size_default + lb_sizes[i]) * 100)

        # Category predictions
        name, dataset, inputPlot, predictPlot, score = function(data, metric, component, look_back_value, train_size_default) 
        

        dict[name + "_" + str(lb_sizes[i])] = score

        # add a new subplot iteratively
        ax = plt.subplot(3, 2, i+1)
        ax.plot(dataset, color='brown', label="dataset")
        ax.plot(inputPlot, color='blue', label="input")
        ax.plot(predictPlot, color='red', label="predictions")
        # ax.set_title(name + " with " + str(neurons[i]) + " neurons")
        ax.set_title("Look Back: " + str(int(lb_sizes[i] * 100)) + "% of dataset")
        # ax.set_ylabel(metric)
        ax.set_xlabel("Weeks")
        ax.legend(loc="upper left")



    # plt_folder = "results/lstm_v3/"

    name = str(counter) + "_class_graph_" + str(component) + "_metric_" + metric + "_train_" + str(train_size) + "_lb_" + str(look_back_value)

    # plt_name = plt_folder + name + ".png"
    # plt.savefig(plt_name)
    # plt.close()

    return name, dict




# create both cases for look back and train size parameter in one graph to compare the results
# also save the results in a dictionary for table creation later
def category_graphs_lb_tr(counter, data, function, metric, component=0, lb_sizes=1, tr_sizes=0.5):
    
    # plot baseline and predictions
    plt.figure(figsize=(30,16))
    plt.subplots_adjust(hspace=0.4)
    # train_size = int((train_size_default + lb_size) * 100)
    suptitle = "Plot dropped component " + str(component) + " predictions analysis for metric " + metric
    plt.suptitle(suptitle, fontsize=15, y=0.95)
    
    dict = {}

    for j in range(len(tr_sizes)):


        for i in range(len(lb_sizes)):

            train_size_default = tr_sizes[j]

            look_back_value = int(lb_sizes[i]*len(weeks[component]))

            if look_back_value == 0:
                look_back_value = 1
                

            train_size = int((tr_sizes[j] + lb_sizes[i]) * 100)


            print("Component: " + str(component) + " metric: " + str(metric) + " train_size: " + str(train_size) + " look_back: " + str(look_back_value))


            # Category predictions
            name, dataset, inputPlot, predictPlot, score, duration = function(data, metric, component, look_back_value, train_size_default) 
            

            dict[name + "_" + str(tr_sizes[j]) + "_" + str(lb_sizes[i])] = score, duration

            # add a new subplot iteratively
            ax = plt.subplot(3, 2, j*len(lb_sizes) + i+1)
            ax.plot(dataset, color='brown', label="dataset")
            ax.plot(inputPlot, color='blue', label="input")
            ax.plot(predictPlot, color='red', label="predictions")
            # ax.set_title(name + " with " + str(neurons[i]) + " neurons")
            # ax.set_title("Look Back: " + str(int(lb_sizes[i] * 100)) + "% of dataset")
            ax.set_title("train: " + str(int((tr_sizes[j])* 100)) + "% and lb: " + str(int(lb_sizes[i] * 100)) + "% of dataset")
            # ax.set_ylabel(metric)
            ax.set_xlabel("Weeks")
            ax.legend(loc="upper left")



    # plt_folder = "results/FINAL/gru/"

    name = str(counter) + "_class_graph_" + str(component) + "_metric_" + metric + "_train_" + str(train_size) + "_lb_" + str(look_back_value)

    # plt_name = plt_folder + name + ".png"
    # plt.savefig(plt_name)
    # plt.close()


    # output_file = "results/FINAL/gru/results.txt"
    # with open(output_file,'a') as file:
    #         file.write(str('\n')) 
    #         file.write(str(dict))  


    return name, dict




# create plots according to the best look back and train size parameters from previous analysis
def category_graphs_best(counter, data, function, category_names, categories, metric, component=0, lb_sizes=1, tr_sizes=0.5):
    
    # plot baseline and predictions
    plt.figure(figsize=(30,16))
    plt.subplots_adjust(hspace=0.4)
    # train_size = int((train_size_default + lb_size) * 100)
    suptitle = "Plot dropped component " + str(component) + " predictions analysis for metric " + metric
    plt.suptitle(suptitle, fontsize=15, y=0.95)
    
    dict = {}

    for i in range(len(categories)):
        category = categories[i]
        category_name = category_names[categories.index(category)]

        weights = [1 for i in range(len(category))]

        for i in range(len(lb_sizes)):

            train_size_default = tr_sizes[j]

            look_back_value = int(lb_sizes[i]*len(weeks[component]))

            if look_back_value == 0:
                look_back_value = 1
                

            train_size = int((tr_sizes[j] + lb_sizes[i]) * 100)


            print("Component: " + str(component) + " metric: " + str(metric) + " train_size: " + str(train_size) + " look_back: " + str(look_back_value))


            # Category predictions
            name, dataset, inputPlot, predictPlot, score, duration = function(data, metric, component, look_back_value, train_size_default) 
            

            dict[name + "_" + str(tr_sizes[j]) + "_" + str(lb_sizes[i])] = score, duration

            # add a new subplot iteratively
            ax = plt.subplot(3, 2, j*len(lb_sizes) + i+1)
            ax.plot(dataset, color='brown', label="dataset")
            ax.plot(inputPlot, color='blue', label="input")
            ax.plot(predictPlot, color='red', label="predictions")
            # ax.set_title(name + " with " + str(neurons[i]) + " neurons")
            # ax.set_title("Look Back: " + str(int(lb_sizes[i] * 100)) + "% of dataset")
            ax.set_title("train: " + str(int((tr_sizes[j])* 100)) + "% and lb: " + str(int(lb_sizes[i] * 100)) + "% of dataset")
            # ax.set_ylabel(metric)
            ax.set_xlabel("Weeks")
            ax.legend(loc="upper left")



    # plt_folder = "results/FINAL/gru/"

    name = str(counter) + "_class_graph_" + str(component) + "_metric_" + metric + "_train_" + str(train_size) + "_lb_" + str(look_back_value)

    # plt_name = plt_folder + name + ".png"
    # plt.savefig(plt_name)
    # plt.close()


    # output_file = "results/FINAL/gru/results.txt"
    # with open(output_file,'a') as file:
    #         file.write(str('\n')) 
    #         file.write(str(dict))  


    return name, dict





# function remove peaks when data is too high or too low from previous weeks based on percentage 
# aslo apply smoothing process to dataset
# used for category values
def category_remove_peaks_from_dataset(dataset, component, category, smooth_weight, peak_duration, perc, weights):

    weigted_cat = []
    # weights = [1 for i in range(len(category))]
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


# function remove peaks when data is too high or too low from previous weeks based on percentage 
# aslo apply smoothing process to dataset
# used for metric values
def metric_remove_peaks_from_dataset(dataset, component, metric, smooth_weight, peak_duration, perc):

    # find dataset
    metric_list = metric_data_to_list(data, metric)
    dataset = np.array(metric_list[component])

    smooth_dataset = dataset

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

    return dataset, smooth_dataset

# category aggregation with weights 
# mean value for now for each metric in category
def category_aggregation(data, category=complexity, component=0, look_back=1, smooth_weight=0.7, peak_duration=8, perc=1.2, weights=[1], train_size_default = 0.5):

    # remove peaks from dataset and smooth it
    real_dataset, smooth_dataset = category_remove_peaks_from_dataset(data, component, category, smooth_weight, peak_duration, perc, weights)

    # reshape
    dataset = smooth_dataset.astype('float32').reshape(-1, 1)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * train_size_default)
    test_size = len(dataset) - train_size
    train_size_extended = train_size + look_back
    print("train_size: " + str(train_size))
    print("train_size_extended: " + str(train_size_extended))

    # train_size, test_size = calculate_test_size(len(dataset), look_back)
    train, test = dataset[0:train_size_extended,:], dataset[train_size:len(dataset),:]

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


    return dataset, trainX, trainY, test, testX, testY, scaler, test_size



# data process before modeling smooth version
# used for metric values
def data_process_smooth(data, metric="LOC_list", component=0, look_back=1, smooth_weight=0.5, peak_duration=8, perc=1.2, train_size_default = 0.5):

    # remove peaks from dataset and smooth it
    real_dataset, smooth_dataset = metric_remove_peaks_from_dataset(data, component, metric, smooth_weight, peak_duration, perc)

    # reshape
    dataset = smooth_dataset.astype('float32').reshape(-1, 1)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * train_size_default)
    test_size = len(dataset) - train_size
    train_size_extended = train_size + look_back

    # train_size, test_size = calculate_test_size(len(dataset), look_back)
    train, test = dataset[0:train_size_extended,:], dataset[train_size:len(dataset),:]

    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


    return dataset, trainX, trainY, test, testX, testY, scaler, test_size



######################################################

# Import needed data

projectPath = "/home/kosmas/Desktop/thesis/thesis_meta/"

csvPath = "./gen_data/antlr/data_dictionary_antlr_antlr4_completed_preprocessed_progress_3_non_straight"
data_antlr = pd.read_csv(projectPath + csvPath + ".csv")

csvPath = "./gen_data/grails/data_dictionary_grails_core_completed_preprocessed_progress_3_non_straight"
data_grail = pd.read_csv(csvPath + ".csv")

csvPath = "./gen_data/aws/data_dictionary_aws_sdk_java_completed_preprocessed_progress_3_non_straight"
data_aws = pd.read_csv(csvPath + ".csv")


data = pd.concat([data_antlr, data_grail, data_aws], ignore_index=True)

weeks = metric_data_to_list(data, "Weeks")


######################################################

#### ANALYSIS 1 ####

# # # Process to calculate the best neurons value for each function
# # # This also creates 6 graphs plots for each case
# # # v1 v2 v3 gru functions 


# # random choosed 25 components with non straight lines 
# random_components = [11, 17, 18, 57, 60, 62, 271, 272, 375, 378, 511, 538, 591, 667, 668, 881, 887, 901, 902, 970, 1002, 1011, 1049, 1085, 1093]

# # used function 
# function = gru_n

# # output_file = "results/gru/output.txt"
# model_dict = {}
# neurons = [4, 8, 16, 32, 64, 128]
# counter = 10

# for i in range(len(random_components)):
#     # metric = metrics[i] + "_list"
#     metric = random.choice(metricsToKeep)
#     component = random_components[i]
#     look_back_value = int(0.20*len(weeks[component]))
#     train_size_default = 0.50

#     print("Component: " + str(random_components[i]) + " Metric: " + metric)

#     counter += 1
#     name, dict = category_graphs_6(counter, data, function, metric, neurons, component, look_back_value, train_size_default, 0.20)

#     model_dict[name] = dict

#     # with open(output_file,'a') as file:
#     #         file.write(str('\n')) 
#     #         file.write(str(model_dict))  






######################################################

# # Process to find best num of neurons value for each function or look back value after the next task

# # importing the module
# import ast

# neurons = [4, 8, 16, 32, 64, 128]
# lb_sizes = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40]

# parameter = neurons

# # output_file = 'v2.txt'


# fun_dict = {}
# # open the file and read the content in a dictionary
# with open(output_file) as f:
#     for line in f:
#         d = ast.literal_eval(line)
#         fun_dict.update(d)


# # get all the look back values 
# list = []
# for key, value in fun_dict.items():
#     row_list = []
#     for key,value in value.items():
#         row_list.append(value)

#     if np.average(row_list) != 0:
#         list.extend(row_list)


# # average of all the look back values
# av_list = []
# for j in range(len(parameter)):
#     sum = 0
#     for i in range(int(len(list)/len(parameter))):
#         sum += list[i*len(parameter)+j]
#     av_list.append(sum/int(len(list)/len(parameter)))

# # print best look back value for each function
# best_neuron = parameter[av_list.index(min(av_list))]
# # print("Best neuron value: ", best_neuron)
# # print("RMSE Average list: ", av_list)








######################################################

#### ANALYSIS 2 ####

# # # Process to calculate the best look back value for each function
# # # This also creates 6 graphs plots for each case
# # # v1 v2 v3 gru functions 

# # random choosed 25 components with non straight lines 
# random_components = [11, 17, 18, 57, 60, 62, 271, 272, 375, 378, 511, 538, 591, 667, 668, 881, 887, 901, 902, 970, 1002, 1011, 1049, 1085, 1093]

# # used function
# function = lstm_v3

# # output_file = "results/lstm_v3/output.txt"
# model_dict = {}
# lb_sizes = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40]
# counter = 0

# for i in range(len(random_components)):
#     metric = random.choice(metricsToKeep)

#     component = random_components[i]
#     train_size_default = 0.50

#     print("Component: " + str(random_components[i]) + " Metric: " + metric)

#     counter += 1
#     name, dict = category_graphs_lb(counter, data, function, metric, component, lb_sizes, train_size_default)

#     model_dict[name] = dict

#     # with open(output_file,'a') as file:
#     #         file.write(str('\n')) 
#     #         file.write(str(model_dict))  




######################################################

#### ANALYSIS 3 ####

# # # Process to calculate the table with the best look back value for each function
# # # This also creates 6 graphs plots for each case
# # # v1 v2 v3 gru functions 


# # random choosed 25 components with non straight lines 
# random_components = [11, 17, 18, 57, 60, 62, 271, 272, 375, 378, 511, 538, 591, 667, 668, 881, 887, 901, 902, 970, 1002, 1011, 1049, 1085, 1093]

# metrics = ["McCC_list", "McCC_list", "McCC_list", "RFC_list", "RFC_list", "RFC_list", "LCOM5_list", "LCOM5_list", "NOD_list", "NOD_list", "NOI_list", "NOI_list"] + metricsToKeep

# # used function
# function = gru

# # output_file = "results/FINAL/gru_output.txt"
# model_dict = {}
# lb_sizes = [0.10, 0.15, 0.20]
# tr_sizes = [0.40, 0.50]
# counter = 0

# for i in range(len(random_components)):
#     metric = random.choice(metricsToKeep)

#     component = random_components[i]
#     print("Component: " + str(random_components[i]) + " Metric: " + metric)

#     counter += 1
#     name, dict = category_graphs_lb_tr(counter, data, function, metric, component, lb_sizes, tr_sizes)

#     model_dict[name] = dict

#     # with open(output_file,'a') as file:
#     #         file.write(str('\n')) 
#     #         file.write(str(model_dict))  




# ######################################################

# # Gather results from the output file and create a table 

# # importing the module
# import ast

# lb_sizes = [0.10, 0.15, 0.20]
# tr_sizes = [0.40, 0.50]

# parameter = lb_sizes

# output_file = 'v2.txt'


# fun_dict = {}
# # open the file and read the content in a dictionary
# with open(output_file) as f:
#     for line in f:
#         d = ast.literal_eval(line)
#         fun_dict.update(d)


# # get all the look back values 
# values_list = []
# duration_list = []
# for key, value in fun_dict.items():
#     row_list = []
#     duration_row_list = []
#     for key,value in value.items():
#         # print(key, value)
#         row_list.append(value[0])
#         duration_row_list.append(value[1])


#     if np.average(row_list) != 0:
#         values_list.extend(row_list)
#         duration_list.extend(duration_row_list)


# # average of all the look back values
# av_values_list = []
# av_duration_list = []
# for j in range(len(parameter)):
#     sum_values = 0
#     sum_duration = 0
#     for i in range(int(len(values_list)/len(parameter))):
#         sum_values += values_list[i*len(parameter)+j]
#         sum_duration += duration_list[i*len(parameter)+j]
#     av_values_list.append(sum_values/int(len(values_list)/len(parameter)))
#     av_duration_list.append(sum_duration/int(len(duration_list)/len(parameter)))
    

# # print best look back value for each function
# best_index = av_values_list.index(min(av_values_list))
# print("Best index value: ", best_index)
# print("RMSE Average list: ", av_values_list)
# print("Duration Average list: ", av_duration_list)


# # print 3 digits after the comma
# for i in range(len(av_values_list)):
#     print("%.3f" % av_values_list[i] + " & %.3f" % (av_duration_list[i] / 1000) + " \\\\")