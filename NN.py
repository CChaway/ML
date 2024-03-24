
# Imports & Random Seeds

import os
# Prevents using GPU, in order to reproduce results
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Import packages and set pseudorandom number seeds
import numpy as np
import random as rn
import tensorflow as tf

# Set seed value
seed = 1
# Set numpy random seed
np.random.seed(seed)
# Set random random seed
rn.seed(seed)
# Set tensorflow random seed
tf.random.set_seed(seed)

# Make results single-threaded, in order to reproduce results
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)

with tf.compat.v1.Session(config=config) as sess:
    pass

# Function to reset random seeds
def reset_random_seeds():
    # Reset the numpy random seed
    np.random.seed(seed)
    # Reset the random random seed
    rn.seed(seed)
    # Reset the tensorflow random seed
    tf.random.set_seed(seed)

# Import packages for data analysis, visualisation and time monitoring
import math
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm
import time
from pylab import plt, mpl

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
# %matplotlib inline

# Import Keras package used for building neural networks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.initializers import random_normal
from tensorflow.keras.callbacks import ModelCheckpoint

# Raw Data

# Read csv file into DataFrame
raw = pd.read_csv('TS.csv', sep=';', decimal=',', index_col=0, parse_dates=True, dayfirst=True).dropna()
# raw = pd.read_csv('TS.csv', index_col='Timestamp', parse_dates=True).dropna()
# Change the DataFrame to chronological order
raw = raw[::-1]

raw.info()
raw['ELGBDAHD'].plot(figsize=(16, 6));
plt.show()
# Data Statistics & Features Engineering

# Create data DataFrame using raw DataFrame
data = pd.DataFrame(raw)

# Log return of Last
data['r'] = np.log(data['ELGBDAHD'] / data['ELGBDAHD'].shift(1))

# Daily directional movement
data['d'] = np.where(data['r'] > 0, 1, 0)

# Delete data rows with nan values
data.dropna(inplace=True)

# Top 3 rows of data
data.head(3)

# Plot log returns histogram with normal distribution overlay
plt.figure(figsize=(16, 6))
# Log returns distribution
plt.hist(data['r'], bins=70, density=True,
         label='log returns', color='b')
# Name x axis
plt.xlabel('log returns')
# Name y axis
plt.ylabel('frequency')
# Add title
plt.title('Brent log returns distribution')
x = np.linspace(plt.axis()[0], plt.axis()[1])
# Normal distribution
plt.plot(x, scs.norm.pdf(x, data['r'].mean(), data['r'].std()),
         'r', lw=2.0, label='pdf')
# Display legend
plt.legend();

# Generate qq plot
sm.qqplot(data['r'], line='s')
# Add title
plt.title('Q-Q plot')
# Name x axis
plt.xlabel('theoretical quantiles')
# Name y axis
plt.ylabel('sample quantiles');

# Simple Moving Averages
# 2 week
data['sma14'] = data['ELGBDAHD'].rolling(14).mean()
# 2 month
data['sma60'] = data['ELGBDAHD'].rolling(60).mean()

# Rolling Volatility (annualized assuming 365 trading days)
# 2 week
data['vol14'] = data['r'].rolling(14).std() * np.sqrt(365)
# 2 month
data['vol60'] = data['r'].rolling(60).std() * np.sqrt(365)

# Bollinger Bands
# 20 day simple moving average
data['sma20'] = data['ELGBDAHD'].rolling(20).mean()
# Upper band
data['b_upper'] = data['sma20'] + 2 * data['sma20'].rolling(20).std()
# Lower band
data['b_lower'] = data['sma20'] - 2 * data['sma20'].rolling(20).std()

# Delete Volume column
# data.drop(['Volume', 'VWAP'], axis=1, inplace=True)

# Delete data rows with nan values
data.dropna(inplace=True)

# Top 3 rows of data
data.head(3)

# List of features in data set
features = list(data.columns)
# Features that will not be normalized
exclude = ['r', 'd', 'vol14', 'vol60']

def normalize(x, mu, std):
    return (x - mu) / std

# 80% train/test split
split = int(len(data) * 0.8)
# 15% validation partition
val_size = int(split * 0.15)
# Create train data set
train = data.iloc[:split]
# Calculate train data mean
mu = train.mean()
# Calculate train data standard deviation
std = train.std()

# Number of days lagged data to use for each sample
lags = 10

# Empty list to store feature column names
cols = []
# For each feature
for f in features:
    for lag in range(1, lags + 1):
        # Name column
        col = f'{f}_lag_{lag}'
        if f not in exclude:
            # Lag feature and normalize using mu and std previously calculated
            data[col] = normalize(data[f].shift(lag), mu[f], std[f])
        else:
            # Lag feature
            data[col] = data[f].shift(lag)
        # Append feature column name to cols
        cols.append(col)
# Drop nan values
data.dropna(inplace=True)

# Train data prior to validation split
train = data.iloc[:split].copy()
# Validation data
val = train[-val_size:]
# Train data after validation split
train = train[:-val_size]
# Test data
test = data.iloc[split:].copy()

# Top 3 rows of train
train.head(3)

# Training the Models

# Dense Neural Network Model

def build_dnn(name, neurons=64, hidden_layers=1, drop_pct=0):
    # Reset random seeds to reproduce results
    reset_random_seeds()
    # Name the model
    model = Sequential(name=name)

    # Hidden layer which specifies input shape
    model.add(Dense(neurons, activation='relu',
                    kernel_initializer=random_normal(seed=seed),
                    input_shape=(len(cols),)))

    # If specified, dropout layer to reduce overfitting
    if drop_pct > 0:
        model.add(Dropout(rate=drop_pct, seed=seed))

    # If specified, additional hidden layers added
    if hidden_layers > 1:
        for _ in range(1, hidden_layers):
            model.add(Dense(neurons, activation='relu',
                            kernel_initializer=random_normal(seed=seed)))
            # If specified, dropout layer to reduce overfitting
            if drop_pct > 0:
                model.add(Dropout(rate=drop_pct, seed=seed))

    # Final layer with 1 neuron uses sigmoid to output a scalar between 0 and 1
    model.add(Dense(1, activation='sigmoid',
                    kernel_initializer=random_normal(seed=seed)))
    # Model is compiled
    model.compile(optimizer='Adagrad',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Create 4 Dense Neural Networks (DNNs) with different hyperparameters
# 64 neurons, 2 hidden layers
DNN1 = build_dnn('DNN1', 64, 2, 0.4)
# 128 neurons, 2 hidden layers
DNN2 = build_dnn('DNN2', 128, 2, 0.4)
# 128 neurons, 3 hidden layers
DNN3 = build_dnn('DNN3', 128, 3, 0.4)
# 256 neurons, 4 hidden layers
DNN4 = build_dnn('DNN4', 256, 4, 0.4)

# Create a list of the 4 DNNs
DNN_models = [DNN1, DNN2, DNN3, DNN4]

# DNN Model Training

def fit(model, network, n_epochs, repeat):
    # Create empty DataFrame and lists
    sample_res = pd.DataFrame()
    mean_val_loss = []
    mean_val_acc = []
    mean_loss = []
    mean_acc = []

    # Callback to save weights where val acc is highest
    callbacks = [ModelCheckpoint(filepath=str(model.name + '.hdf5'),
                                 monitor='val_accuracy',
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='auto',
                                 period=1)]

    # Repeat fit process
    for i in range(1, repeat + 1):
        # Reset random seeds to reproduce results
        reset_random_seeds()
        # Check model type and then use appropriate input dimensions
        if network == 'DNN':
            # Use 2 dimensional input data
            model.fit(train[cols], train['d'],
                      epochs=n_epochs, batch_size=32, verbose=False,
                      validation_data=(val[cols], val['d']),
                      callbacks=callbacks, shuffle=False)
        elif network in ['RNN', 'CNN', 'CNN_RNN']:
            # Use 3 dimensional input data
            model.fit(train_3d, train['d'],
                      epochs=n_epochs, batch_size=32, verbose=False,
                      validation_data=(val_3d, val['d']),
                      callbacks=callbacks)

        # Store individual results
        individual_res = pd.DataFrame(model.history.history)
        individual_res.rename(columns={'val_loss': f'{i}_val_loss',
                                       'val_accuracy': f'{i}_val_acc',
                                       'loss': f'{i}_loss',
                                       'accuracy': f'{i}_acc'},
                              inplace=True)

        # Add column names to be used in sample mean calculations
        mean_val_loss.append(f'{i}_val_loss')
        mean_val_acc.append(f'{i}_val_acc')
        mean_loss.append(f'{i}_loss')
        mean_acc.append(f'{i}_acc')

        # Concatenate individual results with sample results
        sample_res = pd.concat([sample_res, individual_res], axis=1)

        if i % 10 == 0:
            print(f'{i}/{repeat} iterations of {n_epochs} epochs run')

    # Calculate mean values across all samples for val_loss, val_acc, loss, acc
    sample_res[f'{model.name}_mean_val_loss'] = sample_res[
        mean_val_loss].mean(axis=1)
    sample_res[f'{model.name}_mean_val_acc'] = sample_res[
        mean_val_acc].mean(axis=1)
    sample_res[f'{model.name}_mean_loss'] = sample_res[
        mean_loss].mean(axis=1)
    sample_res[f'{model.name}_mean_acc'] = sample_res[
        mean_acc].mean(axis=1)

    # Create DataFrame with mean results
    mean_res = pd.DataFrame(sample_res[[f'{model.name}_mean_val_loss',
                                        f'{model.name}_mean_val_acc',
                                        f'{model.name}_mean_loss',
                                        f'{model.name}_mean_acc']])

    return mean_res

def fit_models(network, models=[], n_epochs=40):

    # Empty DataFrame to store results
    res = pd.DataFrame()

    # Iterate through different hyperparameter tunings
    for model in models:
        print(f'Model: {model.name}')
        # Start time of model training
        start_time = time.time()
        results = fit(model, network, n_epochs, 1)
        # Print run time, maximum accuracy and final accuracy
        print('Run time: %s seconds' % round((time.time() - start_time), 2))
        print('Max Accuracy: '
              # + str(results[f'{model.name}_mean_val_acc'].max().round(4)))
              + str(round(results[f'{model.name}_mean_val_acc'].max())))
        print('Final Accuracy: '
              + str(round(results.iloc[-1][f'{model.name}_mean_val_acc'])))
        # Concatenate results to res DataFrame
        res = pd.concat([res, results], axis=1)
        print('=' * 33)

    return res

DNN_results = fit_models(network='DNN', models=DNN_models)

def plot_results(models, results):
    # Create figure and axes
    fig, axes = plt.subplots(nrows=math.ceil(len(models)/2),
                             ncols=2, figsize=(16, 10),
                             sharex=True, sharey=True)
    # Initial figure coordinates
    sub_rows = 0
    sub_cols = 0
    # Iterate through each model parameterisation
    for model in models:
        results[[f'{model.name}_mean_val_loss',
                 f'{model.name}_mean_val_acc',
                 f'{model.name}_mean_loss',
                 f'{model.name}_mean_acc']].plot(
            ax=axes[sub_rows, sub_cols],
            title=f'{model.name} fit results',
            style=['--', '--', '-', '-'])
        # Adjust figure coordinates
        sub_rows += sub_cols
        if sub_cols > 0:
            sub_cols = 0
        else:
            sub_cols = 1

import warnings; warnings.simplefilter('ignore')

# Plot the DNN training results
plot_results(DNN_models, DNN_results)

# Recurrent Neural Network Model

def build_rnn(name, neurons=64, hidden_layers=1, drop_pct=0):
    # Reset random seeds to reproduce results
    reset_random_seeds()
    # Type of neural network
    model = Sequential(name=name)

    # Hidden layer which specifies input shape
    model.add(SimpleRNN(neurons, activation='relu',
                        input_shape=(lags, len(features)),
                        return_sequences=True))

    # If specified, dropout layer to reduce overfitting
    if drop_pct > 0:
        model.add(Dropout(rate=drop_pct, seed=seed))

    # If specified, additional hidden layers added
    if hidden_layers > 2:
        for _ in range(2, hidden_layers):
            model.add(SimpleRNN(neurons, activation='relu',
                                return_sequences=True))
            # If specified, dropout layer to reduce overfitting
            if drop_pct > 0:
                model.add(Dropout(rate=drop_pct, seed=seed))
        model.add(SimpleRNN(neurons, activation='relu'))
        # If specified, dropout layer to reduce overfitting
        if drop_pct > 0:
            model.add(Dropout(rate=drop_pct, seed=seed))
    elif hidden_layers > 1:
        model.add(SimpleRNN(neurons, activation='relu'))
        # If specified, dropout layer to reduce overfitting
        if drop_pct > 0:
            model.add(Dropout(rate=drop_pct, seed=seed))

    # Final layer with 1 unit uses sigmoid to output a scalar between 0 and 1
    model.add(Dense(1, activation='sigmoid'))
    # Model is compiled
    model.compile(optimizer='Adagrad',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Create 4 Recurrent Neural Networks (RNNs) with different parameters
# 64 neurons, 2 hidden layers
RNN1 = build_rnn('RNN1', 64, 2, 0.4)
# 128 neurons, 2 hidden layers
RNN2 = build_rnn('RNN2', 128, 2, 0.4)
# 128 neurons, 3 hidden layers
RNN3 = build_rnn('RNN3', 128, 3, 0.4)
# 256 neurons, 4 hidden layers
RNN4 = build_rnn('RNN4', 256, 4, 0.4)

# Create a list of the 4 RNNs
RNN_models = [RNN1, RNN2, RNN3, RNN4]

# Create 4 Recurrent Neural Networks (RNNs) with different parameters
# 64 neurons, 2 hidden layers
RNN1 = build_rnn('RNN1', 64, 2, 0.4)
# 128 neurons, 2 hidden layers
RNN2 = build_rnn('RNN2', 128, 2, 0.4)
# 128 neurons, 3 hidden layers
RNN3 = build_rnn('RNN3', 128, 3, 0.4)
# 256 neurons, 4 hidden layers
RNN4 = build_rnn('RNN4', 256, 4, 0.4)

# Create a list of the 4 RNNs
RNN_models = [RNN1, RNN2, RNN3, RNN4]

train_3d = np.array(train[cols])
val_3d = np.array(val[cols])
test_3d = np.array(test[cols])

train_3d = train_3d.reshape(len(train), lags, len(features))
val_3d = val_3d.reshape(len(val), lags, len(features))
test_3d = test_3d.reshape(len(test), lags, len(features))

# RNN Model Training

RNN_results = fit_models(network='RNN', models=RNN_models)

# Plot the RNN training results
plot_results(RNN_models, RNN_results)

# Convolutional Neural Network Model

def build_cnn(name, filters=64, hidden_layers=1, drop_pct=0):
    # Reset random seeds to reproduce results
    reset_random_seeds()
    # Type of neural network
    model = Sequential(name=name)

    # Hidden layer which specifies input shape
    model.add(Conv1D(filters=filters, kernel_size=3, activation='relu',
                     input_shape=(lags, len(features))))

    # If specified, dropout layer to reduce overfitting
    if drop_pct > 0:
        model.add(Dropout(rate=drop_pct, seed=seed))

    # If specified, additional hidden layers added
    if hidden_layers > 2:
        for _ in range(2, hidden_layers):
            model.add(Conv1D(filters=filters, kernel_size=3,
                             activation='relu'))
            # If specified, dropout layer to reduce overfitting
            if drop_pct > 0:
                model.add(Dropout(rate=drop_pct, seed=seed))
        model.add(Conv1D(filters=filters, kernel_size=3, activation='relu'))
        # If specified, dropout layer to reduce overfitting
        if drop_pct > 0:
            model.add(Dropout(rate=drop_pct, seed=seed))
    elif hidden_layers > 1:
        model.add(Conv1D(filters=filters, kernel_size=3, activation='relu'))
        # If specified, dropout layer to reduce overfitting
        if drop_pct > 0:
            model.add(Dropout(rate=drop_pct, seed=seed))

    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    if drop_pct > 0:
        model.add(Dropout(rate=drop_pct, seed=seed))

    # Final layer with 1 unit uses sigmoid to output a scalar between 0 and 1
    model.add(Dense(1, activation='sigmoid'))
    # Model is compiled
    model.compile(optimizer='Adagrad',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Create 4 Convolutional Neural Networks (CNNs) with different parameters
# 64 neurons, 2 hidden layers
CNN1 = build_cnn('CNN1', 64, 2, 0.4)
# 128 neurons, 2 hidden layers
CNN2 = build_cnn('CNN2', 128, 2, 0.4)
# 128 neurons, 3 hidden layers
CNN3 = build_cnn('CNN3', 128, 3, 0.4)
# 256 neurons, 4 hidden layers
CNN4 = build_cnn('CNN4', 256, 4, 0.4)

# Create a list of the 4 RNNs
CNN_models = [CNN1, CNN2, CNN3, CNN4]

CNN4.summary()

# CNN Model Training

CNN_results = fit_models(network='CNN', models=CNN_models)

# Plot the CNN training results
plot_results(CNN_models, CNN_results)

# CNN-RNN Model

# Reset random seeds to reproduce results
reset_random_seeds()
# Type of neural network
CNN_RNN = Sequential(name='CNN_RNN')

# Convolutional neural network
CNN_RNN.add(Conv1D(filters=128, kernel_size=3, activation='relu',
                   input_shape=(lags, len(features))))
CNN_RNN.add(Dropout(rate=0.4, seed=seed))
CNN_RNN.add(MaxPooling1D(pool_size=2))

# Recurrent neural network
CNN_RNN.add(SimpleRNN(256, activation='relu',
                      return_sequences=True))
CNN_RNN.add(Dropout(rate=0.4, seed=seed))
CNN_RNN.add(SimpleRNN(256, activation='relu'))

CNN_RNN.add(Dropout(rate=0.4, seed=seed))
CNN_RNN.add(Dense(1, activation='sigmoid'))

CNN_RNN.compile(optimizer='Adagrad',
                loss='binary_crossentropy',
                metrics=['accuracy'])

CNN_RNN.summary()

# CNN-RNN Model Training

CNN_RNN_results = fit_models(network='CNN_RNN', models=[CNN_RNN])

CNN_RNN_results[['CNN_RNN_mean_val_loss',
                 'CNN_RNN_mean_val_acc',
                 'CNN_RNN_mean_loss',
                 'CNN_RNN_mean_acc']].plot(figsize=(8, 5),
                                           title='CNN_RNN fit results',
                                           style=['--', '--', '-', '-']);

# Evaluation of Different Models

# List of models with highest training accuracy
final_models = [DNN1, RNN2, CNN1, CNN_RNN]

for model in final_models:
    print(model.name + ' evaluation results...')

    if model.name == 'DNN1':
        # Use 2 dimensional input
        model.load_weights(str(model.name + '.hdf5'))
        print('Accuracy: '
              + str(round(model.evaluate(test[cols], test['d'])[1], 4)))
    else:
        # Use 3 dimensional input
        model.load_weights(str(model.name + '.hdf5'))
        print('Accuracy: '
              + str(round(model.evaluate(test_3d, test['d'])[1], 4)))

    print('=' * 56)

# Vectorized Backtest

# List of gross return column names
strategies = ['r']

# Iterate through each of the final models
for model in final_models:
    if model.name == 'DNN1':
        # Use 2 dimensional input
        test[str('p_' + model.name)] = model.predict_classes(test[cols])
    else:
        # Use 3 dimensional input
        test[str('p_' + model.name)] = model.predict_classes(test_3d)
    # Transform binary predictions to a long or short position
    test[str('p_' + model.name)] = np.where(test[str('p_'
                                                     + model.name)] > 0, 1, -1)
    # Calculate a strategy return (position * return)
    test[str('strategy_' + model.name)] = test[str('p_'
                                                   + model.name)] * test['r']
    # Append the gross strategy return column name
    strategies.append(str('strategy_' + model.name))

print(test[strategies].sum().apply(np.exp))

test[strategies].cumsum().apply(np.exp).plot(figsize=(16, 6),
                                             title='gross strategy returns');

print('Number of transactions for each strategy')
print('=' * 40)
# Iterate through final models
for model in final_models:
    # Print model name
    print(str('strategy_' + model.name + ':'))
    # Print number of position changes
    print(sum(test[str('p_' + model.name)].diff() != 0))

# Proportional transaction cost
ptc = 0.0025

# List of net return column names
strategies_tc = ['r']

for model in final_models:
    # Where there is a change in the position
    test[str('strategy_tc_' + model.name)] = np.where(test[str('p_' + model.name)].diff() != 0,  # Subtract ptc from gross return
                                       test[str('strategy_' + model.name)] - ptc,  # Otherwise, do nothing
                                       test[str('strategy_' + model.name)])

    # Append the net strategy return column name
    strategies_tc.append(str('strategy_tc_' + model.name))

# test[strategies_tc].sum().apply(np.exp)

test[strategies_tc].cumsum().apply(np.exp).plot(figsize=(16, 6), title='net strategy returns');

plt.show()

input("Input")
