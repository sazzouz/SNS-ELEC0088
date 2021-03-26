# Code exported from Google Colab document

### Setup ###

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler 
from pandas_profiling import ProfileReport
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
# Importing the necessary libraries to create/construct the neural network model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, GRU, RepeatVector, Conv1D, MaxPool1D, Flatten, TimeDistributed, Dropout, AvgPool1D
from tensorflow.keras import initializers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import he_uniform, Constant
import tensorflow as tf
import kerastuner as kt
import plotly.express as px

sns.set_style('dark')
sns.set_palette("flare")
sns_dir = '/content/drive/MyDrive/sns/'
florida_data_path = '/content/drive/MyDrive/sns/florida.csv'
florida21_data_path = '/content/drive/MyDrive/sns/florida21.csv'

### Data ###
florida21_df = pd.read_csv(
  florida21_data_path, 
  parse_dates=['date'], 
  index_col="date"
)

### Modelling ###

# Shared callbacks
callbacks = [
            #  EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='min'),
            #  ModelCheckpoint(simple_cnn_model_path, monitor='loss', save_best_only=True, mode='min', verbose=0),
             ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.001, verbose=1)
          ]

# Hyper Stacked LSTM
class HyperStackedLSTM(kt.HyperModel):
  def __init__(self, lstm_units=50, input_shape=None, features=None):
    self.lstm_units = lstm_units
    self.input_shape = input_shape
    self.features = features
    
# Dynamic optimizer retreival from hyper-parameter tuning
  def get_optimizer(self, hp, optimizer):
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    if optimizer == 'adam':
      return Adam(learning_rate=lr)
    elif optimizer == 'sgd':
      return SGD(learning_rate=lr)
    elif optimizer == 'rmsprop':
      return RMSprop(learning_rate=lr)

  def build(self, hp):
    ###### Setup hyperparamaters
    dropout = hp.Float('dropout', 0.1, 0.6)
    bias_constant = hp.Float('bias', 0.01, 0.03)
    optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])

    ###### Construct model
    # Initially, the network model is defined 
    model = Sequential()
    # Add layers
    model.add(LSTM(units=self.lstm_units, input_shape=self.input_shape, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=int(self.lstm_units*0.5), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(features, activation='sigmoid', kernel_initializer=initializers.he_uniform(seed=0), bias_initializer=initializers.Constant(bias_constant)))
    # Compile model
    model.compile(
      optimizer=self.get_optimizer(hp, optimizer),
      loss='mse', 
      )
    return model


# Set params
max_epochs = 10
executions_per_trial = 3
hyperband_iterations = 2

hyper_simple_lstm = HyperSimpleLSTM(input_shape=(length, features), features=features)
hyper_simple_lstm_tuner = kt.Hyperband(
    hyper_simple_lstm,
    objective='loss',
    max_epochs=max_epochs,
    executions_per_trial=executions_per_trial,
    hyperband_iterations=hyperband_iterations,
    directory=sns_dir + 'tuning/',
    project_name='hyper_simple_lstm_tuning',
    seed=66,
    overwrite=True
)
hyper_simple_lstm_search_space_summary = hyper_simple_lstm_tuner.search_space_summary()

### Conduct tuning

# Set params
search_epochs = 3

hyper_simple_lstm_tuner.search(
    time_series_generator,
    epochs=search_epochs,
    verbose=1
)
hyper_simple_lstm_results_summary = hyper_simple_lstm_tuner.results_summary()

# Extract the best model following tuning
hyper_simple_lstm_best_hps = hyper_simple_lstm_tuner.get_best_hyperparameters()[0]
hyper_simple_lstm_best_hps_config = hyper_simple_lstm_best_hps.get_config()
hyper_simple_lstm_best_model = hyper_simple_lstm_tuner.hypermodel.build(hyper_simple_lstm_best_hps)
hyper_simple_lstm_best_model_config = hyper_simple_lstm_best_model.get_config()
hyper_simple_lstm_best_model_summary = hyper_simple_lstm_best_model.summary()

# Plot the best model
plot_model(hyper_simple_lstm_best_model, show_shapes=True, show_layer_names=True, to_file='simple_lstm_model.png')

# Save the best model to disk
hyper_simple_lstm_best_model.save(simple_lstm_model_path, overwrite=True)

# Reload the best model
simple_lstm_model = load_model(simple_lstm_model_path)

# Train the model
simple_lstm_history = simple_lstm_model.fit(time_series_generator, epochs=100, callbacks=callbacks)

# Training loss plot
simple_lstm_loss = pd.DataFrame()
simple_lstm_loss['loss'] = simple_lstm_history.history['loss']
min_simple_lstm_loss = min(simple_lstm_loss['loss'])
simple_lstm_loss.plot()
plt.title('Stacked LSTM Training Loss', fontweight='bold')
plt.xlabel('Epochs', fontweight='bold')
plt.ylabel("Loss", fontweight='bold');

# Determining all predicted values so as to compare them with the actual test values 
n_features = scaled_train_data.shape[1]
test_outputs = []
batch = scaled_train_data[-length:].reshape((1, length, n_features))

for i in range(len(test_data)):
    test_out = simple_lstm_model.predict(batch)[0]
    test_outputs.append(test_out) 
    batch = np.append(batch[:,1:,:],[[test_out]],axis=1)
    
# Applying the inverse_transform function to the test_outputs to get their original values
true_outputs = scaler.inverse_transform(test_outputs)

# Converting the true_outputs from np.ndarray to pandas dataframe
true_outputs = pd.DataFrame(data=true_outputs,columns=test_data.columns,index=test_data.index)

fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(12,8))

pre_florida21_df['positiveIncrease'][:-test_index].plot(linestyle='--',marker='.',ax=axes[0,0],label='Train')
test_data['positiveIncrease'].plot(linestyle='--',marker='.',ax=axes[0,0],label='Targets', color='grey')
true_outputs['positiveIncrease'].plot(linestyle='--',marker='.',ax=axes[0,0],label='Predictions', color='c')
axes[0,0].set_title ('Positive Increase',fontweight='bold',fontsize=12)
axes[0,0].legend()

pre_florida21_df['deathIncrease'][:-test_index].plot(linestyle='--',marker='.',ax=axes[0,1],label='Train')
test_data['deathIncrease'].plot(linestyle='--',marker='.',ax=axes[0,1],label='Targets', color='grey')
true_outputs['deathIncrease'].plot(linestyle='--',marker='.',ax=axes[0,1],label='Predictions', color='c')
axes[0,1].set_title ('Death Increase',fontweight='bold',fontsize=12)
axes[0,1].legend()

pre_florida21_df['negativeIncrease'][:-test_index].plot(linestyle='--',marker='.',ax=axes[1,0],label='Train')
test_data['negativeIncrease'].plot(linestyle='--',marker='.',ax=axes[1,0],label='Targets', color='grey')
true_outputs['negativeIncrease'].plot(linestyle='--',marker='.',ax=axes[1,0],label='Predictions', color='c')
axes[1,0].set_title ('Negative Increase',fontweight='bold',fontsize=12)
axes[1,0].legend()

pre_florida21_df['hospitalizedIncrease'][:-test_index].plot(linestyle='--',marker='.',ax=axes[1,1],label='Train')
test_data['hospitalizedIncrease'].plot(linestyle='--',marker='.',ax=axes[1,1],label='Targets', color='grey')
true_outputs['hospitalizedIncrease'].plot(linestyle='--',marker='.',ax=axes[1,1],label='Predictions', color='c')
axes[1,1].set_title ('Hospitalized Increase',fontweight='bold',fontsize=12)
axes[1,1].legend()

# Hyper Stacked GRU
class HyperStackedGRU(kt.HyperModel):
  def __init__(self, gru_units=50, input_shape=None, features=None):
    self.gru_units = gru_units
    self.input_shape = input_shape
    self.features = features

# Dynamic optimizer retreival from hyper-parameter tuning
  def get_optimizer(self, hp, optimizer):
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    if optimizer == 'adam':
      return Adam(learning_rate=lr)
    elif optimizer == 'sgd':
      return SGD(learning_rate=lr)
    elif optimizer == 'rmsprop':
      return RMSprop(learning_rate=lr)

  def build(self, hp):
    ###### Setup hyperparamaters
    dropout = hp.Float('dropout', 0.1, 0.6)
    bias_constant = hp.Float('bias', 0.01, 0.03)
    optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])

    ###### Construct model
    # Initially, the network model is defined 
    model = Sequential()
    # Add layers
    model.add(GRU(units=self.gru_units, input_shape=self.input_shape, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(units=int(self.gru_units*0.5), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(self.features, activation='sigmoid', kernel_initializer=initializer, bias_initializer=Constant(bias_constant)))
    # Compile model
    model.compile(
      optimizer=self.get_optimizer(hp, optimizer),
      loss='mse', 
      )
    return model

# Search space inspection

# Set params
max_epochs = 10
executions_per_trial = 3
hyperband_iterations = 1

hyper_simple_gru = HyperSimpleGRU(input_shape=(length, features), features=features)
hyper_simple_gru_tuner = kt.Hyperband(
    hyper_simple_gru,
    objective='loss',
    max_epochs=max_epochs,
    executions_per_trial=executions_per_trial,
    hyperband_iterations=hyperband_iterations,
    directory=sns_dir + 'tuning/',
    project_name='hyper_simple_gru_tuning',
    seed=66,
    overwrite=True
)
hyper_simple_gru_search_space_summary = hyper_simple_gru_tuner.search_space_summary()

# Conduct tuning

# Set params
search_epochs = 3

hyper_simple_gru_tuner.search(
    time_series_generator,
    epochs=search_epochs,
    verbose=1
)
hyper_simple_gru_results_summary = hyper_simple_gru_tuner.results_summary()

# Extract the best model following tuning
hyper_simple_gru_best_hps = hyper_simple_gru_tuner.get_best_hyperparameters()[0]
hyper_simple_gru_best_hps_config = hyper_simple_gru_best_hps.get_config()
hyper_simple_gru_best_model = hyper_simple_gru_tuner.hypermodel.build(hyper_simple_gru_best_hps)
hyper_simple_gru_best_model_config = hyper_simple_gru_best_model.get_config()
hyper_simple_gru_best_model_summary = hyper_simple_gru_best_model.summary()

# Plot the best model
plot_model(hyper_simple_gru_best_model, show_shapes=True, show_layer_names=True, to_file='simple_gru_model.png')

# Save the best model to disk
hyper_simple_gru_best_model.save(simple_gru_model_path, overwrite=True)

# Reload the best model
simple_gru_model = load_model(simple_gru_model_path)

# Fit the model
simple_gru_history = simple_gru_model.fit(time_series_generator, epochs=100, callbacks=callbacks)

# Training loss plot
simple_gru_loss = pd.DataFrame()
simple_gru_loss['loss'] = simple_gru_history.history['loss']
min_simple_gru_loss = min(simple_gru_loss['loss'])
simple_gru_loss.plot()
plt.title('Stacked GRU Training Loss', fontweight='bold')
plt.xlabel('Epochs', fontweight='bold')
plt.ylabel("Loss", fontweight='bold');

# Determining all predicted values so as to compare them with the actual test values 
n_features = scaled_train_data.shape[1]
test_outputs = []
batch = scaled_train_data[-length:].reshape((1, length, n_features))

for i in range(len(test_data)):
    test_out = simple_gru_model.predict(batch)[0]
    test_outputs.append(test_out) 
    batch = np.append(batch[:,1:,:],[[test_out]],axis=1)
    
    # Applying the inverse_transform function to the test_outputs to get their original values
true_outputs = scaler.inverse_transform(test_outputs)

# Converting the true_outputs from np.ndarray to pandas dataframe
true_outputs = pd.DataFrame(data=true_outputs,columns=test_data.columns,index=test_data.index)

fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(12,8))

pre_florida21_df['positiveIncrease'][:-test_index].plot(linestyle='--',marker='.',ax=axes[0,0],label='Train')
test_data['positiveIncrease'].plot(linestyle='--',marker='.',ax=axes[0,0],label='Targets', color='grey')
true_outputs['positiveIncrease'].plot(linestyle='--',marker='.',ax=axes[0,0],label='Predictions', color='c')
axes[0,0].set_title ('Positive Increase',fontweight='bold',fontsize=12)
axes[0,0].legend()

pre_florida21_df['deathIncrease'][:-test_index].plot(linestyle='--',marker='.',ax=axes[0,1],label='Train')
test_data['deathIncrease'].plot(linestyle='--',marker='.',ax=axes[0,1],label='Targets', color='grey')
true_outputs['deathIncrease'].plot(linestyle='--',marker='.',ax=axes[0,1],label='Predictions', color='c')
axes[0,1].set_title ('Death Increase',fontweight='bold',fontsize=12)
axes[0,1].legend()

pre_florida21_df['negativeIncrease'][:-test_index].plot(linestyle='--',marker='.',ax=axes[1,0],label='Train')
test_data['negativeIncrease'].plot(linestyle='--',marker='.',ax=axes[1,0],label='Targets', color='grey')
true_outputs['negativeIncrease'].plot(linestyle='--',marker='.',ax=axes[1,0],label='Predictions', color='c')
axes[1,0].set_title ('Negative Increase',fontweight='bold',fontsize=12)
axes[1,0].legend()

pre_florida21_df['hospitalizedIncrease'][:-test_index].plot(linestyle='--',marker='.',ax=axes[1,1],label='Train')
test_data['hospitalizedIncrease'].plot(linestyle='--',marker='.',ax=axes[1,1],label='Targets', color='grey')
true_outputs['hospitalizedIncrease'].plot(linestyle='--',marker='.',ax=axes[1,1],label='Predictions', color='c')
axes[1,1].set_title ('Hospitalized Increase',fontweight='bold',fontsize=12)
axes[1,1].legend()

# Hyper Stacked CNN
class HyperStackedCNN(kt.HyperModel):
  def __init__(self, filters=16, kernel_size=3, input_shape=None, features=None):
    self.filters = filters
    self.kernel_size = kernel_size
    self.input_shape = input_shape
    self.features = features

# Dynamic optimizer retreival from hyper-parameter tuning
  def get_optimizer(self, hp, optimizer):
    lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    if optimizer == 'adam':
      return Adam(learning_rate=lr)
    elif optimizer == 'sgd':
      return SGD(learning_rate=lr)
    elif optimizer == 'rmsprop':
      return RMSprop(learning_rate=lr)

  def build(self, hp):
    ###### Setup hyperparamaters
    pooling_1 = hp.Choice('pooling_1', ['avg', 'max'])
    pooling_2 = hp.Choice('pooling_2', ['avg', 'max'])
    pooling_3 = hp.Choice('pooling_3', ['avg', 'max'])
    dense_units = hp.Int('dense_units', min_value=16, max_value=128, step=16)
    dropout = hp.Float('dropout', 0.1, 0.6)
    bias_constant = hp.Float('bias', 0.01, 0.03)
    optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])

    ###### Construct model
    # Instantiate sequential model.
    model=Sequential()
    model.add(Conv1D(self.filters, self.kernel_size, padding='same', activation='relu', input_shape=self.input_shape))
    if pooling_1 == 'max':
      model.add(MaxPool1D())
    else:
      model.add(AvgPool1D())
    model.add(Conv1D(self.filters*2, self.kernel_size, padding='same', activation='relu'))
    if pooling_2 == 'max':
      model.add(MaxPool1D())
    else:
      model.add(AvgPool1D())
    model.add(Conv1D(self.filters*4, self.kernel_size, padding='same', activation='relu'))
    if pooling_3 == 'max':
      model.add(MaxPool1D())
    else:
      model.add(AvgPool1D())
    # Add dropout layer before flattening
    model.add(Dropout(dropout))
    # Flatten for use with fully-connected layers
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    # Regularization layer using dropout
    model.add(Dropout(dropout))
    # Output layer with Softmax activation due to multivariate value
    model.add(Dense(
        self.features,
        kernel_initializer=he_uniform(seed=0),
        bias_initializer=Constant(bias_constant),
        activation='sigmoid'
        )
    )
    # Compile model
    model.compile(
      optimizer=self.get_optimizer(hp, optimizer),
      loss='mse', 
      )
    return model

# Search space inspection

# Set params
max_epochs = 10
executions_per_trial = 3
hyperband_iterations = 1

hyper_simple_cnn = HyperSimpleCNN(input_shape=(length, features), features=features)
hyper_simple_cnn_tuner = kt.Hyperband(
    hyper_simple_cnn,
    objective='loss',
    max_epochs=max_epochs,
    executions_per_trial=executions_per_trial,
    hyperband_iterations=hyperband_iterations,
    directory=sns_dir + 'tuning/',
    project_name='hyper_simple_cnn_tuning',
    seed=66,
    overwrite=True
)
hyper_simple_cnn_search_space_summary = hyper_simple_cnn_tuner.search_space_summary()

# Conduct tuning

# Set params
search_epochs = 3

hyper_simple_cnn_tuner.search(
    time_series_generator,
    epochs=search_epochs,
    verbose=1
)
hyper_simple_cnn_results_summary = hyper_simple_cnn_tuner.results_summary()

# Extract the best model following tuning
hyper_simple_cnn_best_hps = hyper_simple_cnn_tuner.get_best_hyperparameters()[0]
hyper_simple_cnn_best_hps_config = hyper_simple_cnn_best_hps.get_config()
hyper_simple_cnn_best_model = hyper_simple_cnn_tuner.hypermodel.build(hyper_simple_cnn_best_hps)
hyper_simple_cnn_best_model_config = hyper_simple_cnn_best_model.get_config()
hyper_simple_cnn_best_model_summary = hyper_simple_cnn_best_model.summary()

# Plot the best model
plot_model(hyper_simple_cnn_best_model, show_shapes=True, show_layer_names=True, to_file='simple_cnn_model.png')

# Save the best model to disk
hyper_simple_cnn_best_model.save(simple_cnn_model_path, overwrite=True)

# Reload the best model
simple_cnn_model = load_model(simple_cnn_model_path)

# Train the model
simple_cnn_history = simple_cnn_model.fit(time_series_generator, epochs=100, callbacks=callbacks)

# Training loss plot
simple_cnn_loss = pd.DataFrame()
simple_cnn_loss['loss'] = simple_cnn_history.history['loss']
min_simple_cnn_loss = min(simple_cnn_loss['loss'])
simple_cnn_loss.plot()
plt.title('CNN Training Loss', fontweight='bold')
plt.xlabel('Epochs', fontweight='bold')
plt.ylabel("Loss", fontweight='bold');

# Determining all predicted values so as to compare them with the actual test values 
n_features = scaled_train_data.shape[1]
test_outputs = []
batch = scaled_train_data[-length:].reshape((1, length, n_features))

for i in range(len(test_data)):
    test_out = simple_cnn_model.predict(batch)[0]
    test_outputs.append(test_out) 
    batch = np.append(batch[:,1:,:],[[test_out]],axis=1)
    
# Applying the inverse_transform function to the test_outputs to get their original values
true_outputs = scaler.inverse_transform(test_outputs)

# Converting the true_outputs from np.ndarray to pandas dataframe
true_outputs = pd.DataFrame(data=true_outputs,columns=test_data.columns,index=test_data.index)

acc_df = pd.DataFrame(columns=list(test_data.columns))
acc_df['date'] = list(test_data.index)
acc_df = acc_df.set_index('date')
for col in list(acc_df.columns):
  diff_list = []
  for i in range(len(test_data)):
    diff = test_data[str(col)].iloc[i] - true_outputs[str(col)].iloc[i]
    if not diff > 1:
      diff = diff*-1
    diff_list.append(diff)
  acc_df[str(col)] = diff_list
acc_df.head()

fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(12,8))

pre_florida21_df['positiveIncrease'][:-test_index].plot(linestyle='--',marker='.',ax=axes[0,0],label='Train')
test_data['positiveIncrease'].plot(linestyle='--',marker='.',ax=axes[0,0],label='Targets', color='grey')
true_outputs['positiveIncrease'].plot(linestyle='--',marker='.',ax=axes[0,0],label='Predictions', color='c')
axes[0,0].set_title ('Positive Increase',fontweight='bold',fontsize=12)
axes[0,0].legend()

pre_florida21_df['deathIncrease'][:-test_index].plot(linestyle='--',marker='.',ax=axes[0,1],label='Train')
test_data['deathIncrease'].plot(linestyle='--',marker='.',ax=axes[0,1],label='Targets', color='grey')
true_outputs['deathIncrease'].plot(linestyle='--',marker='.',ax=axes[0,1],label='Predictions', color='c')
axes[0,1].set_title ('Death Increase',fontweight='bold',fontsize=12)
axes[0,1].legend()

pre_florida21_df['negativeIncrease'][:-test_index].plot(linestyle='--',marker='.',ax=axes[1,0],label='Train')
test_data['negativeIncrease'].plot(linestyle='--',marker='.',ax=axes[1,0],label='Targets', color='grey')
true_outputs['negativeIncrease'].plot(linestyle='--',marker='.',ax=axes[1,0],label='Predictions', color='c')
axes[1,0].set_title ('Negative Increase',fontweight='bold',fontsize=12)
axes[1,0].legend()

pre_florida21_df['hospitalizedIncrease'][:-test_index].plot(linestyle='--',marker='.',ax=axes[1,1],label='Train')
test_data['hospitalizedIncrease'].plot(linestyle='--',marker='.',ax=axes[1,1],label='Targets', color='grey')
true_outputs['hospitalizedIncrease'].plot(linestyle='--',marker='.',ax=axes[1,1],label='Predictions', color='c')
axes[1,1].set_title ('Hospitalized Increase',fontweight='bold',fontsize=12)
axes[1,1].legend()

# Init full results object
results = {}
# LSTM metrics
simple_lstm = {}
simple_lstm['min_loss'] = min_simple_lstm_loss
simple_lstm['model_path'] = simple_lstm_model_path
simple_lstm['history'] = simple_lstm_history.history['loss']
results['simple_lstm'] = simple_lstm
# GRU
simple_gru = {}
simple_gru['min_loss'] = min_simple_gru_loss
simple_gru['model_path'] = simple_gru_model_path
simple_gru['history'] = simple_gru_history.history['loss']
results['simple_gru'] = simple_gru
# CNN
simple_cnn = {}
simple_cnn['min_loss'] = min_simple_cnn_loss
simple_cnn['model_path'] = simple_cnn_model_path
simple_cnn['history'] = simple_cnn_history.history['loss']
results['simple_cnn'] = simple_cnn
# Print to show details
results

# Transform info DataFrame for exporting
results_df = pd.DataFrame()
results_df['simple_lstm'] = results['simple_lstm']['history']
# results_df['bi_lstm'] = results['bi_lstm']['history']
results_df['simple_gru'] = results['simple_gru']['history']
# results_df['enc_dec'] = results['enc_dec']['history']
results_df['simple_cnn'] = results['simple_cnn']['history']
results_df

# Visualise trainning performance of each respective model
def plot_all_loss(results, task_name):
    # Plot accuracies
    fig, ax = plt.subplots()
    # summarize history for accuracy
    ax.plot(results['simple_lstm']['history'], 'b--', label="loss")
    # ax.plot(results['bi_lstm']['history'], 'm--', label="loss")
    ax.plot(results['simple_gru']['history'], 'y--', label="loss" )
    # ax.plot(results['enc_dec']['history'], 'c--', label="loss" )
    ax.plot(results['simple_cnn']['history'], 'g--', label="loss")
    ax.set_title('Model Training Losses', fontweight='bold', fontsize=12)
    ax.set_ylabel('Loss', fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch', fontweight='bold',fontsize=12)
    #ax.legend(['Simple LSTM', 'Bidirectional LSTM', 'Simple GRU', 'Encoder Decoder', 'CNN'], loc='upper right')
    ax.legend(['Stacked LSTM', 'Stacked GRU', 'CNN'], loc='upper right')
    fig.savefig('{0}_all_loss'.format(task_name), bbox_inches='tight')

plot_all_loss(results, 'SNS Full Results')