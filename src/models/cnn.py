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

  def tune(self):
    pass

  def plot(self):
    pass
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