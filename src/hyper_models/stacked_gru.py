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