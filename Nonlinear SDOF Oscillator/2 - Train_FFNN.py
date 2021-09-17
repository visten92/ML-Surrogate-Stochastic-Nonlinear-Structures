import tensorflow as tf
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import time

# Load data and set hyperparameters
data = scipy.io.loadmat('solutions.mat')
solutions = data['Displacements']
del data
parameters = scipy.io.loadmat('parameters.mat')
alpha = parameters['alpha']
M = parameters['Mass']
del parameters
training_samples = 250
test_solutions = solutions[training_samples:600,:]
test_solutions = np.expand_dims(test_solutions,axis = 2)
N_time_steps = 400
total_dofs = 1
solutions = solutions[0:training_samples,:]
solutions = np.expand_dims(solutions,axis = 2)
parameters = np.concatenate([M, alpha], axis=1)
test_parameters = parameters[training_samples:600,:]
parameters= parameters[0:training_samples,:]
latent_space_dimension = 8
epochs = 20000
batch_size = 16
learning_rate = 1e-4
hidden_size = 64
u_abs_max = np.load('u_max.npy')
solutions = solutions / u_abs_max
test_solutions = test_solutions / u_abs_max

#Load Encoder
CAE_encoder = tf.keras.models.load_model('Encoder')

# Feed Forward Neural Network (FFNN)
FFNN  = tf.keras.Sequential([
(tf.keras.Input(shape = 2)),
tf.keras.layers.Dense(hidden_size),
tf.keras.layers.LeakyReLU(),
tf.keras.layers.Dense(hidden_size),
tf.keras.layers.LeakyReLU(),
tf.keras.layers.Dense(hidden_size),
tf.keras.layers.LeakyReLU(),
tf.keras.layers.Dense(hidden_size),
tf.keras.layers.LeakyReLU(),
tf.keras.layers.Dense(hidden_size),
tf.keras.layers.LeakyReLU(),
tf.keras.layers.Dense(hidden_size),
tf.keras.layers.LeakyReLU(),
tf.keras.layers.Dense(latent_space_dimension)
])

FFNN.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = learning_rate), loss='mse')
FFNN_output = CAE_encoder(solutions)
test_FFNN_output = CAE_encoder(test_solutions)
start_time = time.time()
FFNN_history = FFNN.fit(parameters, FFNN_output, validation_data=(test_parameters, test_FFNN_output), batch_size = batch_size, epochs = epochs, shuffle=True)
training_time = time.time() - start_time
print('Training time: ',   training_time, ' sec')
FFNN.save('FFNN')

# FFNN L2-norm error
FFNN_results = FFNN(test_parameters)
encoded_results = FFNN(test_parameters)
encoded_solutions = CAE_encoder(test_solutions)
N_test_samples = test_parameters.shape[0]
error_nn = 0
for sample in range(N_test_samples):
    error_nn =  error_nn + (1/N_test_samples) * (np.linalg.norm(encoded_solutions[sample,:] - encoded_results[sample,:],2 ) / np.linalg.norm(encoded_solutions[sample,:],2 ))
print('L2 norm error - FFNN = ', error_nn)

# Summarize history for loss
plt.plot(FFNN_history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()