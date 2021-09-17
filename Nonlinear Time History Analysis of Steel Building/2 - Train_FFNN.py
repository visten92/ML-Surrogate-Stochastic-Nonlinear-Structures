import tensorflow as tf
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import time

# Load data
dof = 'Ux' # Type the desired degree of freedom (Ux, Uy, Uz, Rx, Ry, Rz)
data = scipy.io.loadmat(dof + '.mat')
solutions = data[dof]
del data
parameters = scipy.io.loadmat('parameters.mat')
alpha = parameters['alpha']
E = parameters['E']/2e+08 # Scale by dividing with the mean value
Fy = parameters['Fy']/2.75e+05 # Scale by dividing with the mean value
del parameters

# Split the data to training and test set
split_ratio = 0.70 # train/test
total_samples = solutions.shape[0]
N_time_steps = solutions.shape[1]
total_dofs = solutions.shape[2]
training_samples = int(split_ratio * total_samples)
test_samples = total_samples - training_samples
test_solutions = solutions[training_samples:total_samples,:,:]
solutions = solutions[0:training_samples,:,:]
parameters = np.concatenate([E,alpha,Fy], axis=1)
test_parameters = parameters[training_samples:total_samples,:]
parameters= parameters[0:training_samples,:]

# Set hyperparameters
latent_space_dimension = 64
epochs = 5000
batch_size = 16
learning_rate = 1e-4
hidden_size = 256

# Scale the data by dividing with max absolute value
u_abs_max = np.load(dof + '_max.npy')
solutions = solutions / u_abs_max
test_solutions = test_solutions / u_abs_max

# Load the trained Encoder
solutions_encoder = tf.keras.models.load_model('Encoder_' + dof)

# Feed Forward Neural Network (FFNN)
FFNN  = tf.keras.Sequential([
(tf.keras.Input(shape = 3)),
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
FFNN_output = solutions_encoder(solutions)
test_FFNN_output = solutions_encoder(test_solutions)
start_time = time.time()
FFNN_history = FFNN.fit(parameters, FFNN_output, validation_data=(test_parameters, test_FFNN_output), batch_size = batch_size, epochs = epochs, shuffle=True)
training_time = time.time() - start_time
print('Training time: ',   training_time, ' sec')
FFNN.save('FFNN_' + dof)

# FFNN L2-norm error
FFNN_results = FFNN(test_parameters)
encoded_results = FFNN(test_parameters)
encoded_solutions = solutions_encoder(test_solutions)
error_nn = 0
for sample in range(test_samples):
    error_nn =  error_nn + (1/test_samples) * (np.linalg.norm(encoded_solutions[sample,:] - encoded_results[sample,:],2 ) / np.linalg.norm(encoded_solutions[sample,:],2 ))
print('L2 norm error - FFNN = ', error_nn)

# Summarize history for loss
plt.plot(FFNN_history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()