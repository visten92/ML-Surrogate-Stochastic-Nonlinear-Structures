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

# Split the data into training and test sets
split_ratio = 0.70 # train/test
total_samples = solutions.shape[0]
N_time_steps = solutions.shape[1]
total_dofs = solutions.shape[2]
training_samples = int(split_ratio * total_samples)
test_samples = total_samples - training_samples
test_solutions = solutions[training_samples:total_samples,:,:]
solutions = solutions[0:training_samples,:,:]

# Set hyperparameters
latent_space_dimension = 64
epochs = 1000
batch_size = 8
learning_rate = 1e-4

# Scale the data by dividing with max absolute value
u_min = np.abs(solutions.min())
u_max = np.abs(solutions.max())
if (u_min>u_max):
    u_max = u_min
np.save(dof + '_max.npy', u_max)
solutions = solutions / u_max
test_solutions = test_solutions / u_max

# Convolutional Autoencoder (CAE)
CAE_encoder = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(latent_space_dimension),
    tf.keras.layers.LeakyReLU()
])
CAE_decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape = latent_space_dimension),
    tf.keras.layers.Dense(1200),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Reshape(target_shape=(150, 8)),
    tf.keras.layers.Conv1DTranspose(filters=16, kernel_size=5, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=5, strides=2,  padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=5, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv1DTranspose(filters=total_dofs, kernel_size=5, strides=2,  padding='same')
])

CAE_input = tf.keras.Input(shape = (N_time_steps, total_dofs))
encoded_input = CAE_encoder(CAE_input)
decoded_output = CAE_decoder(encoded_input)
CAE = tf.keras.Model(CAE_input, decoded_output)
CAE.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = learning_rate), loss='mse')
CAE.summary()
start_time =  time.time()
CAE_history = CAE.fit(solutions, solutions, validation_data=(test_solutions, test_solutions), batch_size = batch_size, epochs = epochs, shuffle=True)
training_time = time.time() - start_time
print('Training time: ',   training_time, ' sec')
CAE_encoder.save('Encoder_' + dof)
CAE_decoder.save('Decoder_' + dof)

# CAE L2-norm error
test_results = u_max * CAE(test_solutions)
test_solutions = u_max * test_solutions
N_test_samples = test_solutions.shape[0]
error_autoencoder = 0
for sample in range(N_test_samples):
    error_autoencoder = error_autoencoder + (1 / N_test_samples) * np.linalg.norm(test_solutions[sample, :, :] - test_results[sample, :, :], 2) / np.linalg.norm(test_solutions[sample, :, :], 2)
print('L2 norm error - Autoencoder = ', error_autoencoder)

# Summarize history for loss
plt.plot(CAE_history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()