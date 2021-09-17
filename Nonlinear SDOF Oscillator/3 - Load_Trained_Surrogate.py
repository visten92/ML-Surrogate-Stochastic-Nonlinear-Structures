import tensorflow as tf
import numpy as np
import scipy.io

# Import dataset and the trained surrogate model (CAE & FFNN)
data = scipy.io.loadmat('test_solutions.mat')
test_parameters = scipy.io.loadmat('test_parameters.mat')
test_solutions = data['Displacements']
del data
alpha = test_parameters['alpha']
M = test_parameters['Mass']
test_parameters = np.concatenate([M, alpha], axis=1)
s_abs_max = np.load('u_max.npy')
test_solutions = np.expand_dims(test_solutions,axis = 2)
CAE_decoder = tf.keras.models.load_model('Decoder')
FFNN = tf.keras.models.load_model('FFNN')
print('....................................Models loaded....................................')

# Use the surrogate to predict
surrogate_results = s_abs_max * CAE_decoder(FFNN(test_parameters))

# Save the surrogate's predictions in a .mat file
scipy.io.savemat('results.mat',
                 {'results': surrogate_results.numpy()})

# Check surrogate's accuracy
N_test_samples = test_solutions.shape[0]
error_surrogate= 0
for sample in range(N_test_samples):
    error_surrogate = error_surrogate + (1 / N_test_samples) * np.linalg.norm(test_solutions[sample, :, 0:1] - surrogate_results[sample, :, 0:1], 2) / np.linalg.norm(test_solutions[sample, :, 0:1], 2)
print('error = ', error_surrogate*100, '%')

