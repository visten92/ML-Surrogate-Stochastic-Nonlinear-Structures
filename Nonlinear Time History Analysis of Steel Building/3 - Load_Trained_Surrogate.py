import tensorflow as tf
import numpy as np
import scipy.io
import time

# Import test dataset and the trained surrogate model (CAE & FFNN)
dof = 'Ux'
test_data = scipy.io.loadmat(dof + '.mat')
test_solutions = test_data[dof]
del test_data
split_ratio = 0.70
total_samples = test_solutions.shape[0]
training_samples = int(split_ratio * total_samples)
test_samples = total_samples - training_samples
test_solutions = test_solutions[training_samples:total_samples,:,:]
u_abs_max = np.load(dof + '_max.npy')
parameters = scipy.io.loadmat('parameters.mat')
alpha = parameters['alpha']
E = parameters['E']/2e+08 # Scale by dividing with the mean value
Fy = parameters['Fy']/2.75e+05 # Scale by dividing with the mean value
parameters = np.concatenate([E,alpha,Fy], axis=1)
test_parameters = parameters[training_samples:total_samples,:]

# Load trained models
CAE_decoder = tf.keras.models.load_model('Decoder_' + dof)
FFNN = tf.keras.models.load_model('FFNN_' + dof)
print('....................................models loaded....................................')

# Use the surrogate to predict
start_time = time.time()
surrogate_results = u_abs_max * CAE_decoder(FFNN(test_parameters))
training_time = time.time() - start_time
print('Prediction time: ',   training_time, ' sec')

# Save the surrogate's predictions in a .mat file
scipy.io.savemat(dof +'_test_sur.mat', {'results': surrogate_results.numpy()})

# Check surrogate's accuracy
error_surrogate = 0
for sample in range(test_samples):
    error_surrogate = error_surrogate + (1 / test_samples) * np.linalg.norm(test_solutions[sample, :, :] - surrogate_results[sample, :, :], 2) / np.linalg.norm(test_solutions[sample, :, :], 2)
print('error = ', error_surrogate * 100, '%')