import numpy as np
import h5py
from sklearn.utils import shuffle

def get_data(filepath, subject, num_validation=50, num_test=50, subtract_mean=True, subtract_axis=0, transpose=False):
    """
    Load data from all .mat files, combine them, eliminate EOG signals, shuffle and seperate
    training data, validation data and testing data. Also do mean subtraction on x.
    """
    
    x = []
    y = []

    # import all data in the desired subject .mat files
    file = filepath + '/A0' + str(subject) + 'T_slice.mat'
    A0XT = h5py.File(file, 'r')
    x.append(np.copy(A0XT['image']))
    y.append(np.copy(A0XT['type']))

    # reshape x in 3d data(N*25*1000) and y in 1d data(N)
    x = np.asarray(x)
    y = np.asarray(y, dtype=np.int32)
    y = y[0:9,-1]
    y = y[:,0:x.shape[1]:1] - 769
    x = np.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
    y = np.reshape(y, (y.shape[0]*y.shape[1]))
    
    # choose only to use the first 22 EEG channel
    x = x[:,0:-3,:]

    # check if there're NAN values
    np.isnan(np.sum(x))

    # remove those lines
    mask = ~np.any(np.isnan(x), axis=(1,2))
    x = x[mask]
    y = y[mask]
    np.isnan(np.sum(x))

    # now data loading is done, shuffle and seperate training, validation, test set
    N, C, H = x.shape
    num_training = N - num_test - num_validation
    x, y = shuffle(x, y, random_state=0)
    X_train = x[0:num_training,:,:]
    y_train = y[0:num_training]
    X_val = x[num_training:num_training+num_validation,:,:]
    y_val = y[num_training:num_training+num_validation]    
    X_test = x[num_training+num_validation:N,:,:]
    y_test = y[num_training+num_validation:N]
    
    # Transpose the second and third dimension
    if transpose:
        X_train = np.transpose(X_train, (0, 2, 1))
        X_val = np.transpose(X_val, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))

    # Normalize the data: subtract the mean image
    if subtract_mean:
      mean_image = np.mean(X_train, axis=subtract_axis)
      mean_image = np.expand_dims(mean_image, axis=subtract_axis)
      X_train -= mean_image
    
      mean_image = np.mean(X_val, axis=subtract_axis)
      mean_image = np.expand_dims(mean_image, axis=subtract_axis)    
      X_val -= mean_image
        
      mean_image = np.mean(X_test, axis=subtract_axis)
      mean_image = np.expand_dims(mean_image, axis=subtract_axis)        
      X_test -= mean_image

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }   
    
    
    
    
    

    