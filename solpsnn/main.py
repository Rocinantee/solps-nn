import os
import sys

import numpy as np
from tensorflow import keras
from sklearn.preprocessing import QuantileTransformer

# The path to the package directory
package_dir = os.path.dirname(os.path.realpath(__file__))

species_dict = {'D0':0,'D1':1,'N0':2,'N1':3,'N2':4,'N3':5,'N4':6,'N5':7,'N6':8,'N7':9}

def input_pre_processing(X_in):
    """Change into units expected by the model"""
    X_out = np.copy(X_in)
    X_out[:,2] = X_in[:,2]/2.0
    X_out[:,3] = np.log10(X_in[:,3])
    X_out[:,4] = np.log10(X_in[:,4])
    X_out[:,5] = np.log10(X_in[:,5])
    return X_out
class QuantileTransformer1D:
    def __init__(self):
        self.qt = QuantileTransformer(n_quantiles=100, output_distribution="normal", copy=True, random_state=0)
    def fit_transform(self,y):
        return self.qt.fit_transform(y)
    def transform(self,y):
        return self.qt.transform(y)
    def inverse_transform(self,y):
        return self.qt.inverse_transform(y)
class QuantileTransformer2D:
    def __init__(self):
        self.qt = QuantileTransformer(n_quantiles=100, output_distribution="normal", copy=True, random_state=0)
    def fit_transform(self,y):
        # Change from 2D to 1D and fit_transform and back to 2D
        shape = y.shape
        y_1D = y.reshape((shape[0],shape[1]*shape[2]))
        y_normed_1D = self.qt.fit_transform(y_1D)
        return y_normed_1D.reshape(shape)
    def transform(self,y):
        # Change from 2D to 1D and transform and back to 2D
        shape = y.shape
        y_1D = y.reshape((shape[0],shape[1]*shape[2]))
        y_normed_1D = self.qt.transform(y_1D)
        return y_normed_1D.reshape(shape)
    def inverse_transform(self,y):
        # 2D to 1D, Inverse transform and change from 1D to 2D
        shape = y.shape
        y_1D = y.reshape((shape[0],shape[1]*shape[2]))
        y_normed_1D = self.qt.inverse_transform(y_1D)
        return y_normed_1D.reshape(shape)
class Model:
    def __init__(self,item='te',species=None,custom_model_folder=None):
        if custom_model_folder == None:
            data_folder = os.path.join(package_dir,'model_data')
        else:
            data_folder = custom_model_folder
        # Selecting model folder
        if item in ['te','ti','pwmxap','fnixap','psol']:
            model_folder = os.path.join(data_folder,item)
        elif item in ['na','ua']:
            if species == None:
                raise ValueError('Particle species needs to be specified')
            species_index = species_dict[species]
            model_folder = os.path.join(data_folder,'{}{}'.format(item,species_index))
        else:
            raise NotImplementedError('The requested model is not implemented.')
        # Check if model folder exists
        if not os.path.isdir(model_folder):
            error_msg = f'The requested model is not available. To download the model invoke solpsnn.download_model({item=},{species=})'
            raise FileNotFoundError(error_msg)
        # Loading input preprocessing
        if os.path.isfile(os.path.join(data_folder,'X_mean.npy')):
            self.X_mean = np.load(os.path.join(data_folder,'X_mean.npy'))
        else:
            error_msg = f'Input pre-processing is missing. Re-download the model using solpsnn.download_model({item=},{species=})'
            raise FileNotFoundError(error_msg)
        if os.path.isfile(os.path.join(data_folder,'X_std.npy')):
            self.X_std = np.load(os.path.join(data_folder,'X_std.npy'))
        else:
            error_msg = f'Input pre-processing is missing. Re-download the model using solpsnn.download_model({item=},{species=})'
            raise FileNotFoundError(error_msg)
        # Creating the output preprocessing
        if item in ['pwmxap','fnixap','psol']:
            self.qtND = QuantileTransformer1D()
        else:
            self.qtND = QuantileTransformer2D()
        self.qtND.qt.references_ = np.load(os.path.join(model_folder,'references.npy'))
        self.qtND.qt.quantiles_ = np.load(os.path.join(model_folder,'quantiles.npy'))
        self.qtND.qt.n_features_in_ = len(self.qtND.qt.quantiles_[0])
        self.qtND.qt.n_quantiles_ = len(self.qtND.qt.references_)
        # Loading the NN models
        self.models = []
        for split_count in range(1,6):
            self.models.append(keras.models.load_model(os.path.join(model_folder,'fold{}'.format(split_count))))
    def normalize(self,X_in):
        """Apply unit transformation and standardization to inputs"""
        X = input_pre_processing(X_in)
        return (X-self.X_mean)/self.X_std
    def transform(self,y):
        """Top level access to quantile transformer, for debugging purposes"""
        return self.qtND.transform(y)
    def inverse_transform(self,y):
        """Top level access to quantile transformer, for debugging purposes"""
        return self.qtND.inverse_transform(y)
    def predict_fold(self,X,fold,normalized=False):
        """Prediction of a single fold (1-5), for debugging purposes"""
        if fold not in range(1,len(self.models)+1):
            raise ValueError(f'fold must be between 1 and {len(self.models)}, got {fold}')
        X_normed = self.normalize(X)
        if normalized:
            return self.models[fold-1].predict(X_normed)
        else:
            return self.qtND.inverse_transform(self.models[fold-1].predict(X_normed))
    def predict(self,X):
        """Full model prediction"""
        X_normed = self.normalize(X)
        return np.mean([self.qtND.inverse_transform(self.models[i].predict(X_normed)) for i in range(0,5)],axis=0)