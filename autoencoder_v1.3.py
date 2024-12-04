# -*- coding: utf-8 -*-
"""
Autoencoder test for the purposes of devving out GAN model in the next build
"""
# Author: James Brickner
# GitHub username: jamdatajam
# Date: 05/04/2024
# Description: Autoencoder test code for 4tend.com (ANN). Encodes then decodes 
# to discover hidden base truths in datasets. As such, the latent variables are
# the underlying patterns in the data that we can use to create reconstructions
# of synthetic data points until realistic or natural seeming patterns appear.
# Encoding is the compression of data into key patterns (lower dimensional space)
# decoding is restoring new data from the detected underlying patterns (restores
# the original dimensionality).


# include
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import losses #replaces 'objectives' import from old py
from keras.datasets import mnist
import numpy as np

# hyperparams #26
batch_size = 100 # samples per run
original_dim = 28*28 # dimensionality
latent_dim = 2 # number of neurons in the compression of the latent space
intermediate_dim = 256 # size of the latent dimension (essential features)
nb_epoch = 5 # number of times we run the training set through
epsilon_std = 1.0 # noise stdev

# helper funs
def sampling(args): # called from lambda to sample latent space and feed into decoder
    '''
    This function samples from the latent space and returns mean/var from the
    normal/gaussian distro
    '''
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# defining the encoder
x = Input(shape=(original_dim,), name="input") # input, sets input shape
h = Dense(intermediate_dim, activation='relu', name="encoding")(x) # intermediate layer, product with the input layer
z_mean = Dense(latent_dim, name="mean")(h) # gets the mean of the latent space
z_log_var = Dense(latent_dim, name="log-variance")(h) # defines the variance of the latent space
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var]) # =hyperparam for regularization/loss function
encoder = Model(x, [z_mean, z_log_var, z], name="encoder") # builds the model (keras)

# defining the decoder
input_decoder = Input(shape=(latent_dim,), name="decoder_input") # decoder input
decoder_h = Dense(intermediate_dim, activation='relu', # from latent to intermediate dim
                  name="decoder_h")(input_decoder)
x_decoded = Dense(original_dim, activation='sigmoid',
                  name="flat_decoded")(decoder_h) # creates mean of original dim
decoder = Model(input_decoder, x_decoded, name="decoder") # decoder as Keras

# combined encoder/decoder

# loss function

# train/test split

# fit

# additional testing