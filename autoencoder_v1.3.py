# -*- coding: utf-8 -*-
"""
Autoencoder test for the purposes of devving out GAN model in the next build
"""
# Author: James Brickner
# GitHub username: jamdatajam
# Date: 05/04/2024
# Description: Autoencoder test code for 4tend.com (ANN for VAE). Encodes then decodes 
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
from keras import metrics
from keras.datasets import mnist
import numpy as np
import tensorflow as tf

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
x_decoded_mean = tf.keras.backend.mean(x_decoded)


# combined encoder/decoder
encoded_data = encoder(x)[2] # encode the input
output_combined = decoder(encoded_data) # decodes the encoded data
variational_autoencoder = Model(x, output_combined) # models between the original data and the output
variational_autoencoder.summary() # give some results

# loss function as means to decrease error via gradient descent
def vae_loss(x: tf.Tensor, x_decoded_mean: tf.Tensor, z_log_var = z_log_var, z_mean = z_mean, original_dim = original_dim):
    # binary cross entropy + KL divergence to get overall loss
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
             
    # kullback-leiber divergence for the entropy btwn the distros
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    
    vae_loss = K.mean(xent_loss + kl_loss)
    return vae_loss # total loss fucntion

variational_autoencoder.compile(optimizer='rmsprop', loss=vae_loss) # compiles it
variational_autoencoder.summary()

# train/test split
# we linearize our matrices into a 1D array (classic)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# fit
variational_autoencoder.fit(x_train, x_train, 
        shuffle=True, # we make this un-ordered 
        #nb_epoch = nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        verbose=1)

# additional testing

