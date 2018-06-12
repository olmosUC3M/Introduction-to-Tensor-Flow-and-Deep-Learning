# -*- coding: utf-8 -*-

## 2 fully connected layers at both encoding and decoding

import tensorflow as tf
import numpy as np


def encoder(X,z_dim,batch_size):
    
    #Model inference 
       
    hidden_dim = 200
    
    def_init = tf.random_normal_initializer(stddev=0.05)
    
    h = tf.layers.dense(inputs=X,units=hidden_dim,activation=tf.nn.tanh,kernel_initializer=def_init,name='layer_1_mean_enc',reuse=None)
    
    enc_mean = tf.layers.dense(inputs=h,units=z_dim,activation=tf.nn.tanh,kernel_initializer=def_init,name='layer_2_mean_enc',reuse=None)
    
    enc_mean *= 3
        
    h_logvar = tf.layers.dense(inputs=X,units=hidden_dim,activation=tf.nn.tanh,kernel_initializer=def_init,name='layer_1_var_enc',reuse=None)
    
    enc_logvar = tf.layers.dense(inputs=h_logvar,units=z_dim,activation=None,kernel_initializer=def_init,name='layer_2_var_enc',reuse=None)    


    # Reparameterization trick. We sample from a Enc_hdim-dimensional independent Gaussian distribution
    eps = tf.random_normal((batch_size, z_dim), 0, 1, dtype=tf.float32)
    z = enc_mean + tf.multiply(tf.sqrt(tf.exp(enc_logvar)), eps)
    
    # KL distribution (we consider a N(0,I) prior)
    
    KL = 0.5 * tf.reduce_sum(tf.exp(enc_logvar) + tf.square(enc_mean) - 1 - enc_logvar, 1)
    
    
    return enc_logvar, enc_mean, z, KL


def decoder(inputX,sample_z,sigma_reconstruction,output_dim):
    
    hidden_dim = 200
    
    def_init = tf.random_normal_initializer(stddev=0.05)
    
    h = tf.layers.dense(inputs=sample_z,units=hidden_dim,activation=tf.nn.tanh,kernel_initializer=def_init,name='layer_1_mean_dec',
                        reuse=None)
    
    dec_mean = tf.layers.dense(inputs=h,units=output_dim,activation=None,kernel_initializer=def_init,name='layer_2_mean_dec',reuse=None)
    
    loglik = -0.5 / sigma_reconstruction * tf.reduce_sum(tf.squared_difference(dec_mean, inputX), 1)
            

    return loglik,dec_mean



def new_samples(num_imgs,z_dim, output_dim):
    
    hidden_dim = 200
    
    # Sample from N(0,1)
    eps_sample = tf.random_normal((num_imgs, z_dim), 0, 1, dtype=tf.float32)
            
    h = tf.layers.dense(inputs=eps_sample,units=hidden_dim,activation=tf.nn.tanh,name= 'layer_1_mean_dec',reuse=True)
    
    samples = tf.layers.dense(inputs=h,units=output_dim,activation=None,name= 'layer_2_mean_dec',reuse=True)    
        
    return samples



