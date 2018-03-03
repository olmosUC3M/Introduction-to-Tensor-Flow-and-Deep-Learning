# -*- coding: utf-8 -*-

## 2 fully connected layers at both encoding and decoding

import tensorflow as tf
import numpy as np


def encoder(X,shape,z_dim,num_samples):
    
    #Model inference 
    
    X = tf.reshape(X,[1,-1])
       
    hidden_dim = 50
    
    def_init = tf.random_normal_initializer(stddev=0.05)
    
    
    h = tf.layers.dense(inputs=X,units=hidden_dim,activation=tf.nn.tanh,kernel_initializer=def_init,name='layer_1_mean_enc',reuse=None)
    
    mean = tf.reshape(tf.layers.dense(inputs=h,units=z_dim,activation=None,kernel_initializer=def_init,name='layer_2_mean_enc',reuse=None),[z_dim,])
    
    h_logvar = tf.layers.dense(inputs=X,units=hidden_dim,activation=tf.nn.tanh,kernel_initializer=def_init,name='layer_1_var_enc',reuse=None)

    log_var = tf.reshape(tf.layers.dense(inputs=h_logvar,units=z_dim,activation=None,kernel_initializer=def_init,name='layer_2_var_enc',reuse=None),[z_dim,])


    # Sampling from q(z|x). Reparameterization trick. We sample from an independent Gaussian distribution
    
    eps = tf.random_normal((num_samples, z_dim), 0, 1, dtype=tf.float32)
      
    samples_z = mean + tf.multiply(tf.sqrt(tf.exp(log_var)), eps) 
    
    # KL distribution (we consider a N(0,I) prior)
    
    KL = -0.5*z_dim +0.5*tf.reduce_sum(tf.exp(log_var) +tf.square(mean) - log_var)
    
    
    return log_var, mean, samples_z, KL


def log_like(M,X,z,S,pos_s,A,var_s):
    
    #Evaluate log-likelihood 
    
    means = -1.0 * tf.multiply(A, tf.log(tf.reduce_sum(tf.square(pos_s - z),1)))
    
    loglik = -0.5 * tf.divide(tf.square(X-tf.reshape(means,[-1,1]))**2,var_s) 
    
    loglik += -0.5*np.log(2*np.pi)-0.5*np.log(var_s) #Normalizing constant (not important for inference)
    
    return tf.reduce_sum(loglik)



def decoder(loc_info,X,samples_z,num_samples_avg):
    
    # Compute average log_lik across z samples drawn from the encoder
    
    loglik = 0.0
    
    for i in range(num_samples_avg):
        
        loglik += log_like(loc_info['N'],X,samples_z[i,:],loc_info['S'],loc_info['pos_s'],loc_info['A'],loc_info['var_s'])
        
    loglik /=num_samples_avg
    
    return loglik






