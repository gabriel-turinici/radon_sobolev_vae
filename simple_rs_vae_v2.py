# -*- coding: utf-8 -*-
"""
This code illustrates the Radon Sobolev Variational Auto-Encoder as described in arXiv:1911.13135
Some parts of the code build on the public implementation of the CWAE available at https://github.com/gmum/cwae

Reference this code as: Gabriel Turinici, Radon-Sobolev Variational AutoEncoder https://github.com/gabriel-turinici/radon_sobolev_vae 

(c) Gabriel Turinici 2021
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from os import path
#print(tf.__version__)

"""## Load the dataset
Can choose in the code below between 
- MNIST, Fashion-MNIST (they share the same architecture)
-  CIFAR10 (small adaptation required)
- any given 1D-data (need to use a csv file and indicate its location, see below the loader function)
"""

#dataset_name='MNIST'#can be "MNIST" or "FMNIST" or "CIFAR10"
dataset_name='CIFAR10'#can be "MNIST" or "FMNIST" or "CIFAR10"
dataset_name='MNIST'

#dataset_name='1D_object'
# as an example can use: The "PTB Diagnostic ECG Database" cf. https://www.kaggle.com/shayanfazeli/heartbeat for downsampled version and 
# https://www.physionet.org/content/ptbdb/1.0.0/ for initial version

def loader_1D():
  #implement your own 1D train and validation sets
  datafile='./sample_data/'+'ptbdb_normal.csv' 
  #load data from a local file: for this to work you need to put the file in the right place
  if path.exists(datafile):
    dataarray = np.array(pd.read_csv(datafile))
    print('loaded 1D data from file='+datafile)
  else:
    dataarray = np.random.rand(100,50)#just a random file to test
  #alternative loader_1D=tf.keras.datasets.mnist.load_data
  nb_validation_samples=int(dataarray.shape[0]/10)
  #keep 10% for validation
  return (dataarray[0:-nb_validation_samples,:],None),(dataarray[-nb_validation_samples:,:],None)


loaders={'1D_object':loader_1D,
          'MNIST':tf.keras.datasets.mnist.load_data,
         'FMNIST':tf.keras.datasets.fashion_mnist.load_data,
         'CIFAR10':tf.keras.datasets.cifar10.load_data}

loader=loaders[dataset_name]
(train_objects, _), (validation_objects, _) = loader()

object_shape=train_objects.shape[1:]
object_dims=len(object_shape)
print('object shape=',object_shape,' object dim=',object_dims)
shape_placeholder=[None]+list(object_shape)

#need to rescale if images
if (dataset_name=='MNIST') or (dataset_name=='FMNIST') or (dataset_name=='CIFAR10') :
  train_objects = train_objects.astype(np.float32) / 255.0
  validation_objects = validation_objects.astype(np.float32) / 255.0
else:#implement your own rescaling function for 1D objects if necessary
  None


print('type train_objects=',type(train_objects),' shape=',train_objects.shape)
print('type validation_objects=',type(validation_objects),' shape=',validation_objects.shape)

"""## AutoEncoder architecture and reconstruction error
Define the encoder $E_{\theta_e}$ and decoder $D_{\theta_d}$ and the reconstruction error
\begin{equation*}
Loss_{rec}(X;\theta_e,\theta_d)=\frac{1}{n}\sum_{i=1}^n 
\|x_i-D_{\theta_d} (E_{\theta_e}(x_i) ) \|^2.
\end{equation*}
"""

latent_sizes={'1D_object':8,'MNIST':8,'FMNIST':8,'CIFAR10':64}
z_dim = latent_sizes[dataset_name]

def encoder_map_mnist(x):
    hidden_layer_neurons_count = 200
    
    h = tf.keras.layers.Flatten()(x)

    h = tf.keras.layers.Dense(hidden_layer_neurons_count, activation='relu', name='Encoder_1')(h)
    h = tf.keras.layers.Dense(hidden_layer_neurons_count, activation='relu', name='Encoder_2')(h)
    h = tf.keras.layers.Dense(hidden_layer_neurons_count, activation='relu', name='Encoder_3')(h)

    return tf.keras.layers.Dense(z_dim, name='Encoder_output')(h)

def encoder_map_cifar10(x):
    h = tf.layers.conv2d(x, kernel_size=(2, 2), activation=tf.nn.relu, filters=3, name='Encoder_Conv_0')
    h = tf.layers.conv2d(h, kernel_size=(2, 2), strides=(2, 2), activation=tf.nn.relu, filters=32,
                          name='Encoder_Conv_1')
    h = tf.layers.conv2d(h, kernel_size=(2, 2), activation=tf.nn.relu, filters=32, name='Encoder_Conv_2')
    h = tf.layers.conv2d(h, kernel_size=(2, 2), activation=tf.nn.relu, filters=32, name='Encoder_Conv_3')
    h = tf.keras.layers.Flatten(name='Encoder_Flatten')()(h)
    h = tf.keras.layers.Dense(units=128, activation='relu', name='Encoder_FC')(h)
    return tf.keras.layers.Dense(z_dim, name='Encoder_output')(h)


encoders={'1D_object':encoder_map_mnist,'MNIST':encoder_map_mnist,'FMNIST':encoder_map_mnist,'CIFAR10':encoder_map_cifar10}
encoder_map=encoders[dataset_name]

def decoder_map_mnist(z):
    hidden_layer_neurons_count = 200

    h = z
    h = tf.keras.layers.Dense(hidden_layer_neurons_count, activation='relu', name='Decoder_1')(h)
    h = tf.keras.layers.Dense(hidden_layer_neurons_count, activation='relu', name='Decoder_2')(h)
    h = tf.keras.layers.Dense(hidden_layer_neurons_count, activation='relu', name='Decoder_3')(h)
    h = tf.keras.layers.Dense(units=np.product(np.array(object_shape)), activation='sigmoid', name='Output')(h)
    h = tf.reshape(h, [-1]+list(object_shape))
    return h


def decoder_map_1D(z):
    hidden_layer_neurons_count = 200

    h = z
    h = tf.keras.layers.Dense(hidden_layer_neurons_count, activation='relu', name='Decoder_1')(h)
    h = tf.keras.layers.Dense(hidden_layer_neurons_count, activation='relu', name='Decoder_2')(h)
    h = tf.keras.layers.Dense(hidden_layer_neurons_count, activation='relu', name='Decoder_3')(h)

    h = tf.keras.layers.Dense(units=np.product(np.array(object_shape)), activation='relu', name='Output')(h)
    h = tf.reshape(h, [-1]+list(object_shape))
    return h


def decoder_map_cifar10(z):
    h = tf.keras.layers.Dense(units=128, activation='relu', name='Decoder_FC_0')(z)
    h = tf.keras.layers.Dense(units=8192, activation='relu', name='Decoder_FC_1')()(h)
    h = tf.reshape(h, [-1, 16, 16, 32])
    h = tf.layers.conv2d_transpose(h, kernel_size=(2, 2), padding='same', activation=tf.nn.relu, filters=32,
                                    name='Decoder_Deconv_1')
    h = tf.layers.conv2d_transpose(h, kernel_size=(2, 2), padding='same', activation=tf.nn.relu, filters=32,
                                    name='Decoder_Deconv_2')
    h = tf.layers.conv2d_transpose(h, kernel_size=(3, 3), strides=(2, 2), filters=32, activation=tf.nn.relu,
                                    name='Decoder_Deconv_3')
    h = tf.layers.conv2d(h, kernel_size=(2, 2), filters=3, activation=tf.nn.sigmoid, name='Decoder_Conv_1')
    #h = tf.reshape(h, [-1, 32, 32,3])
    return h

decoders={'1D_object':decoder_map_1D,'MNIST':decoder_map_mnist,'FMNIST':decoder_map_mnist,'CIFAR10':decoder_map_cifar10}
decoder_map=decoders[dataset_name]


def euclidean_norm_squared(X, axis=None):
    return tf.reduce_sum(tf.square(X), axis=axis)

def squared_euclidean_norm_reconstruction_error(input, output):
    return euclidean_norm_squared(input - output, axis=1)

def mean_squared_euclidean_norm_reconstruction_error(x, y):
    return tf.reduce_mean(squared_euclidean_norm_reconstruction_error(tf.keras.layers.Flatten()(x), tf.keras.layers.Flatten()(y)))

reconstruction_error=mean_squared_euclidean_norm_reconstruction_error


tensor_input_x = tf.placeholder(shape=shape_placeholder, dtype=tf.float32)
tensor_z = encoder_map(tensor_input_x)
tensor_output_x = decoder_map(tensor_z)

"""##  Define distances
We define here some distances used to converge the latent space.
"""


def xs_distance(Z): #the Radon-Sobolev distance
    Nf = tf.stop_gradient(tf.cast(tf.shape(Z)[0], tf.float32))#batch size as float
    Df = tf.stop_gradient(tf.cast(tf.shape(Z)[1], tf.float32))#dimension as float

    f0 =tf.stop_gradient(
            (tf.sqrt(2.)-1.)*tf.exp(tf.math.lgamma(Df/2.0+0.5)
            -tf.math.lgamma(Df/2.0))
            )
    ddf0= tf.stop_gradient(
    tf.exp(tf.math.lgamma(.5+Df/2.)-tf.math.lgamma(1.+Df/2.))/tf.sqrt(2.))
    c0=f0 - 1./ddf0 #first constant in function
    c1=1/(ddf0**2) #second constant

    dist_real_Z = euclidean_norm_squared(Z, axis=1)
    distZZ = euclidean_norm_squared(tf.subtract(tf.expand_dims(Z, 0), tf.expand_dims(Z, 1)), axis=2)    

    #set diag = 1 because of the singularity of sqrt'(x) in zero. This is a numerical
    #issue due to low precision of the GPU computations; it is compensated
    #exactly in the return value
    distZZ = tf.matrix_set_diag(distZZ,tf.ones([ tf.stop_gradient(tf.shape(Z)[0]) ]),name=None) 
    #remark: this is similar to a pseudo-Huber function
    
    smalleps=1e-4
    return c0+tf.reduce_mean(tf.sqrt(dist_real_Z+c1)) \
      -0.5*tf.reduce_mean(tf.sqrt(distZZ+1e-6))+0.5/Nf+2.*tf.sqrt(smalleps) 
    #Note: 0.5/Nf has been added in order to compensate the "1" set on the diagonal
    #if necessary to apply the log in the cost, add a small constant for smoothness near zero

"""# Choice of the driver distance and of the reporting metrics.
"""


#this is the driver distance
main_distance=xs_distance
name_main_distance='Radon-Sobolev dist'

"""## Latent cost and overall cost function
Denote $X$ the current batch. The latent cost is 
$Loss_{lat}(X;\theta_e) = d_{XH}(E_{\theta_e}(X),N(0,I))^2$. 
The total cost function: 
$Loss_{total}(X;\theta_e,\theta_d)= Loss_{lat} (X;\theta_e)  
+  \lambda Loss_{rec} (X;\theta_e,\theta_d)$.
Change the code below 
- if set "lambda_factor=0" : no latent loss
- if set "rec_factor=0" : no reconstruction loss.
"""

tensor_rec_error = reconstruction_error(tensor_input_x, tensor_output_x)
tensor_main_distance = main_distance(tensor_z)
rec_factor=1.0
lambda_factor=100.0
tensor_cost_function = rec_factor*tensor_rec_error + lambda_factor*tensor_main_distance

"""# Optimizer"""

optimizer = tf.train.AdamOptimizer()
train_ops = optimizer.minimize(tensor_cost_function)

"""#Training"""

epochs_count = 1000
batch_size = 128

report_frequency = 5 #in epochs

main_distances = list()
reconstruction_errors = list()
epochs = list()
costs = list()


#print some outputs
def plot_results(iter_no):
    """#Reconstructed objects"""
    
    nb_reconstructions=64
    nb_reconstructions=32
    reconstruction_indexes = list(np.random.randint(0, len(train_objects), nb_reconstructions))
    
    reconstruction_objects = list()
    for i,train_index in enumerate(reconstruction_indexes):
      input_object = train_objects[train_index,...]
      one_latent, reconstructed_object = sess.run([tensor_z, tensor_output_x], feed_dict={tensor_input_x: [input_object]}) 
      reconstruction_objects.extend([input_object, reconstructed_object[0]])
    
    fig = plt.figure(figsize=(8, 8)) 
    plt.title('Reconstructed')
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    plt.axis('off')
    
    for i, img in enumerate(reconstruction_objects):
    #    sub = fig.add_subplot(8, 8*2, i + 1)
        sub = fig.add_subplot(8, 4*2, i + 1)
        sub.axis('off')
        if (object_dims==2):
          sub.imshow(img)
        elif (object_dims==1):
          sub.plot(img)
    plt.savefig(np.str(iter_no)+'_reconstructions.png')

    """### Sampled objects"""
    
    x = np.random.randn(64,z_dim) #plot a 8x8 matrix of sampled objects 
    decoded_objects = sess.run(tensor_output_x, feed_dict={tensor_z: x})
    fig = plt.figure(figsize=(8, 8)) 
    plt.title('Sampled')
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    plt.axis('off')
    for i in range(64):
        sub = fig.add_subplot(8, 8, i + 1,)
        sub.axis('off')
        if (object_dims==2):
          sub.imshow(decoded_objects[i])
        elif (object_dims==1):
          sub.plot(decoded_objects[i])
    plt.savefig(np.str(iter_no)+'_sampled.png')
    
    """## Interpolations"""
    
    interpolations_step_count = 16
    interpolations_count = 7
    
    interpolation_indexes = list(np.reshape(np.random.randint(0, len(train_objects), interpolations_count * 2), [-1, 2]))
    objects = list()
    
    for i, train_indexes in enumerate(interpolation_indexes):
        first_index = train_indexes[0]
        second_index = train_indexes[1]
    
        first_object = train_objects[first_index]
        second_object = train_objects[second_index]
    
        first_latent, first_object_decoded = sess.run([tensor_z, tensor_output_x], feed_dict={tensor_input_x: [first_object]})
        first_latent, first_object_decoded = first_latent[0], first_object_decoded[0]
        second_latent, second_latent_decoded = sess.run([tensor_z, tensor_output_x], feed_dict={tensor_input_x: [second_object]}) 
        second_latent, second_latent_decoded = second_latent[0], second_latent_decoded[0]
        objects.extend([first_object, first_object_decoded])
    
        latent_step = (second_latent - first_latent) / (interpolations_step_count + 1)
        for j in range(interpolations_step_count):
            next_latent = first_latent + (j + 1) * latent_step
            next_object = sess.run(tensor_output_x, feed_dict={tensor_z: [next_latent]})[0]
            objects.append(next_object)
    
        objects.extend([second_latent_decoded, second_object])
    
    #print(len(objects))
    
    fig = plt.figure(figsize=(10,3.5)) 
    plt.title('Interpolated')
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    plt.axis('off')
    for i, img in enumerate(objects):
        sub = fig.add_subplot(7, 20, i + 1)
        sub.axis('off')
#        sub.imshow(img)
        if (object_dims==2):
          sub.imshow(img)
        elif (object_dims==1):
          sub.plot(img)



    plt.pause(0.1)
    plt.savefig(np.str(iter_no)+'_interpolations.png')



sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for i in range(1, epochs_count+1):
    sys.stdout.flush()
    sys.stderr.flush()
    
    np.random.shuffle(train_objects)
    for b in np.array_split(train_objects, batch_size):
        sess.run(train_ops, feed_dict={tensor_input_x: b})
        
    if (i == epochs_count) or (i==1) or (i % report_frequency == 0):#report
        saver.save(sess, 'model_backup',global_step=i)    
        rec_error, val_main_distance, cost = sess.run(
             [tensor_rec_error, tensor_main_distance,tensor_cost_function],
              feed_dict={tensor_input_x: validation_objects})
        print('Epoch ', i, 'Rec error: ', rec_error, 
              name_main_distance, val_main_distance, 'Cost: ', cost)

        #log various costs
        main_distances.append(val_main_distance)
        reconstruction_errors.append(rec_error)
        costs.append(cost)
        epochs.append(i)
        
        #produce objects
        plot_results(i)
        
#final plots
plt.figure(3,figsize=(16,4))
plt.subplot(1,5,1)
plt.loglog(epochs,main_distances, label=name_main_distance)
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,5,4)
plt.loglog(epochs, reconstruction_errors, label='Reconstruction error')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,5,5)
plt.loglog(epochs, reconstruction_errors + np.log(main_distances), label='Validation cost')
plt.xlabel('Epoch')
plt.legend()

#plt.show()
plt.savefig('convergence.jpg')