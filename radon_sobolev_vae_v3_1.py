# -*- coding: utf-8 -*-
"""
This code illustrates the Radon Sobolev Variational Auto-Encoder as described in arXiv:1911.13135

Some parts of the code build on the public implementation of the CWAE available at https://github.com/gmum/cwae
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
import numpy as np
import sys
#print(tf.__version__)

"""## Load the dataset
Can choose in the code below between MNIST and Fashion-MNIST (they share the same architecture).
"""

#dataset_name='MNIST'#can be "MNIST" or "FMNIST" or "CIFAR10"
dataset_name='CIFAR10'#can be "MNIST" or "FMNIST" or "CIFAR10"
dataset_name='MNIST'

loaders={'MNIST':tf.keras.datasets.mnist.load_data,
         'FMNIST':tf.keras.datasets.fashion_mnist.load_data,
         'CIFAR10':tf.keras.datasets.cifar10.load_data}

loader=loaders[dataset_name]
(tr_images, _), (validation_images, _) = loader()
tr_images = tr_images.astype(np.float32) / 255.0
validation_images = validation_images.astype(np.float32) / 255.0

nr_test_img=validation_images.shape[0]

print('type tr_images=',type(tr_images),' shape=',tr_images.shape)
print('type validation_images=',type(validation_images),' shape=',validation_images.shape)

"""## AutoEncoder architecture and reconstruction error
Define the encoder $E_{\theta_e}$ and decoder $D_{\theta_d}$ and the reconstruction error
\begin{equation*}
Loss_{rec}(X;\theta_e,\theta_d)=\frac{1}{n}\sum_{i=1}^n 
\|x_i-D_{\theta_d} (E_{\theta_e}(x_i) ) \|^2.
\end{equation*}
"""

latent_sizes={'MNIST':8,'FMNIST':8,'CIFAR10':64}
z_dim = latent_sizes[dataset_name]

def encoder_map_mnist(x):
    hidden_layer_neurons_count = 200
    
    h = tf.layers.flatten(x)
    h = tf.layers.dense(h, hidden_layer_neurons_count, activation=tf.nn.relu, name='Encoder_1')
    h = tf.layers.dense(h, hidden_layer_neurons_count, activation=tf.nn.relu, name='Encoder_2')
    h = tf.layers.dense(h, hidden_layer_neurons_count, activation=tf.nn.relu, name='Encoder_3')

    return tf.layers.dense(h, z_dim, name='Encoder_output')

def encoder_map_cifar10(x):
    h = tf.layers.conv2d(x, kernel_size=(2, 2), activation=tf.nn.relu, filters=3, name='Encoder_Conv_0')
    h = tf.layers.conv2d(h, kernel_size=(2, 2), strides=(2, 2), activation=tf.nn.relu, filters=32,
                          name='Encoder_Conv_1')
    h = tf.layers.conv2d(h, kernel_size=(2, 2), activation=tf.nn.relu, filters=32, name='Encoder_Conv_2')
    h = tf.layers.conv2d(h, kernel_size=(2, 2), activation=tf.nn.relu, filters=32, name='Encoder_Conv_3')
    h = tf.layers.flatten(h, name='Encoder_Flatten')
    h = tf.layers.dense(h, units=128, activation=tf.nn.relu, name='Encoder_FC')

    return tf.layers.dense(h, z_dim, name='Encoder_output')

encoders={'MNIST':encoder_map_mnist,'FMNIST':encoder_map_mnist,'CIFAR10':encoder_map_cifar10}
encoder_map=encoders[dataset_name]

def decoder_map_mnist(z):
    hidden_layer_neurons_count = 200

    h = z
    h = tf.layers.dense(h, hidden_layer_neurons_count, activation=tf.nn.relu, name='Decoder_1')
    h = tf.layers.dense(h, hidden_layer_neurons_count, activation=tf.nn.relu, name='Decoder_2')
    h = tf.layers.dense(h, hidden_layer_neurons_count, activation=tf.nn.relu, name='Decoder_3')

    h = tf.layers.dense(h, units=28*28, activation=tf.nn.sigmoid, name='Output')
    h = tf.reshape(h, [-1, 28, 28])
    return h


def decoder_map_cifar10(z):
    h = tf.layers.dense(z, units=128, activation=tf.nn.relu, name='Decoder_FC_0')
    h = tf.layers.dense(h, units=8192, activation=tf.nn.relu, name='Decoder_FC_1')
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

decoders={'MNIST':decoder_map_mnist,'FMNIST':decoder_map_mnist,'CIFAR10':decoder_map_cifar10}
decoder_map=decoders[dataset_name]

#def reconstruction_error(x, y):
#    diff = tf.layers.flatten(x) - tf.layers.flatten(y)
#    return tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=1))


def euclidean_norm_squared(X, axis=None):
    return tf.reduce_sum(tf.square(X), axis=axis)

def squared_euclidean_norm_reconstruction_error(input, output):
    return euclidean_norm_squared(input - output, axis=1)

def mean_squared_euclidean_norm_reconstruction_error(x, y):
    return tf.reduce_mean(squared_euclidean_norm_reconstruction_error(tf.layers.flatten(x), tf.layers.flatten(y)))

reconstruction_error=mean_squared_euclidean_norm_reconstruction_error


shapes={'MNIST':[28,28],'FMNIST':[28,28],'CIFAR10':[32,32,3]}
shape_placeholder=[None]+shapes[dataset_name]
#print(shape_placeholder)
#tensor_input_x = tf.placeholder(shape=[None, 28, 28], dtype=tf.float32)
tensor_input_x = tf.placeholder(shape=shape_placeholder, dtype=tf.float32)
tensor_z = encoder_map(tensor_input_x)
tensor_output_x = decoder_map(tensor_z)

#print(tr_images[0:8,:,:,:].shape)
#print(tr_images[8,:,:,:].shape)
#encoder_map_cifar10(tr_images[0:8,:,:,:])

"""##  Define distances

We define here some distances used to converge the latent space.
"""

#def euclidean_norm_squared(X, axis=None):
#    return tf.reduce_sum(tf.square(X), axis=axis)

def cw_distance(Z):   #distance from the paper https://arxiv.org/pdf/1805.09235.pdf 
    D = tf.cast(tf.shape(Z)[1], tf.float32)
    N = tf.cast(tf.shape(Z)[0], tf.float32)
    y = tf.pow(4/(3*N), 0.4)

    K = 1/(2*D-3)

    A1 = euclidean_norm_squared(tf.subtract(tf.expand_dims(Z, 0), tf.expand_dims(Z, 1)), axis=2)
    A = (1/(N**2)) * tf.reduce_sum((1/tf.sqrt(y + K*A1)))

    B1 = euclidean_norm_squared(Z, axis=1)
    B = (2/N)*tf.reduce_sum((1/tf.sqrt(y + 0.5 + K*B1)))

    return (1/tf.sqrt(1+y)) + A - B

def sw_distance(Z): #the Sliced Wasserstein distance
  D = tf.shape(Z)[1]#dimension
  N = tf.shape(Z)[0]#batch size

  nr_directions=1000 #number of directions on the sphere, some papers take 10'000
  randomed_normal = tf.random_normal(shape=(nr_directions, D))
  theta = randomed_normal / tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(randomed_normal), axis=1)), (-1, 1))  

  #project the latent samples Z
  proj_samples = tf.keras.backend.dot(Z, tf.transpose(theta))
  #sample from the normal distribution and project
  sampled_normal = tf.random_normal(shape=(N, D))
  proj_normal = tf.keras.backend.dot(sampled_normal, tf.transpose(theta))

  return tf.reduce_mean( (tf.sort(proj_samples,axis=0)
                        - tf.sort(proj_normal,  axis=0) )**2 )


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

    smalleps=1e-4
    return c0+tf.reduce_mean(tf.sqrt(dist_real_Z+c1)) \
      -0.5*tf.reduce_mean(tf.sqrt(distZZ+1e-6))+0.5/Nf+2.*tf.sqrt(smalleps) 
    #Note: 0.5/Nf has been added in order to compensate the "1" set on the diagonal
    #if necessary to apply the log in the cost, add a small constant for smoothness near zero

"""# Choice of the driver distance and of the reporting metrics

Choose the "driver" distance and the report distance. 

Note: using the Sliced-Wasserstein distance is time-consuming because random directions are drawn on the high dimensional sphere.
"""

#to speed up computations we only test a "driver" distance, by default XSH and a report metric

#this is the driver distance
main_distance=xs_distance
name_main_distance='Radon-Sobolev dist'

#these other two are only used for reporting
other_distance=cw_distance
name_other_distance='Cramer-Wold dist'
other_distance2=sw_distance
name_other_distance2='Sliced Wasserstein dist'


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
tensor_other_distance=tf.stop_gradient(other_distance(tensor_z))
tensor_other_distance2=tf.stop_gradient(other_distance2(tensor_z))
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
other_distances = list()
other_distances2 = list()
costs = list()


#print some outputs
def plot_results(iter_no):
    """#Reconstructed images"""
    
    nb_reconstructions=64
    nb_reconstructions=32
    reconstruction_indexes = list(np.random.randint(0, len(tr_images), nb_reconstructions))
    
    reconstruction_images = list()
    for i,train_index in enumerate(reconstruction_indexes):
      input_image = tr_images[train_index,...]
      one_latent, reconstructed_image = sess.run([tensor_z, tensor_output_x], feed_dict={tensor_input_x: [input_image]}) 
      reconstruction_images.extend([input_image, reconstructed_image[0]])
    
    fig = plt.figure(figsize=(8, 8)) 
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    plt.axis('off')
    
    for i, img in enumerate(reconstruction_images):
    #    sub = fig.add_subplot(8, 8*2, i + 1)
        sub = fig.add_subplot(8, 4*2, i + 1)
        sub.axis('off')
        sub.imshow(img)
    plt.savefig(np.str(iter_no)+'_reconstructions.png')

    """### Sampled images"""
    
    x = np.random.randn(64,z_dim) #plot a 8x8 matrix of sampled images 
    decoded_images = sess.run(tensor_output_x, feed_dict={tensor_z: x})
    fig = plt.figure(figsize=(8, 8)) 
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    plt.axis('off')
    for i in range(64):
        sub = fig.add_subplot(8, 8, i + 1,)
        sub.axis('off')
        sub.imshow(decoded_images[i])
    plt.savefig(np.str(iter_no)+'_sampled.png')
    
    """## Interpolations"""
    
    interpolations_step_count = 16
    interpolations_count = 7
    
    interpolation_indexes = list(np.reshape(np.random.randint(0, len(tr_images), interpolations_count * 2), [-1, 2]))
    images = list()
    
    for i, train_indexes in enumerate(interpolation_indexes):
        first_index = train_indexes[0]
        second_index = train_indexes[1]
    
        first_image = tr_images[first_index]
        second_image = tr_images[second_index]
    
        first_latent, first_image_decoded = sess.run([tensor_z, tensor_output_x], feed_dict={tensor_input_x: [first_image]})
        first_latent, first_image_decoded = first_latent[0], first_image_decoded[0]
        second_latent, second_latent_decoded = sess.run([tensor_z, tensor_output_x], feed_dict={tensor_input_x: [second_image]}) 
        second_latent, second_latent_decoded = second_latent[0], second_latent_decoded[0]
        images.extend([first_image, first_image_decoded])
    
        latent_step = (second_latent - first_latent) / (interpolations_step_count + 1)
        for j in range(interpolations_step_count):
            next_latent = first_latent + (j + 1) * latent_step
            next_image = sess.run(tensor_output_x, feed_dict={tensor_z: [next_latent]})[0]
            images.append(next_image)
    
        images.extend([second_latent_decoded, second_image])
    
    #print(len(images))
    
    fig = plt.figure(figsize=(10,3.5)) 
    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    plt.axis('off')
    for i, img in enumerate(images):
        sub = fig.add_subplot(7, 20, i + 1)
        sub.axis('off')
        sub.imshow(img)
    
    plt.savefig(np.str(iter_no)+'_interpolations.png')



sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for i in range(1, epochs_count+1):
    sys.stdout.flush()
    sys.stderr.flush()
    
    np.random.shuffle(tr_images)
    for b in np.array_split(tr_images, batch_size):
        sess.run(train_ops, feed_dict={tensor_input_x: b})
        
    if (i == epochs_count) or (i==1) or (i % report_frequency == 0):#report
        saver.save(sess, 'model_backup',global_step=i)    
        rec_error, val_main_distance,\
         val_other_distance, val_other_distance2, cost = sess.run(
             [tensor_rec_error, tensor_main_distance, 
             tensor_other_distance,tensor_other_distance2,tensor_cost_function],
              feed_dict={tensor_input_x: validation_images})
        print('Epoch ', i, 'Rec error: ', rec_error, 
              name_main_distance, val_main_distance,
              name_other_distance, val_other_distance,
              name_other_distance2, val_other_distance2,
              'Cost: ', cost)

        #log various costs
        main_distances.append(val_main_distance)
        other_distances.append(val_other_distance)
        other_distances2.append(val_other_distance2)
        reconstruction_errors.append(rec_error)
        costs.append(cost)
        epochs.append(i)
        
        #produce images
        plot_results(i)
        
        #compute FID score: generate nr_test_img images and dump the array in a file
        x = np.random.randn(nr_test_img,z_dim) 
        sampled_img = sess.run(tensor_output_x, feed_dict={tensor_z: x})
        np.save('generated_'+np.str(i),sampled_img)

#final plots
plt.figure(3,figsize=(16,4))
plt.subplot(1,5,1)
plt.loglog(epochs,main_distances, label=name_main_distance)
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1,5,2)
plt.loglog(epochs,other_distances, label=name_other_distance)
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1,5,3)
plt.loglog(epochs,other_distances2, label=name_other_distance2)
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

    
