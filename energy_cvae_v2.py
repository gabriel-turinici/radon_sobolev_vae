# -*- coding: utf-8 -*-
"""energy_cvae_v2.ipynb

#Energy Convolutional Variational Autoencoder : EVAE#

#Energy VAE#
This is a modification of the Tensorflow CVAE tutorial (as of January 2025 https://www.tensorflow.org/tutorials/generative/cvae) inititally licensed under Apache 2.0 license available at
https://www.apache.org/licenses/LICENSE-2.0

This modification uses the "energy" distance to be used in the VAE, cf the  implementation of the Radon-Sobolev Variational Auto-Encoder as described in arXiv:1911.13135
Published version: Neural Networks Volume 141, September 2021, Pages 294-305

With slight modifications this can accomodate more general, Radon-Sobolev or Huber type distances, cf. references.

(c) Gabriel Turinici 2025

## Setup
"""

# to generate gifs
!pip install imageio

from IPython import display

import glob
#import imageio
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import time
from tqdm import tqdm

"""## Load the MNIST dataset and preprocess

"""

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  #enable this to binarize image
  #return np.where(images > .5, 1.0, 0.0).astype('float32')
  return images.astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = 60000
batch_size = 128
test_size = 10000

"""## Use *tf.data* to batch and shuffle the data"""

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

"""## Define the encoder and decoder networks with *tf.keras.Sequential*

We follow the intial Tensorflow tutorial and use two small ConvNets for the encoder and decoder networks. Encoder transforms initial image into a vector $z$ in the latent space. Decoder transforms this back to an image.

"""

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self,nb_samples=100):
    """ samples from the latent space target distribution which is a gaussian"""
    latent_vectors=tf.random.normal(shape=(nb_samples, self.latent_dim))
    return self.decode(latent_vectors, apply_sigmoid=True)

  def encode(self, x):
    return self.encoder(x)

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

"""#Define the energy distance to a Gaussian#"""

import tensorflow as tf

@tf.function
def euclidean_norm_squared(X, axis=None):
  return tf.reduce_sum(tf.square(X), axis=axis)

# the 'energy' distance to a Gaussian
@tf.function
def energy_distance_to_gaussian(Z):
    """
    Computes the energy distance to a multi-dimensional Gaussian. See the reference for
    the formulas.
    Input Z tensor of shape (batch_size=Nf,latent_dim=Df)
    """

    Nf = tf.stop_gradient(tf.cast(tf.shape(Z)[0], tf.float32)) # batch size as float
    Df = tf.stop_gradient(tf.cast(tf.shape(Z)[1], tf.float32)) # dimension as float

    f0 = tf.stop_gradient(
            (tf.sqrt(2.) - 1.) * tf.exp(tf.math.lgamma(Df / 2.0 + 0.5)
            - tf.math.lgamma(Df / 2.0))
            )
    ddf0 = tf.stop_gradient(
    tf.exp(tf.math.lgamma(0.5 + Df / 2.) - tf.math.lgamma(1. + Df / 2.)) / tf.sqrt(2.))
    c0 = f0 - 1. / ddf0 # first constant in function
    c1 = 1 / (ddf0**2) # second constant

    dist_real_Z = euclidean_norm_squared(Z, axis=1)
    distZZ = euclidean_norm_squared(tf.subtract(tf.expand_dims(Z, 0), tf.expand_dims(Z, 1)), axis=2)

    # set diag = 1 because of the singularity of sqrt'(x) in zero. This is a numerical
    # issue due to low precision of the GPU computations; it is compensated
    # exactly in the return value
    distZZ = tf.linalg.set_diag(distZZ, tf.ones([tf.stop_gradient(tf.shape(Z)[0])]), name=None)
    # remark: this is similar to a pseudo-Huber function

    smalleps = 1e-4
    return c0 + tf.reduce_mean(tf.sqrt(dist_real_Z + c1)) \
      - 0.5 * tf.reduce_mean(tf.sqrt(distZZ + 1e-6)) + 0.5 / Nf + 2. * tf.sqrt(smalleps)

"""## Define the loss function and the optimizer

VAEs train by minimizing the mean L2 reconstruction loss plus a the energy distance between the distribution of latent vectors and the normal distribution;
in practice, optimize the expectation of :

$$ \|Decoder(Encoder(x))-x \|^2_{L^2} + dist(\text{latent distrib}, \text{target distrib})^2$$.


"""

optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def compute_loss(model, x,coefficient=1.0):
  latent_vectors = model.encode(x)
  xrec= model.decode(latent_vectors)
  reconstruction_error= tf.reduce_mean(tf.square(x-xrec))
  distance= energy_distance_to_gaussian(latent_vectors)
  return reconstruction_error+ distance,reconstruction_error,distance


@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss,rec_err,dist = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss,rec_err,dist

"""## Training

* Start by iterating over the dataset
* During each iteration, pass the image to the encoder, decoder to obtain the reconstructed images and the latent vectors;
* construct the loss

### Generating images

* After training generate some images: generate latent vectors from the target distribution (multivariate normal) and then pass them through the decoder.
"""

epochs = 40
# set the dimension of the latent space =2 for visualization
latent_dim = 4
num_examples_to_generate = 16

# keep the random vector constant for generation to see the evolution
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

def reconstruct_images(model, epoch, test_sample):
  """ generates images from test samples, saves in jpg file
  We plot initial images then reconstructed image"""
  z = model.encode(test_sample)
  predictions = model.decode(z).numpy()
  fig = plt.figure(figsize=(0.75*8,0.75*4))
  for i in range(predictions.shape[0]):
    plt.subplot(4, 8, 2*i + 1)
    plt.imshow(test_sample[i, :, :, 0], cmap='gray')
    plt.axis('off')
    plt.subplot(4, 8, 2*i + 2)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  plt.tight_layout()# minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  return z,predictions

def generate_and_plot(model, z=random_vector_for_generation,label=""):
  """ generates images from the latent z, saves in jpg file"""
  predictions = model.decode(z).numpy()
  fig = plt.figure(figsize=(8*0.75,2*0.75))
  for i in range(predictions.shape[0]):
    plt.subplot(2, 8, i +1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')
  plt.tight_layout()# minimizes the overlap between 2 sub-plots
  plt.savefig('generated'+label+'.jpg')
  plt.show()
  return None

def generate_latent_representation(model, epoch, test_sample):
  """ This function generate a latent space representation of a test sample
  and plots it if the dimension is 2"""
  z = model.encode(test_sample)
  if(z.shape[-1]==2):
    fig = plt.figure(figsize=(4, 4))
    plt.scatter(z[:,0],z[:,1],s=0.5)
    plt.tight_layout()# minimizes the overlap between 2 sub-plots
    plt.savefig('latent_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
  return z

# Pick a sample of the test set for generating output images
#assert batch_size >= num_examples_to_generate
#for test_batch in test_dataset.take(1):
#  test_sample = test_batch[0:num_examples_to_generate, :, :, :]
test_sample=test_images[np.random.choice(test_images.shape[0],num_examples_to_generate),:,:,:]
num_latent_points=500
test_sample_for_latent=test_images[np.random.choice(test_images.shape[0],num_latent_points),:,:,:]

_=reconstruct_images(model, 0, test_sample)
generate_and_plot(model)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in (tqdm_pbar:= tqdm(train_dataset)):
    loss,rec_err,dist=train_step(model, train_x, optimizer)
    #generate some messages containing main components of the loss function
    tqdm_pbar.set_description(f"Epoch: {epoch} Loss: {loss.numpy():.4f}, rec_err: {rec_err.numpy():.4f} dist: {dist.numpy():.4f}")
  end_time = time.time()

  loss_mean = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    loss,rec_err,dist=compute_loss(model, test_x)
    loss_mean(loss)
  loss_mean_value = loss_mean.result()
#  display.clear_output(wait=False)
  print('Epoch: {}, Test set loss: {}, time elapse for current epoch: {}'
        .format(epoch, loss_mean_value, end_time - start_time))
  _,_=reconstruct_images(model, epoch, test_sample)
  generate_and_plot(model,random_vector_for_generation,label=f"_iter{epoch}")
  if(model.latent_dim==2):
    _=generate_latent_representation(model, epoch, test_sample_for_latent)

"""### Display a generated image from the last training epoch"""

def display_image(epoch_no):
  #return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch-1))

plt.imshow(display_image(epoch))
_=plt.axis('off')  # Display images

"""### Display an animated GIF of all the saved images"""

anim_file = 'cvae.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

#import tensorflow_docs.vis.embed as embed
#embed.embed_file(anim_file)
from PIL import Image
from IPython.display import display
# Load the GIF file
gif = Image.open(anim_file)
# Display the GIF
display(gif)