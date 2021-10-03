# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 20:11:11 2021
Distribute only on the same licence as the repository where this file is located

@author: Gabriel Turinici
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('mnist_encoder_decoder_weights.pkl', 'rb') as file:
    myvar = pickle.load(file)
    

#myvar.keys()
#dict_keys(['Encoder_1/kernel:0', 'Encoder_1/bias:0', 'Encoder_2/kernel:0',
# 'Encoder_2/bias:0', 'Encoder_3/kernel:0', 'Encoder_3/bias:0', 
# 'Encoder_output/kernel:0', 'Encoder_output/bias:0', 'Decoder_1/kernel:0',
# 'Decoder_1/bias:0', 'Decoder_2/kernel:0', 'Decoder_2/bias:0', 
#'Decoder_3/kernel:0', 'Decoder_3/bias:0', 'Output/kernel:0', 'Output/bias:0'])


shape=(28,28)#mnist
z_dim=myvar['Encoder_output/bias:0'].shape[0] #=8
hidden_layer_neurons_count = myvar['Encoder_output/kernel:0'].shape[0] #200

def encoder(x):
    '''

    input: a 3D array batch_size x 28 x 28

    returns: the encoded vector, size batch_size x z_dim

    '''
    assert x.shape[-2]==shape[0]
    assert x.shape[-1]==shape[1]

    batch_size=x.shape[0]
    total_size=x.shape[-2]*x.shape[-1]
    
    z=x.reshape(batch_size,total_size)#only the last two dimensions
    
    for k in range(3):
        z  = np.maximum( z@ myvar['Encoder_'+str(k+1)+'/kernel:0']+ 
                        myvar['Encoder_'+str(k+1)+'/bias:0'],0)
        
    z =  z@myvar['Encoder_output/kernel:0']+myvar['Encoder_output/bias:0']        
    return z
   
#test encoder
encoder(np.random.rand(10,shape[0],shape[1]))

def sigmoid(Z):
 """
 The sigmoid function in numpy
 """
 return 1./(1.+(np.exp((-np.clip(Z,-20.,20.)))))


def decoder(x):
    '''

    input: a 2D array batch_size x z_dim

    returns: the decoded vector,  batch_size x 28x28

    '''
    assert x.shape[1]==z_dim
    batch_size=x.shape[0]
    
    z=x.copy()
    
    for k in range(3):
        z  = np.maximum( z@ myvar['Decoder_'+str(k+1)+'/kernel:0']+ 
                        myvar['Decoder_'+str(k+1)+'/bias:0'],0)
        
    z =  sigmoid(z@myvar['Output/kernel:0']
                 +myvar['Output/bias:0'])
    h = np.reshape(z, [-1, 28, 28])
 
    assert h.shape[0]==batch_size
    return h


# Test encoder and decoder

tr_images=np.load('tr_images2.npz')['arr_0']
latent_images=np.load('latent_images_array2.npz')['arr_0']


index=[34,12345,456,23,44,78]
#index=list(range(0,20000))# 20k of them !
#index=list(range(0,len(tr_images)))#all of them !

index_in_batch=np.random.choice(len(index))#select which one to plot
image_to_encode=tr_images[index]
true_latent=latent_images[index][index_in_batch]

all_latent=encoder(image_to_encode)
our_latent=all_latent[index_in_batch]#only take one index
our_image=decoder(encoder(image_to_encode))
latent_err=np.max(np.abs(our_latent-true_latent))

plt.figure(4)
plt.subplot(2,2,1)
plt.imshow(image_to_encode[index_in_batch],cmap='gray')
plt.title('image to encode')
plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(our_image[index_in_batch],cmap='gray')
plt.title('our reconstructed image')
plt.axis('off')
plt.subplot(1,2,2)
plt.plot(list(range(z_dim)),true_latent,'g*-',
         list(range(z_dim)),our_latent,'b-o')
plt.legend(['true latent','our latent'])
plt.title('max latent err='+str(latent_err))



#encode some axial images i.e. images with 1 on some coordinate and 0 elsewhere
axial_images=decoder(np.eye(z_dim))
axial_images2=decoder(0.5*np.eye(z_dim))
plt.figure(4,figsize=(6,2))
for ii in range(z_dim):
    plt.subplot(2,z_dim,ii+1)
    plt.imshow(axial_images[ii],cmap='gray')
    plt.axis('off')
    plt.subplot(2,z_dim,ii+1+z_dim)
    plt.imshow(axial_images2[ii],cmap='gray')
    plt.axis('off')
