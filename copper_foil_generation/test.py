import os
import pandas as pd
import numpy as np
from load import *
from model import *
from util import *



n_epochs = 600
learning_rate = 0.00002
batch_size = 80
image_shape = [90,90,3]  
dim_z = 512
dim_latent_codes=10
dim_W1 = 512#1024
dim_W2 = 256#512
dim_W3 = 128#256
dim_W4 = 64#128
dim_W5= 32
dim_Wl = 3
num_mixs=15
latent_code_mean=np.random.uniform(-1,1,size=(num_mixs,dim_latent_codes))
description='InfoGAN_new'
visualize_dim=60


infodcgan_model = InfoDCGAN(
        batch_size=batch_size,
        image_shape=image_shape,
        dim_z=dim_z,
        dim_latent_codes=dim_latent_codes,
        dim_W1=dim_W1,
        dim_W2=dim_W2,
        dim_W3=dim_W3,
        dim_W4=dim_W4,
        dim_W5=dim_W5
        )

Z_tf, latent_codes_tf, image_tf, d_cost_tf, g_cost_tf, aux_D_cost_tf, aux_G_cost_tf, K, gamma_tf = infodcgan_model.build_model()

Z_tf_sample, latent_tf_sample, image_tf_sample = infodcgan_model.samples_generator(batch_size=visualize_dim)

init = tf.global_variables_initializer()

#Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim,dim_z))
Z_np_sample = np.random.normal(0,1,size=(visualize_dim,dim_z))
latent_codes_np_sample=mixtures_noise(latent_code_mean,num_mixs,visualize_dim, dim_latent_codes)

###################
#### saver
saver=tf.train.Saver()
save_path=os.getcwd()+'/'+description+'_model/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
                         

sess = tf.Session()
sess.run(init)
###################
### load model
save_path=os.getcwd()+'/'+description+'_model/'
if os.path.exists(save_path):
    saver=tf.train.Saver()
    saver.restore(sess,tf.train.latest_checkpoint(save_path))
else:
    print 'load path do not exist'
##########################
# generate samples by shift

num_repeat=visualize_dim/num_mixs
#test_z=np.random.uniform(-1, 1, size=(num_repeat,dim_z))
for i in range(100):
    test_z=np.random.normal(0, 1, size=(num_repeat,dim_z))
    test_z=np.repeat(test_z,num_mixs, axis=0)
    test_latent_codes=np.random.normal(0,0.1,size=(num_mixs,dim_latent_codes))+latent_code_mean
    test_latent_codes=np.reshape(np.repeat([test_latent_codes], visualize_dim/num_mixs, axis=0),(visualize_dim, dim_latent_codes))

    generated_samples = sess.run(
                        image_tf_sample,
                        feed_dict={
                            Z_tf_sample:test_z,
                            latent_tf_sample:test_latent_codes
                            })
    generated_samples = (generated_samples + 1.)/2.
    save_visualization(generated_samples, (num_repeat,num_mixs), save_path='./shift_image/shift_mixs_'+str(i)+'.png')
for i in range(1000):
    test_z=np.random.normal(0, 1, size=(visualize_dim,dim_z))
    test_latent_codes=np.random.normal(0,0.1,size=(num_mixs,dim_latent_codes))+latent_code_mean
    test_latent_codes=np.reshape(np.repeat([test_latent_codes], visualize_dim/num_mixs, axis=0),(visualize_dim, dim_latent_codes))
    generated_samples = sess.run(
                        image_tf_sample,
                        feed_dict={
                            Z_tf_sample:test_z,
                            latent_tf_sample:test_latent_codes
                            })
    generated_samples = (generated_samples + 1.)/2.
    for j in range(visualize_dim):
        save_path='./Generated_Images/batch_'+str(i)+'_sample_'+str(j)+'.png'
        scipy.misc.imsave(save_path, generated_samples[j,:,:,:])
