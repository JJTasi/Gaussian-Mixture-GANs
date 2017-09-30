import os,random
import pandas as pd
import numpy as np
from load_MNIST import *
from model import *
from util import *


batch_size = 128
image_shape = [28,28,3]  
dim_z = 100
dim_latent_codes=28
dim_W1 = 512#1024
dim_W2 = 256#512
dim_W3 = 128#256
dim_W4 = 64#128
dim_W5 = 32
dim_W6 = 3
num_mixs=10
description='InfoGAN_new'
visualize_dim=100
gamma=1.0
latent_code_mean=np.loadtxt('latent_code_mean.txt')
infodcgan_model = InfoDCGAN(
        num_mixs=num_mixs,
        batch_size=batch_size,
        image_shape=image_shape,
        dim_z=dim_z,
        dim_latent_codes=dim_latent_codes,
        dim_W1=dim_W1,
        dim_W2=dim_W2,
        dim_W3=dim_W3,
        dim_W4=dim_W4,
        dim_W5=dim_W5,
        dim_W6=dim_W6,
        gamma=gamma
        )

Z_tf, Y_tf, latent_codes_tf, component_list_tf, image_tf, d_cost_tf, g_cost_tf, aux_G_cost_tf, aux_M_tf, aux_S_tf = infodcgan_model.build_model()
Z_tf_sample, latent_tf_sample, image_tf_sample = infodcgan_model.samples_generator(batch_size=visualize_dim)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
###################
### load model
save_path=os.getcwd()+'/'+description+'_model/'
if os.path.exists(save_path):
    #saver=tf.train.import_meta_graph(save_path+'model.meta')
    #saver.restore(sess,tf.train.latest_checkpoint(save_path))
    saver=tf.train.Saver()
    saver.restore(sess,tf.train.latest_checkpoint(save_path))
else:
    print 'load path do not exist'
    
for i in range(100):
    num_repeat=visualize_dim/num_mixs
    #test_z=np.random.uniform(-1, 1, size=(num_repeat,dim_z))
    test_z=np.random.normal(0, 1, size=(num_repeat,dim_z))
    test_z=np.repeat(test_z,num_mixs, axis=0)
    #test_latent_codes=np.random.normal(0,0.1,size=(num_mixs,dim_latent_codes))+latent_code_mean
    #test_latent_codes=np.reshape(np.repeat([test_latent_codes], visualize_dim/num_mixs, axis=0),(visualize_dim, dim_latent_codes))
    #test_latent_codes=np.random.normal(0,1,size=(visualize_dim,dim_latent_codes))+np.reshape(np.repeat([latent_code_mean],visualize_dim/num_mixs,axis=0),(visualize_dim,dim_latent_codes))
    #test_latent_codes=truncate_normal(0,1,size=(visualize_dim,dim_latent_codes))+np.reshape(np.repeat([latent_code_mean],visualize_dim/num_mixs,axis=0),(visualize_dim,dim_latent_codes))
    compontent_list=np.asarray(range(num_mixs))
    compontent_list=np.tile(compontent_list,visualize_dim/num_mixs)
    test_latent_codes=circle_sampler(visualize_dim, dim_latent_codes, compontent_list, num_mixs)
    generated_samples = sess.run(
                        image_tf_sample,
                        feed_dict={
                            Z_tf_sample:test_z,
                            latent_tf_sample:test_latent_codes
                            })
    generated_samples = (generated_samples + 1.)/2.
    save_visualization(generated_samples, (num_repeat,num_mixs), save_path='./shift_image/shift_mixs_'+str(i)+'.png')
