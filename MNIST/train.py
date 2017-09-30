import os,random
import pandas as pd
import numpy as np
from load_MNIST import *
from model import *
from util import *




n_epochs = 100
learning_rate = 0.00001
batch_size = 128
image_shape = [28,28,3]
dim_latent_codes=128
dim_W1 = 512#1024
dim_W2 = 256#512
dim_W3 = 128#256
dim_W4 = 64#128
dim_W5 = 32
dim_W6 = 3
num_mixs=10

description='GM_GANs'
visualize_dim=100
path=""

trX, vaX, teX, trY, vaY, teY = mnist_with_valid_set(path)
gmgan_model = GMGANs(
        num_mixs=num_mixs,
        batch_size=batch_size,
        image_shape=image_shape,
        dim_latent_codes=dim_latent_codes,
        dim_W1=dim_W1,
        dim_W2=dim_W2,
        dim_W3=dim_W3,
        dim_W4=dim_W4,
        dim_W5=dim_W5,
        dim_W6=dim_W6
        )

Y_tf, latent_codes_tf, component_list_tf, image_tf, d_cost_tf, g_cost_tf, aux_G_cost_tf, aux_M_tf, aux_S_tf = gmgan_model.build_model()



discrim_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
aux_vars = filter(lambda x: x.name.startswith('aux'), tf.trainable_variables())


train_op_discrim = tf.train.RMSPropOptimizer(learning_rate).minimize(d_cost_tf, var_list=discrim_vars)
train_op_gen = tf.train.RMSPropOptimizer(learning_rate).minimize(g_cost_tf, var_list=gen_vars+aux_vars)
clip_vars = filter(lambda x: x.name.startswith('discrim_W'), tf.trainable_variables())
clip_var_c=[tf.assign(vars, tf.clip_by_value(vars, -0.1, 0.1)) for vars in clip_vars]

latent_tf_sample, image_tf_sample = gmgan_model.samples_generator(batch_size=visualize_dim)

init = tf.global_variables_initializer()

#circle shift
compontent_list=np.asarray(range(num_mixs))
compontent_list=np.tile(compontent_list,visualize_dim/num_mixs)
latent_codes_shift_sample=circle_sampler(visualize_dim, dim_latent_codes, compontent_list, num_mixs)
###################
#### saver
saver=tf.train.Saver()
save_path=os.getcwd()+'/'+description+'_model/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
                         
iterations = 0
k = 3
sess = tf.Session()
sess.run(init)
print 'Model is initialized'
d_loss_re=[]
g_loss_re=[]
aux_loss_re=[]
k_re=[]

for epoch in range(n_epochs):
    for start, end in zip(
            range(0, trX.shape[0], batch_size),
            range(batch_size, trX.shape[0], batch_size)
            ):

        batch_images = trX[start:end]
        batch_label = trY[start:end]
        
        batch_compontent_list=np.random.randint(num_mixs, size=[batch_size])
        batch_latent_codes=circle_sampler(batch_size, dim_latent_codes, batch_compontent_list, num_mixs)

        
        if np.mod( iterations, k ) != 0:
            _, gen_loss_val ,aux_G_loss_val= sess.run(
                    [train_op_gen, g_cost_tf, aux_G_cost_tf],
                    feed_dict={
                        latent_codes_tf: batch_latent_codes,
                        component_list_tf: batch_compontent_list,
                        Y_tf: batch_label,
                        image_tf:batch_images
                        })
            print "=========== updating G =========="
            print "iteration:", iterations, " epoch " , epoch
            print "gen loss:", gen_loss_val
            print "aux gen loss", aux_G_loss_val
            g_loss_re.append(gen_loss_val)
            aux_loss_re.append(aux_G_loss_val)
        else:
            _, discrim_loss_val = sess.run(
                    [train_op_discrim, d_cost_tf],
                    feed_dict={
                        latent_codes_tf: batch_latent_codes,
                        component_list_tf: batch_compontent_list,
                        Y_tf: batch_label,
                        image_tf:batch_images
                        })
            sess.run(clip_var_c)
            print "=========== updating D =========="
            print "iteration:", iterations, " epoch " , epoch
            print "discrim loss:", discrim_loss_val
            d_loss_re.append(discrim_loss_val)
            
        if np.mod(iterations, 100) == 0:
            
            save_visualization(trX[0:visualize_dim], (10,10), save_path='data.png')
            generated_samples = sess.run(
                    image_tf_sample,
                    feed_dict={
                        latent_tf_sample:latent_codes_shift_sample
                        })
            generated_samples = (generated_samples + 1.)/2.
            save_visualization(generated_samples, (10,10), save_path='./vis/sample_'+str(iterations/100)+'.png')
            np.savetxt('g_loss.txt',g_loss_re)
            np.savetxt('d_loss.txt',d_loss_re)
            np.savetxt('aux_loss.txt',aux_loss_re)
            np.savetxt('k_val.txt', k_re)
        iterations += 1
np.savetxt('g_loss.txt',g_loss_re)
np.savetxt('d_loss.txt',d_loss_re)
np.savetxt('aux_loss.txt',aux_loss_re)
np.savetxt('k_val.txt', k_re)
saver.save(sess, save_path+'model')
