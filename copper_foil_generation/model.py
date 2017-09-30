import tensorflow as tf
import numpy as np
from math import cos, sin

def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)#std

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b

    else:
        raise NotImplementedError

    return X

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return -(t * tf.log(o) + (1.- t)*tf.log(1. - o))

def cal_mixture_loss(num_mix, dim_latent_code, aux_latent_code_mean, aux_latent_code_log_sigma, compontent_list):
    loss=0
    for i in range(num_mix):
        shift = 1
        r = 2.0 * np.pi / float(num_mix) * float(i)
        mix_mean=np.tile([shift*cos(r),shift*sin(r)],[dim_latent_code/2])
        condition=tf.equal(compontent_list,i*tf.ones_like(compontent_list,dtype=tf.int32))
        index=tf.where(condition)
        aux_mc_mean=tf.gather(aux_latent_code_mean, index)
        aux_mc_sigma=tf.exp(tf.gather(aux_latent_code_log_sigma,index))
        mean_loss=tf.reduce_sum(tf.square(mix_mean-aux_mc_mean),axis=2)
        var_loss=tf.reduce_sum(aux_mc_sigma+1-2*tf.sqrt(tf.abs(aux_mc_sigma)),axis=2)
        tmp=tf.reduce_mean(mean_loss)+tf.reduce_mean(var_loss)
        loss=loss+tf.reduce_mean(tmp)
        
    return loss

class GMGANs():
    def __init__(
            self,
            num_mixs=10,
            batch_size=100,
            image_shape=[100,100,3],
            dim_latent_codes=10,
            dim_W1=1024,
            dim_W2=512,
            dim_W3=256,
            dim_W4=128,
            dim_W5=128,
            dim_W6=3,
            ):

        self.num_mixs=num_mixs
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_latent_codes=dim_latent_codes

        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_W4 = dim_W4
        self.dim_W5 = dim_W5
        self.dim_W6 = dim_W6

       
        self.gen_W1 = tf.Variable(tf.truncated_normal([dim_latent_codes, dim_W1*23*23], stddev=0.002), name='gen_W1')
        self.gen_bn_g1 = tf.Variable( tf.truncated_normal([dim_W1*23*23], mean=1.0, stddev=0.002), name='gen_bn_g1')
        self.gen_bn_b1 = tf.Variable( tf.zeros([dim_W1*23*23]), name='gen_bn_b1')

        self.gen_W2 = tf.Variable(tf.truncated_normal([4,4,dim_W2, dim_W1], stddev=0.002), name='gen_W2')
        self.gen_bn_g2 = tf.Variable( tf.truncated_normal([dim_W2], mean=1.0, stddev=0.002), name='gen_bn_g2')
        self.gen_bn_b2 = tf.Variable( tf.zeros([dim_W2]), name='gen_bn_b2')

        self.gen_W3 = tf.Variable(tf.truncated_normal([4,4,dim_W3, dim_W2], stddev=0.02), name='gen_W3')
        self.gen_bn_g3 = tf.Variable( tf.truncated_normal([dim_W3], mean=1.0, stddev=0.02), name='gen_bn_g3')
        self.gen_bn_b3 = tf.Variable( tf.zeros([dim_W3]), name='gen_bn_b3')

        self.gen_W4 = tf.Variable(tf.truncated_normal([4,4,dim_W4, dim_W3], stddev=0.02), name='gen_W4')
        self.gen_bn_g4 = tf.Variable( tf.truncated_normal([dim_W4], mean=1.0, stddev=0.02), name='gen_bn_g4')
        self.gen_bn_b4 = tf.Variable( tf.zeros([dim_W4]), name='gen_bn_b4')
        
        self.gen_W5 = tf.Variable(tf.truncated_normal([4,4,dim_W5, dim_W4], stddev=0.02), name='gen_W5')
        self.gen_bn_g5 = tf.Variable( tf.truncated_normal([dim_W5], mean=1.0, stddev=0.02), name='gen_bn_g5')
        self.gen_bn_b5 = tf.Variable( tf.zeros([dim_W5]), name='gen_bn_b5')

        self.gen_W6 = tf.Variable(tf.truncated_normal([4,4,dim_W6, dim_W5], stddev=0.02), name='gen_W6')
        
        
       
        

        self.discrim_W1 = tf.Variable(tf.truncated_normal([4,4,dim_W6,dim_W4], stddev=0.02), name='discrim_W1')
        self.discrim_bn_g1 = tf.Variable( tf.truncated_normal([dim_W4], mean=1.0, stddev=0.02), name='discrim_bn_g1')
        self.discrim_bn_b1 = tf.Variable( tf.zeros([dim_W4]), name='discrim_bn_b1')

        self.discrim_W2 = tf.Variable(tf.truncated_normal([4,4,dim_W4,dim_W3], stddev=0.02), name='discrim_W2')
        self.discrim_bn_g2 = tf.Variable( tf.truncated_normal([dim_W3], mean=1.0, stddev=0.02), name='discrim_bn_g2')
        self.discrim_bn_b2 = tf.Variable( tf.zeros([dim_W3]), name='discrim_bn_b2')

        self.discrim_W3 = tf.Variable(tf.truncated_normal([4,4,dim_W3,dim_W2], stddev=0.02), name='discrim_W3')
        self.discrim_bn_g3 = tf.Variable( tf.truncated_normal([dim_W2], mean=1.0, stddev=0.02), name='discrim_bn_g3')
        self.discrim_bn_b3 = tf.Variable( tf.zeros([dim_W2]), name='discrim_bn_b3')

        self.discrim_W4 = tf.Variable(tf.truncated_normal([4,4,dim_W2,dim_W1], stddev=0.02), name='discrim_W4')
        self.discrim_bn_g4 = tf.Variable( tf.truncated_normal([dim_W1], mean=1.0, stddev=0.02), name='discrim_bn_g4')
        self.discrim_bn_b4 = tf.Variable( tf.zeros([dim_W1]), name='discrim_bn_b4')

        self.discrim_W5 = tf.Variable(tf.truncated_normal([6*6*dim_W1,1], stddev=0.02), name='discrim_W5')
        
        self.aux_q_W1 = tf.Variable(tf.truncated_normal([4,4,dim_W6,dim_W4], stddev=0.02), name='aux_q_W1')
        self.aux_q_bn_g1 = tf.Variable( tf.truncated_normal([dim_W4], mean=1.0, stddev=0.02), name='aux_q_bn_g1')
        self.aux_q_bn_b1 = tf.Variable( tf.zeros([dim_W4]), name='aux_q_bn_b1')

        self.aux_q_W2 = tf.Variable(tf.truncated_normal([4,4,dim_W4,dim_W3], stddev=0.02), name='aux_q_W2')
        self.aux_q_bn_g2 = tf.Variable( tf.truncated_normal([dim_W3], mean=1.0, stddev=0.02), name='aux_q_bn_g2')
        self.aux_q_bn_b2 = tf.Variable( tf.zeros([dim_W3]), name='aux_q_bn_b2')

        self.aux_q_W3 = tf.Variable(tf.truncated_normal([4,4,dim_W3,dim_W2], stddev=0.02), name='aux_q_W3')
        self.aux_q_bn_g3 = tf.Variable( tf.truncated_normal([dim_W2], mean=1.0, stddev=0.02), name='aux_q_bn_g3')
        self.aux_q_bn_b3 = tf.Variable( tf.zeros([dim_W2]), name='aux_q_bn_b3')

        self.aux_q_W4 = tf.Variable(tf.truncated_normal([4,4,dim_W2,dim_W1], stddev=0.02), name='aux_q_W4')
        self.aux_q_bn_g4 = tf.Variable( tf.truncated_normal([dim_W1], mean=1.0, stddev=0.02), name='aux_q_bn_g4')
        self.aux_q_bn_b4 = tf.Variable( tf.zeros([dim_W1]), name='aux_q_bn_b4')

        self.aux_q_W_mean = tf.Variable(tf.truncated_normal([6*6*dim_W1,dim_latent_codes], stddev=0.02), name='aux_q_W_mean')
        self.aux_q_W_sigma= tf.Variable(tf.truncated_normal([6*6*dim_W1,dim_latent_codes], stddev=0.02), name='aux_q_W_sigma')
        
    def build_model(self):

        latent_codes = tf.placeholder(tf.float32,[self.batch_size, self.dim_latent_codes])
        component_list=tf.placeholder(tf.int32, [self.batch_size])
  
        image_real = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        image_gen = self.generate(latent_codes)

        p_real, h_real = self.discriminate(image_real)
        p_gen, h_gen = self.discriminate(image_gen)
        aux_latent_code_mean, aux_latent_code_log_sigma=self.latent_encoder(image_gen)

        aux_G_cost=cal_mixture_loss(self.num_mixs, self.dim_latent_codes, aux_latent_code_mean, aux_latent_code_log_sigma, component_list)
        
        discrim_cost=tf.reduce_mean(p_real)-tf.reduce_mean(p_gen)
        
        gen_cost=tf.reduce_mean(p_gen)+0.01*aux_G_cost
        
        

        return latent_codes, component_list, image_real, discrim_cost, gen_cost, aux_G_cost, aux_latent_code_mean, aux_latent_code_log_sigma

    def discriminate(self, image):
        h1 = lrelu( tf.nn.conv2d( image, self.discrim_W1, strides=[1,2,2,1], padding='SAME' ))
        
        h2 = lrelu( batchnormalize( tf.nn.conv2d( h1, self.discrim_W2, strides=[1,2,2,1], padding='SAME'), g=self.discrim_bn_g2, b=self.discrim_bn_b2) )
        
        h3 = lrelu( batchnormalize( tf.nn.conv2d( h2, self.discrim_W3, strides=[1,2,2,1], padding='SAME'), g=self.discrim_bn_g3, b=self.discrim_bn_b3) )
        
        h4 = lrelu( batchnormalize( tf.nn.conv2d( h3, self.discrim_W4, strides=[1,2,2,1], padding='SAME'), g=self.discrim_bn_g4, b=self.discrim_bn_b4) )
        
        h4 = tf.reshape(h4, [self.batch_size, -1])
        
        h5 = tf.matmul( h4, self.discrim_W5 )
       
        y=h5
        return y, h5
    
    def latent_encoder(self, image):
        h1 = tf.nn.relu( batchnormalize(tf.nn.conv2d( image, self.aux_q_W1, strides=[1,2,2,1], padding='SAME' ),g=self.aux_q_bn_g1, b=self.aux_q_bn_b1))
        
        h2 = tf.nn.relu( batchnormalize( tf.nn.conv2d( h1, self.aux_q_W2, strides=[1,2,2,1], padding='SAME'), g=self.aux_q_bn_g2, b=self.aux_q_bn_b2) )
        
        h3 = tf.nn.relu( batchnormalize( tf.nn.conv2d( h2, self.aux_q_W3, strides=[1,2,2,1], padding='SAME'), g=self.aux_q_bn_g3, b=self.aux_q_bn_b3) )
        
        h4 = tf.nn.relu( batchnormalize( tf.nn.conv2d( h3, self.aux_q_W4, strides=[1,2,2,1], padding='SAME'), g=self.aux_q_bn_g4, b=self.aux_q_bn_b4) )
        
        h4 = tf.reshape(h4, [self.batch_size, -1])
        
        mean = tf.matmul( h4, self.aux_q_W_mean )
        sigma=tf.matmul(h4, self.aux_q_W_sigma)
        
        return mean, sigma

    def generate(self, Z):
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z, self.gen_W1), g=self.gen_bn_g1, b=self.gen_bn_b1))
        
        h1 = tf.reshape(h1, [self.batch_size,23,23,self.dim_W1])
        
        output_shape_l2 = [self.batch_size,45,45,self.dim_W2]
        h2 = tf.nn.conv2d_transpose(h1, self.gen_W2, output_shape=output_shape_l2, strides=[1,2,2,1])
        h2 = tf.nn.relu( batchnormalize(h2, g=self.gen_bn_g2, b=self.gen_bn_b2) )
        
        output_shape_l3 = [self.batch_size,45,45,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,1,1,1])
        h3 = tf.nn.relu( batchnormalize(h3, g=self.gen_bn_g3, b=self.gen_bn_b3) )
        
        
        output_shape_l4 = [self.batch_size,45,45,self.dim_W4]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,1,1,1])
        h4 = tf.nn.relu( batchnormalize(h4, g=self.gen_bn_g4, b=self.gen_bn_b4) )
        
        output_shape_l5 = [self.batch_size,90,90,self.dim_W5]
        h5 = tf.nn.conv2d_transpose(h4, self.gen_W5, output_shape=output_shape_l5, strides=[1,2,2,1])
        h5 = tf.nn.relu( batchnormalize(h5, g=self.gen_bn_g5, b=self.gen_bn_b5) )
        
        
        output_shape_l6 = [self.batch_size,90,90,self.dim_W6]
        h6 = tf.nn.conv2d_transpose(h5, self.gen_W6, output_shape=output_shape_l6, strides=[1,1,1,1])

        
        x = tf.nn.sigmoid(h6)
        return x

    def samples_generator(self, batch_size):

        latent_codes = tf.placeholder(tf.float32,[batch_size, self.dim_latent_codes])
        
        h1 = tf.nn.relu(batchnormalize(tf.matmul(latent_codes, self.gen_W1)))
        h1 = tf.reshape(h1, [batch_size,23,23,self.dim_W1])

        output_shape_l2 = [batch_size,45,45,self.dim_W2]
        h2 = tf.nn.conv2d_transpose(h1, self.gen_W2, output_shape=output_shape_l2, strides=[1,2,2,1])
        h2 = tf.nn.relu( batchnormalize(h2) )

        output_shape_l3 = [batch_size,45,45,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,1,1,1])
        h3 = tf.nn.relu( batchnormalize(h3) )
        

        output_shape_l4 = [batch_size,45,45,self.dim_W4]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,1,1,1])
        h4 = tf.nn.relu( batchnormalize(h4) )
        
        output_shape_l5 = [batch_size,90,90,self.dim_W5]
        h5 = tf.nn.conv2d_transpose(h4, self.gen_W5, output_shape=output_shape_l5, strides=[1,2,2,1])
        h5 = tf.nn.relu( batchnormalize(h5) )
        
        output_shape_l6 = [batch_size,90,90,self.dim_W6]
        h6 = tf.nn.conv2d_transpose(h5, self.gen_W6, output_shape=output_shape_l6, strides=[1,1,1,1])

        
        x = tf.nn.sigmoid(h6)
        return latent_codes, x