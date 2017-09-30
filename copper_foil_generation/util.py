import scipy.misc
import numpy as np
from math import cos, sin

def OneHot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh

def save_visualization(X, (nh, nw), save_path='./vis/sample.png'):
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh, w * nw, 3))

    for n,x in enumerate(X):
        j = n / nw
        i = n % nw
        img[j*h:j*h+h, i*w:i*w+w, :] = x

    scipy.misc.imsave(save_path, img)
    
def mixtures_noise(mean,num_mixture,num_sample, dim):
    vec=np.random.normal(0,0.01,size=(num_sample,dim))
    component_list=np.random.randint(num_mixture,size=[num_sample])
    vec=vec+mean[component_list]
    return vec

def circle_sampler(batchsize, z_dim, batch_indices, n_class):
    if z_dim % 2 != 0:
        raise Exception("z_dim must be a multiple of 2.")

    def sample(x, y, label, n_class):
        shift = 1
        r = 2.0 * np.pi / float(n_class) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        x = np.array([new_x, new_y])
        return x.reshape((2,))

    x_var = 0.5
    y_var = 0.5
    x = np.random.normal(0, x_var, (batchsize, z_dim / 2))
    y = np.random.normal(0, y_var, (batchsize, z_dim / 2))
    z = np.empty((batchsize, z_dim), dtype=np.float32)
    
    for batch in xrange(batchsize):
        for zi in xrange(z_dim / 2):
            z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], batch_indices[batch], n_class)
    return z