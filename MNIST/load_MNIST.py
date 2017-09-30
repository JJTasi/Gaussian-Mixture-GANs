from tensorflow.examples.tutorials.mnist import input_data
import os, random
import pandas as pd
import numpy as np



def mnist_with_valid_set(path):
    mnist = input_data.read_data_sets(path, one_hot=False)
    trX=np.reshape(mnist.train.images,(mnist.train.images.shape[0],28,28,1))
    trY=mnist.train.labels
    vaX=np.reshape(mnist.validation.images,(mnist.validation.images.shape[0],28,28,1))
    vaY=mnist.validation.labels
    teX=np.reshape(mnist.test.images,(mnist.test.images.shape[0],28,28,1))
    teY=mnist.test.labels
    
    
    trX=trX.astype(np.float32)/255.
    vaX=vaX.astype(np.float32)/255.
    teX=teX.astype(np.float32)/255.
    trY=trY.astype(np.float32)
    vaY=vaY.astype(np.float32)
    teY=teY.astype(np.float32)
    
    
    index=range(trX.shape[0])
    random.shuffle(index)
    trX=trX[index,:,:,:]
    trX=np.concatenate([trX,trX,trX],3)
    trY=trY[index]
    
    return trX, vaX, teX, trY, vaY, teY