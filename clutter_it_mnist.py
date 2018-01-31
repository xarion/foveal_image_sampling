### Clutters a given (28x28) image (mnist/fmnist) with 8x8 mnist digit (section control flag_c) parts, with either zoom or no zoom (flag_d)
### Sample use:
    ### from tensorflow.examples.tutorials.mnist import input_data
    ### mnist = input_data.read_data_sets('fMNIST_data', one_hot=True)
    ### import clutter_it_mnist
    ### clutter_it_mnist.clutter_it(mnist.train.images[0:1,:],1)

###--------------------------------------------------------------

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import numpy as np
from scipy.ndimage.interpolation import zoom

# clutter the mnist image by blowing it up to 100x100 and adding upto c_max object parts (8x8), train(1)/val(2)/test(3) flag_c, and target no-zoom(1)/zoom(2) flag_d
def clutter_it(imgd,flag_c,c_max=20,flag_d=1):
    imgd_h = np.zeros([np.shape(imgd)[0],100*100])
    for j in range(np.shape(imgd)[0]):
        img1 = np.zeros([100,100])
        if flag_d == 2:
            dum0 = np.random.random(1)[0]*2.67+0.33 # zoom between 0.33 and 3.0
            #print dum0
            dum0_0 = zoom(np.reshape(imgd[j,:],[28,28]),dum0)
            dum0_1 = np.shape(dum0_0)[0]/2.
            img1[50-np.int(np.ceil(dum0_1)):50+np.int(np.floor(dum0_1)),50-np.int(np.ceil(dum0_1)):50+np.int(np.floor(dum0_1))] = dum0_0 # place image centrally
        if flag_d == 1:
            img1[36:64,36:64] = np.reshape(imgd[j,:],[28,28]) # place image centrally
        dum1 = np.random.randint(c_max+1)
        #print dum1
        if (dum1 > 0):
            for i in range(dum1): # 8x8 patches being added
                if (flag_c == 1):
                    dum2 = np.random.randint(np.shape(mnist.train.images)[0])
                    dum_img = np.reshape(mnist.train.images[dum2,:],[28,28])
                elif (flag_c == 2):
                    dum2 = np.random.randint(np.shape(mnist.validation.images)[0])
                    dum_img = np.reshape(mnist.validation.images[dum2,:],[28,28])
                elif (flag_c == 3):
                    dum2 = np.random.randint(np.shape(mnist.test.images)[0])
                    dum_img = np.reshape(mnist.test.images[dum2,:],[28,28])
                dum_x = np.int((28-8)*np.random.random(1)[0])
                dum_y = np.int((28-8)*np.random.random(1)[0])
                x = np.int((100-8)*np.random.random(1)[0])
                y = np.int((100-8)*np.random.random(1)[0])
                dum_img2 = dum_img[dum_x:dum_x+8,dum_y:dum_y+8]
                img1[x+np.where(dum_img2)[0],y+np.where(dum_img2)[1]] = dum_img2[np.where(dum_img2)[0],np.where(dum_img2)[1]]
        imgd_h[j,:] = np.reshape(img1,[1,100*100])
    return imgd_h

