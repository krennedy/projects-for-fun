#!/usr/bin/env python
"""
An animated image
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Image


fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

#####################################

img_obj_1 = Image.open('figs/vermeer.jpg')
img_1 = np.asarray(img_obj_1)
img_1.flags.writeable = True
img_1_keeper = img_1.copy()

img_obj_2 = Image.open('figs/dali.jpg')
img_2 = np.asarray(img_obj_2)
img_2.flags.writeable = True

img_3 = img_1.copy()
img_3[:,:] = (255, 0, 0)


def take_out(img, i):
    img[i] = np.array([255,0,0]).astype('uint8')
    return img 

def put_in(img_put, img_pull, i):
    """ Here, you want to put in a few pixels of new image
    But using the values input from another image
    """
    img_put[i] = img_pull[i+1]
    return img_put


i=0
im1 = ax1.imshow(take_out(img_1, i), animated=True, interpolation='none')
im2 = ax2.imshow(take_out(img_2, i), animated=True, interpolation='none')
im3 = ax3.imshow(img_3, animated=True, interpolation='none')


def updatefig(j):
    im1.set_array(take_out(img_1, j))
    im2.set_array(take_out(img_2, j))
    im3.set_array(put_in(img_3, img_1_keeper, j))
    return im1, im2, im3

ani = animation.FuncAnimation(fig, updatefig, np.arange(100), interval=1,
                              blit=False)
plt.show()







def updatefig2(*args):
    global x, y
    x += np.pi / 15.
    y += np.pi / 20.
    im.set_array(f(x, y))
    return im,
#x = np.linspace(0, 2 * np.pi, 12)
#y = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
#im = plt.imshow(f(x, y), animated=True, interpolation='none')
