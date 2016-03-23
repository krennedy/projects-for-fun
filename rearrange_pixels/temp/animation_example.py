# http://matplotlib.org/examples/animation/simple_anim.html

"""
A simple example of an animated plot
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Image

img_obj = Image.open('figs/vermeer.jpg')
img = np.asarray(img_obj)

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
#line, = ax.plot(x, np.sin(x))
imline = ax.imshow(img)

def animate2(i):
    line.set_ydata(np.sin(x + i/10.0))  # update the data
    return line
# Init only required for blitting to give a clean slate.
def init2():
    line.set_ydata(np.ma.array(x, mask=True))
    return line


def animate(i):
    if i<2:
        img[0,0] = np.array([0,0,0])
    else:
        img[0,0] = np.array([255,255,255])
    imline = ax.imshow(img, interpolation='none')
    return imline

def init():
    return imline

ani = animation.FuncAnimation(fig, animate, np.arange(1, 4), init_func=init,
                              interval=750, blit=False)
plt.show()
