import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


# GLOBALS

# Figure requirements
FIG = plt.figure(figsize=(14,10))
AX1 = FIG.add_subplot(221)
AX2 = FIG.add_subplot(222)
AX3 = FIG.add_subplot(223)
AX4 = FIG.add_subplot(224)

# How many timesteps to use in animation
NSTEPS = 10

class Animator():


    def __init__(self):
        # im1 : the imshow at ax1
        # map1: the rgb pixel map displayed in im1
        # map3: same pixels as map1, but rearranged to look like pix2
        pass

    def load(self, pixA, pixB):
        """ Wow, we need a docstring!
        """
        # Initial displays (top row), just the original images
        rgb1 = sort_rgb_into_image_order(pixA.RGB, pixA.xy)
        map1 = convert_to_imshow_format(rgb1, pixA.shape)

        rgb2 = sort_rgb_into_image_order(pixB.RGB, pixB.xy)
        map2 = convert_to_imshow_format(rgb2, pixB.shape)

        # New displays (bottom row) START WHITE
        map3 = make_white_image(pixB.shape)
        map4 = make_white_image(pixA.shape)

        # Initialize
        kwargs = dict(animated=True, interpolation='none')
        self.im1 = AX1.imshow(map1, **kwargs)
        self.im2 = AX2.imshow(map2, **kwargs)
        self.im3 = AX3.imshow(map3, **kwargs)
        self.im4 = AX4.imshow(map4, **kwargs)  # Extra attributes we'll need later, urgh

        self.pixA = pixA   # Blaaaaahhhhh
        self.pixB = pixB   # Blaaahhhhhhh

        self.npix = map1.shape[0] * map1.shape[1]
        self.nstep = int(self.npix / float(NSTEPS))  # number PER step

        self.map1 = map1
        self.map2 = map2
        self.map3 = map3
        self.map4 = map4

        # The original maps will get modified as you go, so keep a reference
        self.ref1 = map1.copy()
        self.ref2 = map2.copy()

    def draw(self,):
        ani = animation.FuncAnimation(
            FIG, self.updatefig,
            np.arange(0, self.npix, self.nstep),
            interval=100, blit=False, repeat=False)
        plt.show()

    def updatefig(self, j):
        """ Update all 4. Think rgb_A is now called map1.
        """
        self.take_out(self.map1, self.pixA, j)
        self.take_out(self.map2, self.pixB, j)
        self.put_in(self.ref1, self.map3, self.pixA, j)
        self.put_in(self.ref2, self.map4, self.pixB, j)
        self.im1.set_array(self.map1)
        self.im2.set_array(self.map2)
        self.im3.set_array(self.map3)
        self.im4.set_array(self.map4)

        # return top row to original state if at end of animation
        if j+self.nstep >= self.npix:
            self.im1.set_array(self.ref1)
            self.im2.set_array(self.ref2)


    def take_out(self, img, pix, j):
        initial_shape = img.shape
        img_flat = img.reshape(self.npix, 3)
        img_flat[j: j + self.nstep] = np.array([155, 155, 155]).astype('uint8')
        img = img_flat.reshape(initial_shape)
        return img


    def put_in(self, img_pull, img_put, pix, j):
        """ Here, you want to put in a few pixels of new image
        But using the values input from another image
        """
        initial_shape = img_put.shape

        img_put_flat = img_put.reshape(self.npix, 3)
        img_pull_flat = img_pull.reshape(self.npix, 3)

        # Have to figure out based on x,y order of original, which pixels to put in new first
        x = pix.xy[:, 0]
        y = pix.xy[:, 1]
        idx_sort = np.lexsort((x, y))

        x_new_sorted = pix.xy_new[:, 0][idx_sort]
        y_new_sorted = pix.xy_new[:, 1][idx_sort]

        x_dim = img_put.shape[1]

        positions = y_new_sorted * x_dim + x_new_sorted
        positions_this_time = positions[j: j+self.nstep]

        img_put_flat[positions_this_time] = img_pull_flat[j: j+self.nstep]
        img_put = img_put_flat.reshape(initial_shape)
        return img_put


def sort_rgb_into_image_order(rgb, xy):
    """ We are given an rgb array of pixels, and the xy-coord they are to be at.
    From this, reorder rgb so it can display new image
    """
    x = xy[:, 0]
    y = xy[:, 1]
    idx_sort = np.lexsort((x, y))
    rgb_sorted = rgb[idx_sort]
    return rgb_sorted


def convert_to_imshow_format(rgb, image_shape):
    """ PM what it says. Takes almost no time!
    rgb is pixel LIST in format XXX.
    Reshape to XXX and convert to unsigned 8-bit integers.
    """
    xdim, ydim = image_shape
    rgb_display = rgb.reshape((xdim, ydim, 3))
    rgb_display = rgb_display.astype(np.uint8)
    return rgb_display


def make_white_image(inshape):
    """ Returns an all-white image of the input size
    """
    outshape = inshape + (3,)
    white_map = np.ones(outshape).astype('uint8') * 255
    return white_map


