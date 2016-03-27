import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


# GLOBALS
FIG = plt.figure(figsize=(14,10)) # Define out fig
SAVE_SNAPS = True
NSTEPS = 10 # How many timesteps to use in animation

class Animator():


    def __init__(self):
        # im1 : the imshow at ax1
        # map1: the rgb pixel map displayed in im1
        # map3: same pixels as map1, but rearranged to look like pix2
        pass

    def load(self, pixA, pixB):
        """ Wow, we need a docstring!
        """
        self.pixA = pixA  # Blaaaaahhhhh
        self.pixB = pixB  # Blaaahhhhhhh

        # Initial displays (top row), just the original images
        rgb1 = sort_rgb_into_image_order(pixA.RGB, pixA.xy)
        self.map1 = convert_to_imshow_format(rgb1, pixA.shape)

        rgb2 = sort_rgb_into_image_order(pixB.RGB, pixB.xy)
        self.map2 = convert_to_imshow_format(rgb2, pixB.shape)

        # New displays (bottom row) START WHITE
        self.map3 = make_white_image(pixB.shape)
        self.map4 = make_white_image(pixA.shape)

        self.npix = self.map1.shape[0] * self.map1.shape[1]
        self.nstep = int(self.npix / float(NSTEPS))  # number PER step

        # The original maps will get modified as you go, so keep a reference
        self.ref1 = self.map1.copy()
        self.ref2 = self.map2.copy()

    def initialize_canvas(self):
        kwargs = dict(animated=True, interpolation='none')

        ax1 = FIG.add_subplot(221)
        self.im1 = ax1.imshow(self.ref1, **kwargs)
        plt.title('Image 1')
        plt.axis('off')

        ax2 = FIG.add_subplot(222)
        self.im2 = ax2.imshow(self.ref2, **kwargs)
        plt.title('Image 2')
        plt.axis('off')

        ax3 = FIG.add_subplot(223)
        self.im3 = ax3.imshow(self.map3, **kwargs)
        plt.title('Image 1 Rearranged')
        plt.axis('off')

        ax4 = FIG.add_subplot(224)
        self.im4 = ax4.imshow(self.map4, **kwargs)
        plt.title('Image 2 Rearranged')
        plt.axis('off')

    def draw(self,):
        self.initialize_canvas()
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

        if SAVE_SNAPS == True:
            plt.savefig('saved_snaps/ex_%s.png'%j)


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


