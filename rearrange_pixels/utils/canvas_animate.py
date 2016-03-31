""" This contents of this module contain the instructions on how to
display an animation of the pixel rearrangement.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


# Global variables
# -------------------------------------------------
# Figure object
FIG = plt.figure(figsize=(14,10))

# FIXME: this should be kwarg, not hardcode
SAVE_SNAPS = True

# The number of time-steps to use in the animation
NSTEPS = 25
# -------------------------------------------------


class Animator():
    """
    Attributes
    ---------
    pixA: PixelTracker object containing data about one image
    pixB: PixelTracker object containing data about another image

    map1: original image from pixA (gets depleted in animation)
    map2: original image from pixB (gets depleted in animation)
    map3: rearranged image of pixels from pixA (starts blank)
    map4: rearranged image of pixels from pixB (starts blank)
    ref1: copy of map1 (b/c map1 will get modified in animation)
    ref2: copy of map2 (b/c map1 will get modified in animation)

    im[1-4]: The imshow() objects for each subplot (get modified)

    npix: Number of pixels in the images
    nstep: How many pixels will be 'rearranged' in each step of the animation
    """

    def load(self, pixA, pixB):
        """
        Load in pixA and pixB (PixelTracker objects with information
        about the pixels of two images).

        This method uses this information to create 4 'maps' which are
        in the correct format to be displayed by imshow().

        Also calculate the total number of pixels, and number to be
        updated per time-step.
        """
        self.pixA = pixA
        self.pixB = pixB

        # Initial displays (top row), just the original images.
        rgb1 = sort_rgb_into_image_order(pixA.RGB, pixA.xy)
        self.map1 = convert_to_imshow_format(rgb1, pixA.shape)

        rgb2 = sort_rgb_into_image_order(pixB.RGB, pixB.xy)
        self.map2 = convert_to_imshow_format(rgb2, pixB.shape)

        # New displays (bottom row) - these start white and get filled in.
        self.map3 = make_white_image(pixB.shape)
        self.map4 = make_white_image(pixA.shape)

        # The original maps will get depleted as you go, so keep a reference.
        self.ref1 = self.map1.copy()
        self.ref2 = self.map2.copy()

        # The total number of pixels, and number to change each time-step.
        self.npix = self.map1.shape[0] * self.map1.shape[1]
        self.nstep = int(self.npix / float(NSTEPS))  # number PER step

    def initialize_canvas(self):
        """ Create 4 subplots and display map[1-4] in them.
        """
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

    def draw(self, skip_animation=False):
        """ Do the animation (alternatively can just skip to the end
        result of rearrangement, if skip_animation is set to True.)
        """
        self.initialize_canvas()

        if not skip_animation: # FIXME: handle else True
            draw_me = animation.FuncAnimation(
                FIG, self.updatefig,
                np.arange(0, NSTEPS),
                interval=1, blit=False, repeat=False)
        plt.show()

    def updatefig(self, j):
        """ Each timestep, we will be removing a subset of pixels
        from the top row images, and inserting them into the bottom
        row images.
        Save the outputs if desired. I use to make .gif summary.
        """
        self.take_out(self.map1, j)
        self.take_out(self.map2, j)
        self.put_in(self.ref1, self.map3, self.pixA, j)
        self.put_in(self.ref2, self.map4, self.pixB, j)

        self.im1.set_array(self.map1)
        self.im2.set_array(self.map2)
        self.im3.set_array(self.map3)
        self.im4.set_array(self.map4)

        # When we reach end of animation, revert top row to original images.
        if j >= NSTEPS - 1:
            self.im1.set_array(self.ref1)
            self.im2.set_array(self.ref2)

        if SAVE_SNAPS == True:
            plt.savefig('saved_snaps/ex_%s.png'%j, dpi=40)


    def take_out(self, img, j):
        """ Take some (j*nstep) pixels out of img, and replace them
        with grey (R = G = B = 155).
        """
        initial_shape = img.shape
        img_flat = img.reshape(self.npix, 3)
        start_idx = j * self.nstep
        stop_idx = start_idx + self.nstep
        img_flat[start_idx:stop_idx] = np.array([155, 155, 155]).astype('uint8')
        img = img_flat.reshape(initial_shape)
        return img


    def put_in(self, img_pull, img_put, pix, j):
        """ Take some (j*nstep) pixels from the reference (img_pull),
        and put them into the target (img_put), in the correct
        positions for their specified rearrangement.
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
        start_idx = j*self.nstep
        stop_idx = start_idx + self.nstep
        positions_this_time = positions[start_idx: stop_idx]

        img_put_flat[positions_this_time] = img_pull_flat[start_idx: stop_idx]
        img_put = img_put_flat.reshape(initial_shape)
        return img_put


def sort_rgb_into_image_order(rgb, xy):
    """ Given an rgb array of pixels, and the xy-coordinates they are
    to be at. From this, reorder rgb as they are supposed to appear.
    """
    x = xy[:, 0]
    y = xy[:, 1]
    idx_sort = np.lexsort((x, y))
    rgb_sorted = rgb[idx_sort]
    return rgb_sorted


def convert_to_imshow_format(rgb, image_shape):
    """ Take a list of rgb values which are already in order for
    display, and reshape them into the specified image_shape.
    """
    xdim, ydim = image_shape
    rgb_display = rgb.reshape((xdim, ydim, 3))
    rgb_display = rgb_display.astype(np.uint8)
    return rgb_display


def make_white_image(inshape):
    """ Returns an all-white image (R = G = B = 255) of the input size.
    """
    outshape = inshape + (3,)
    white_map = np.ones(outshape).astype('uint8') * 255
    return white_map


