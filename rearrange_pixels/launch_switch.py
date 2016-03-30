"""
This is the controller script, which the user runs from the command
line.

The functionality of this module is to:
    1. Select of two images (user-supplied or selected at random)
    2. Calculate how to rearrange pixels to look like the other image
    3. Display an animation of the rearrangement.

To run:
> python launch_switch.py [path_to_image_1] [path_to_image_2]

If two valid paths aren't passed, then two images are randomly selected.
"""


import sys

from utils.image_selection import check_sizes_equal, select_random_image_pair
from utils.pixel_tracker import PixelTracker
from utils.canvas_animate import Animator


def select_input_images():
    """ Check whether two valid image files were supplied at the
    command line.
    If two valid image paths were supplied, and the images were of
    identical pixel size, then return these paths. Otherwise, return
    two random same-size image paths from the included image bank.
    """
    cl_args = sys.argv

    # Two images in argv. If not same size, then select random images.
    if len(cl_args) >= 3:
        imfile_1, imfile_2 = cl_args[1:3]

        sizes_equal = check_sizes_equal(imfile_1, imfile_2)
        if not sizes_equal:
            print "Sorry user! Your images were not of equal size."
            imfile_1, imfile_2 = select_random_image_pair()

    # Fewer than two images in argv, so select random images.
    else:
        print "No images supplied"
        imfile_1, imfile_2 = select_random_image_pair()

    return imfile_1, imfile_2

def exchange_pixels(imfile_1, imfile_2):
    """ This function returns a PixelTracker object corresponding to
    each input image. This object contains information about all the
    individual pixels of the input image - including their
    coordinates in the original image, what coordinates they get re-
    mapped to in order to approximate the other image.
    """
    pix1 = PixelTracker(imfile_1)
    pix2 = PixelTracker(imfile_2)

    # Sort the pixels by darkness and color
    pix1.sort_by_fancybins()
    pix2.sort_by_fancybins()

    # Add information about how to rearrange pixels to look like each other
    pix1.rearrange_pixels(pix2)
    pix2.rearrange_pixels(pix1)
    return pix1, pix2


def launch_animation(pix1, pix2):
    """ Given two PixelTracker instances pix1 and pix2, display the
    original images, and then animate how pixels rearrange to become
    the other image.
    """
    canvas = Animator()
    canvas.load(pix1, pix2)
    canvas.draw()


if __name__ == '__main__':
    imfile_1, imfile_2 = select_input_images()
    pix1, pix2 = exchange_pixels(imfile_1, imfile_2)
    launch_animation(pix1, pix2)
