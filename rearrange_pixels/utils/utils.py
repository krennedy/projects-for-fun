""" Some comment
"""

import Image

import numpy as np
from random import shuffle, randint


def check_sizes_equal(im_path_1, im_path_2):
    """ Check sizes equal. Return True if so, False if not.
    If problem opening the images, return False and say so
    """
    try:
        im_1 = Image.open(im_path_1)
        im_2 = Image.open(im_path_2)

        area_1 = im_1.size[0] * im_1.size[1]
        area_2 = im_2.size[0] * im_2.size[1]
        sizes_equal = area_1 == area_2

        return sizes_equal
    except IOError:
        print 'Problems opening file - were these even images?'
        return False


def select_random_image_pair():
    """ These are all just hardcoded from a known set.
    Pick a random set.
    """
    bank_path = 'image_bank'

    # all the images below are 500 X 353 pixels
    im_list = ['beaux.jpg', 'vangogh.jpg', 'gauguin.jpg', 'seurat.jpg']
    shuffle(im_list)
    two_random_ims = im_list[:2]
    imfile_1, imfile_2 = ['/'.join([bank_path, im]) for im in two_random_ims]
    return imfile_1, imfile_2


def find_theta_colorwheel(rgb):
    """ Obvious shortcoming - orange reds will be very close to red,
    but purple reds will be very far.
    Maybe can solve this by resetting scale near least dense portion
    of colormap?
    THIS IS THE COMPUTATIONAL BOTTLENECK OF THE DAMN CODE
    """
    # Find which of R/G/B had the least contribution to overall color
    #rgb = np.column_stack((R, G, B))
    which_least = np.argmin(rgb, axis=1)

    # Based least-dominant color, figure out which 1/3rd of colorwheel to start on
    theta_offset = which_least * (2 * np.pi / 3.)

    # Find the color differences
    rg = rgb[:, 0] - rgb[:, 1]
    gb = rgb[:, 1] - rgb[:, 2]
    br = rgb[:, 2] - rgb[:, 0]

    mask_least_R = which_least == 0
    mask_least_G = which_least == 1
    mask_least_B = which_least == 2

    color_diff = np.empty_like(which_least)
    color_diff[mask_least_R] = gb[mask_least_R]
    color_diff[mask_least_G] = br[mask_least_G]
    color_diff[mask_least_B] = rg[mask_least_B]

    percent_color = color_diff / (255.) # values sposed to be -1 to 1

    delta_theta = percent_color * (np.pi / 3.0)  # div by 3 since constrained to 1/3rd of wheel
    theta_cwheel = theta_offset - delta_theta

    return theta_cwheel


