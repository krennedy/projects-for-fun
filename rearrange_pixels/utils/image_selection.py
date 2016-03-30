""" This module contains utility functions related to initial selection
of suitable images.
"""

import Image

from random import shuffle


def check_sizes_equal(im_path_1, im_path_2):
    """ Open the two images at the inputs paths. Return True if they
    are equal size, and return False otherwise. If there was a problem
    opening the images, return False and report error in console.
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
    """ There is a small bank of images that are the same size (500 X 353)
    Select two randomly and return their paths.
    """
    bank_path = 'image_bank'
    im_list = ['beaux.jpg', 'vangogh.jpg', 'gauguin.jpg', 'seurat.jpg']
    shuffle(im_list)
    two_random_ims = im_list[:2]
    imfile_1, imfile_2 = ['/'.join([bank_path, im]) for im in two_random_ims]
    return imfile_1, imfile_2



