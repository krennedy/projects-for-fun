""" Some comment
"""

import Image

import numpy as np
from random import randint


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
	sets = [['figs/dali.jpg', 'figs/vermeer.jpg'],
			['figs/beaux.jpg', 'figs/vangogh.jpg']]
	nsets = len(sets)
	rand_idx = randint(0, nsets-1)
	imfile_1, imfile_2 = sets[rand_idx]
	print "I will be selecting your images now, muahaha!"
	print "Images selected: %s and %s"%(imfile_1, imfile_2)
	return imfile_1, imfile_2


def convert_to_imshow_format(df, xcol_name, ycol_name):
    """ PM what it says. Takes almost no time!
    """
    xmin = df[xcol_name].min()
    xmax = df[xcol_name].max()
    ymin = df[ycol_name].min()
    ymax = df[ycol_name].max()
    x_dim = xmax - xmin + 1
    y_dim = ymax - ymin + 1
    df.sort(columns=[ycol_name, xcol_name], inplace=True)
    rgb = df[['R','G','B']].values
    rgb = rgb.reshape((x_dim, y_dim, 3))
    rgb = rgb.astype(np.uint8)
    return rgb, df
