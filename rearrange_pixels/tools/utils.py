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
		sizes_equal = im_1.size != im_2.size
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

def find_theta_colorwheel(R, G, B):
	""" Obvious shortcoming - orange reds will be very close to red,
    but purple reds will be very far.
    Maybe can solve this by resetting scale near least dense portion
    of colormap?
    THIS IS THE COMPUTATIONAL BOTTLENECK OF THE DAMN CODE
    """
	# Find which of R/G/B had the least contribution to overall color
	rgb = np.column_stack((R, G, B))
	which_least = np.argmin(rgb, axis=1)

	# Based least-dominant color, figure out which 1/3rd of colorwheel to start on
	theta_offset = which_least * (2 * np.pi / 3.)

	# Find the color differences
	rg = R - G
	gb = G - B
	br = B - R

	mask_least_R = which_least == 0
	mask_least_G = which_least == 1
	mask_least_B = which_least == 2

	color_diff = np.empty_like(R)
	color_diff[mask_least_R] = gb[mask_least_R]
	color_diff[mask_least_G] = br[mask_least_G]
	color_diff[mask_least_B] = rg[mask_least_B]

	percent_color = color_diff / 255. # values from 0 to 1
	delta_theta = np.arccos(percent_color)

	theta_cwheel = delta_theta + theta_offset
	return theta_cwheel


