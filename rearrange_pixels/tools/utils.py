""" Some comment
"""

import Image
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