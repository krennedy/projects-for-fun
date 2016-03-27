"""
This is the controller script, which launches the animation.
Everything else is a tool.
"""

from tools.utils import check_sizes_equal, select_random_image_pair
from pixel_tracker import PixelTracker
from canvas_animate import Animator

import sys

def select_input_images():
	""" 2 cases, either
	"""
	cl_args = sys.argv

	# Check first if the user supplied at least 2 images
	if len(cl_args) >= 3:
		imfile_1, imfile_2 = cl_args[1:3]

		# Use user-supplied images only if they are same size
		sizes_equal = check_sizes_equal(imfile_1, imfile_2)
		if not sizes_equal:
			print "Sorry user! Your images were not of equal size."
			imfile_1, imfile_2 = select_random_image_pair()

	# If 2 images not supplied at command line, select our own
	else:
		print "No images supplied"
		imfile_1, imfile_2 = select_random_image_pair()

	return imfile_1, imfile_2

def exchange_pixels(imfile_1, imfile_2):
	""" pix is an object containing information about all the pixels
	in their target image - where they were in the target, and where they
	need to go in the reconstruction of the reference.
	"""
	pix1 = PixelTracker(imfile_1)
	pix1.sort_by_fancybins()
	pix2 = PixelTracker(imfile_2)
	pix2.sort_by_fancybins()

	pix1.rearrange_pixels(pix2)
	pix2.rearrange_pixels(pix1)
	return pix1, pix2


def launch_animation(pix1, pix2):
	""" Where the Canvassing animation will actually be launched """
	canvas = Animator(pix1, pix2)
	canvas.draw()

if __name__ == '__main__':
	""" Three parts: image selection. calculation of pixels. display.
	"""
	imfile_1, imfile_2 = select_input_images()
	pix1, pix2 = exchange_pixels(imfile_1, imfile_2)
	launch_animation(pix1, pix2)
