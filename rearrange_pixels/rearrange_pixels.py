"""
This is the controller script, which launches the animation.
Everything else is a tool.
"""

from tools.utils import check_sizes_equal, select_random_image_pair

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


def remap_pixels(target, reference):
	""" pix is an object containing information about all the pixels
	in their target image - where they were in the target, and where they
	need to go in the reconstruction of the reference.
	"""
	pix = ()
	return pix


def launch_animation(pix1, pix2):
	""" Where the Canvassing animation will actually be launched """
	(pix1, pix2)


if __name__ == '__main__':
	""" Three parts: image selection. calculation of pixels. display.
	"""
	imfile_1, imfile_2 = select_input_images()
	pix1 = remap_pixels(imfile_1, imfile_2)
	pix2 = remap_pixels(imfile_1, imfile_2)
	launch_animation(pix1, pix2)
