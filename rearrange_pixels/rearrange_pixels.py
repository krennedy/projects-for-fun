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

def launch_animation(imfile_1, imfile_2):
	pass


if __name__ == '__main__':
	imfile_1, imfile_2 = select_input_images()
	launch_animation(imfile_1, imfile_2)
