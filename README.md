#######################################################################
#####  The Problem                                                ##### 
#######################################################################

The inspiration for this project comes from a challenge on the "Code 
Golf" section of StackOverflow, originally posted by user Helke Homba

(http://codegolf.stackexchange.com/questions/33172/american-gothic-in-the-palette-of-mona-lisa-rearrange-the-pixels)

The crux of the problem is to take two images of equal pixel-size, and
to recreate the second image as closely as possible, by best rearrang-
ing the pixels of the first image.

My ambition was to do this in a speedy and user-friendly way - not
worrying about character count. The rearrangement is still coarse -
I'd love even further refinement sometime :)

#######################################################################
#####  Running the Rearrangement                                  ##### 
#######################################################################

Within rearrange_pixels/:
> python launch_switch.py [path_to_first_iamge] [path_to_second_image]

The paths to the two images to be palette-swapped are optional. If none
are provided, then two will be randomly selected for you.


#######################################################################
#####  Examples                                                   ##### 
#######################################################################

A couple of examples of the rearrangement are shown in examples/. This
includes a gif of the rearrangement, and a 'still' of another swap.


