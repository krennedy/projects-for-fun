import Image
import numpy as np


class PixelTracker():
    """ This class parses the information about an input image,
    including the spatial information about each RGB pixel. #FIXME: elaborate better!

    Inputs
    ---------
    path_to_image: The path to an input image, of type .jpg, .png, etc.

    Attributes
    ---------
    shape: tuple containing dimensions of the input image
    xy: 2d-array describing the coordinates of each pixel in the
        original image. The first and second columns contain the x and
        y coordinates, respectively.
    RGB: 2d-array describing the color of each pixel.
        The three columns contain the R, G and B values ranging 0-255.
    xy_new: 2d-array describing the coordinates of each pixel in the
        *reconstructed* image, e.g. indicating where each RGB value
        will move to in the new image display.

    Methods
    ---------
    sort_by_fancybins(): reorder the pixels weighting both darkness and
        color. The pixels are coarse-grained by darkness first, and
        within this coarse-graining, ordered by 'colorwheel' color.
    rearrange_pixels(): Takes input which is also an instance of a
        PixelTracker object, and which has also had sort_by_fancybins()
        applied. Maps new coordinates xy_new onto the original image.
    """

    def __init__(self, path_to_image):
        
        """ Blah.
        """
        img = Image.open(path_to_image)
        self.shape = img.size[::-1]  # reverse order

        img_pix = np.asarray(img)

        R = img_pix[:,:,0].ravel().astype(int)
        G = img_pix[:,:,1].ravel().astype(int)
        B = img_pix[:,:,2].ravel().astype(int)

        x_dim = img_pix.shape[1]
        y_dim = img_pix.shape[0]

        x_axis = np.arange(x_dim)
        y_axis = np.arange(y_dim)
        x, y = np.meshgrid(x_axis, y_axis)

        x = x.ravel()
        y = y.ravel()
        self.xy = np.column_stack((x, y))
        self.RGB = np.column_stack((R, G, B))


    def sort_by_fancybins(self,):
        """ Bin by darkness into N bins
        Then sort by theta within those bins
        FIXME: how much of this hard sorting is actually necessary?
        """
        theta_cwheel = find_theta_colorwheel(self.RGB)
        dist_to_black = np.sum(self.RGB, axis=1)

        # Coarse grain distance to black
        nbins = 20
        darkest = min(dist_to_black)
        lightest = max(dist_to_black)
        binsize = (lightest - darkest) / float(nbins)
        dist_to_black_broad = np.round(dist_to_black/ binsize)

        # Then sort by theta_cwheel within black
        idx_sort = np.lexsort((theta_cwheel, dist_to_black_broad))

        self.xy = self.xy[idx_sort]
        self.RGB = self.RGB[idx_sort]

    def rearrange_pixels(self, target):
        """
        Here, target is the target image. 
        We map our current pixels (reference image) onto the
        target coordinates.
        This assumes that self and target are both already
        ordered correctly.
        """
        self.xy_new = target.xy


def find_theta_colorwheel(rgb):
    """ Obvious shortcoming - orange reds will be very close to red,
    but purple reds will be very far.
    Maybe can solve this by resetting scale near least dense portion
    of colormap?
    THIS IS THE COMPUTATIONAL BOTTLENECK OF THE DAMN CODE
    """
    # Find which of R/G/B had the least contribution to overall color
    which_least = np.argmin(rgb, axis=1)

    # Based least-dominant color, figure out which 1/3rd of colorwheel to start on
    theta_offset = which_least * (2 * np.pi / 3.)

    # Find the color differences
    rg = rgb[:, 0] - rgb[:, 1]
    gb = rgb[:, 1] - rgb[:, 2]
    br = rgb[:, 2] - rgb[:, 0]

    mask_least_r = which_least == 0
    mask_least_g = which_least == 1
    mask_least_b = which_least == 2

    color_diff = np.empty_like(which_least)
    color_diff[mask_least_r] = gb[mask_least_r]
    color_diff[mask_least_g] = br[mask_least_g]
    color_diff[mask_least_b] = rg[mask_least_b]

    percent_color = color_diff / (255.) # values sposed to be -1 to 1

    delta_theta = percent_color * (np.pi / 3.0)  # div by 3 since constrained to 1/3rd of wheel
    theta_cwheel = theta_offset - delta_theta

    return theta_cwheel

