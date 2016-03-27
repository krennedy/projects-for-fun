from tools.utils import find_theta_colorwheel

import Image
import numpy as np


class PixelTracker():
    """
    Some notes about this
    Can I make the returned instance just BE a dataframe
    For which I basically append on function calls not native
    to Pandas?
    """

    def __init__(self, path_to_jpg):
        
        """ Testing to find the format I want for eventual 
        analysis image_dict should eventually be an image
        But here as a dict, assuming initial processing done
        """
        img = Image.open(path_to_jpg)
        self.shape = img.size

        img_pix = np.asarray(img)

        R = img_pix[:,:,0].ravel().astype(int)
        G = img_pix[:,:,1].ravel().astype(int)
        B = img_pix[:,:,2].ravel().astype(int)

        x_dim = img_pix.shape[0]
        y_dim = img_pix.shape[1]

        x_axis = np.arange(x_dim)
        y_axis = np.arange(y_dim)
        x, y = np.meshgrid(x_axis, y_axis)

        x = x.ravel()
        y = y.ravel()

        self.xy = np.column_stack((x, y))
        self.RGB = np.column_stack((R, G, B))
        self.dist_to_black = R + G + B
        self.theta_cwheel = find_theta_colorwheel(R, G, B)


    def sort_by_fancybins(self,):
        """ Bin by darkness into N bins
        Then sort by theta within those bins
        FIXME: how much of this hard sorting is actually necessary?
        """
        npix = len(self.dist_to_black)
        arr = np.arange(npix)
        nbins = 20
        n_per_chunk = npix/nbins + 1  # +1 fudge factor for rounding
        dist_to_black_broad = arr/n_per_chunk


        # Then sort by theta_cwheel within black
        idx_sort = np.lexsort((dist_to_black_broad, self.theta_cwheel))
        self.sort_index = idx_sort
        #self.dist_to_black = self.dist_to_black[idx_sort]
        #self.theta_cwheel = self.theta_cwheel[idx_sort]
        #self.RGB = self.RGB[idx_sort]
        #self.xy = self.xy[idx_sort]

                
    def rearrange_pixels(self, target):
        """
        Here, target is the target image. 
        We map our current pixels (reference image) onto the
        target coordinates.
        This assumes that self and target are both already
        ordered correctly.
        """
        self.xy_new = target.xy
