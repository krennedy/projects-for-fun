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

        x_dim = img_pix.shape[1]
        y_dim = img_pix.shape[0]

        x_axis = np.arange(x_dim)
        y_axis = np.arange(y_dim)
        x, y = np.meshgrid(x_axis, y_axis)

        self.x = x.ravel()
        self.y = y.ravel()
        self.RGB = np.column_stack((R, G, B))


    def sort_by_fancybins(self,):
        """ Bin by darkness into N bins
        Then sort by theta within those bins
        FIXME: how much of this hard sorting is actually necessary?
        """
        # FIXME: the below are not part of init, right? sort_by_fancybins
        # FIXME: and probably shouldnt be attributes, but on-the-fly variables
        dist_to_black = np.sum(self.RGB, axis=0)
        print dist_to_black
        stop
        self.theta_cwheel = find_theta_colorwheel(R, G, B)

        npix = len(self.dist_to_black)
        arr = np.arange(npix)
        nbins = 20
        n_per_chunk = npix/nbins + 1  # +1 fudge factor for rounding
        dist_to_black_broad = arr/n_per_chunk


        # Then sort by theta_cwheel within black
        #FIXME: you arent actually sorting by fancybins as is right now
        #idx_sort = np.lexsort((self.theta_cwheel, dist_to_black_broad))
        idx_sort = np.argsort(self.dist_to_black)
        self.x = self.x[idx_sort]
        self.y = self.y[idx_sort]
        self.RGB = self.RGB[idx_sort]
                
    def rearrange_pixels(self, target):
        """
        Here, target is the target image. 
        We map our current pixels (reference image) onto the
        target coordinates.
        This assumes that self and target are both already
        ordered correctly.
        """
        # First, sort for pure order
        self.x_new = target.x
        self.y_new = target.y