import Image
import numpy as np
import pandas as pd


class ImageObj():
    """
    FIXME: rename something like canvas?
    Some notes about this
    Can I make the returned instance just BE a dataframe
    For which I basically append on function calls not native
    to Pandas?
    """
    
    def __init__(self, path_to_jpg):
        
        """ Testing to find the format I want for eventual 
        analysis image_dict shoudl eventually be an image
        But here as a dict, assuming initial processing done
        """
        
        img = Image.open(path_to_jpg)
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

        self.df = pd.DataFrame(
            {'R': R, 'G': G, 'B': B, 'x': x, 'y': y}
        )
        self.add_distance_to_black_as_column()
        self.add_theta_colorwheel_as_column()

    def add_distance_to_black_as_column(self,):
        """
        In this simplest iteration, distance to black is just
        the total R + G + B value. Min = 0,0,0 (pure black)
        """
        df = self.df
        self.df.loc[:, 'dist_to_black'] = df.R + df.G + df.B
        
    def add_theta_colorwheel_as_column(self,):
        """ Obvious shortcoming - orange reds will be very close to red, 
        but purple reds will be very far.
        Maybe can solve this by resetting scale near least dense portion 
        of colormap?
        """
        df = self.df
        columns_keep = list(df.columns) # keep these + add 1 more
        
        just_rgb = df[['R','G','B']]
        rg = just_rgb.R - just_rgb.G
        gb = just_rgb.G - just_rgb.B
        br = just_rgb.B - just_rgb.R
        
        which_least = just_rgb.apply(np.argmin, axis=1)
        df.loc[:, 'which_least'] = which_least

        df.loc[:, 'rg'] = rg
        df.loc[:, 'gb'] = gb
        df.loc[:, 'br'] = br
        
        map_dict = {'R': 'gb', 'G': 'br', 'B': 'rg'}
        df.loc[:, 'which2use'] = df.which_least.map(map_dict)

        df.loc[:, 'val'] = df.apply(
            lambda row: row[row['which2use']], axis=1)
        df.loc[:, 'val'] = df.loc[:, 'val'] / 255.

        delta_theta = np.arccos(df.val)
        map_dict = {'rg': 0, 'gb': 2*np.pi/3., 'br': 4*np.pi/3.}
        theta_offset = df.which2use.map(map_dict)

        df.loc[:, 'theta_cwheel'] = delta_theta + theta_offset

        columns_keep.append('theta_cwheel')
        self.df = df[columns_keep]


    def sort_by_fancybins(self,):
        """ Bin by darkness into N bins
        Then sort by theta within those bins
        """
        self.df.sort(columns='dist_to_black', inplace=True)
        df = self.df

        npix = len(df)
        arr = np.arange(npix)
        nbins = 20
        n_per_chunk = npix/nbins + 1  # +1 fudge factor for rounding
        dist_to_black_broad = arr/n_per_chunk
        
        df.loc[:, 'dist_to_black_broad'] = dist_to_black_broad

        # Then sort by theta_cwheel within black
        df.sort(columns=['dist_to_black_broad', 'theta_cwheel'], inplace=True)
        self.df = df
                
    def rearrange_pixels(self, target):
        """
        Here, target is the target image. 
        We map our current pixels (reference image) onto the
        target coordinates.
        This assumes that self and target are both already
        ordered correctly.
        """
        self.df.loc[:, 'x_new'] = target.df.x.values
        self.df.loc[:, 'y_new'] = target.df.y.values
