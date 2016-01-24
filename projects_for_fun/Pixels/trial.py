import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#
# Or get average color of image, and map distance to that color?
# Also - only want uint8 once you go to PLOT. Til then int.
# Identify bottlenecks
# Add illustrator
# Convert whole thing to enhanced dataframe object?

fig = plt.figure(figsize=(14,10))
     
class ImageObj():
    """
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

        vmin = 0
        vstep = 100
        #img_pix = img_pix[vmin:vmin+vstep, vmin:vmin+vstep]
 
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

    def sort_by_distance_to_black(self,):
        """ Ascending, descending, who even cares.
        """
        self.df.sort(columns='dist_to_black', inplace=True)

    def sort_by_theta_colorwheel(self,):
        """ Ascending, descending, who even cares.
        """
        self.df.sort(columns='theta_cwheel', inplace=True)

    def sort_by_fancybins(self,):
        """ Bin by darkness into N bins
        Then sort by theta within those bins
        """
        self.sort_by_distance_to_black()
        nbins = 3  # play with this to see what gets best results
        
        print self.df
        stop
        
        
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
        
        
def do_all_preprocessing(path_to_jpg):
    """
    Returns:
       img: the fully processed/sorted image
    This should probably be subdeffed above too.
    """
    img = ImageObj(path_to_jpg)
    img.sort_by_distance_to_black()
    #img.sort_by_theta_colorwheel()
    #img.sort_by_fancybins()

    return img

def make_plots(tgt, ref):
    """
    This should eventually be animated
    """
    interpolation='hanning'
    
    ax = fig.add_subplot(221)
    pix_map = convert_to_imshow_format(tgt, 'x', 'y')
    ax.imshow(pix_map, interpolation=interpolation, cmap=plt.get_cmap('gray'))

    ax = fig.add_subplot(222)
    pix_map = convert_to_imshow_format(ref, 'x', 'y')
    ax.imshow(pix_map, interpolation=interpolation)

    ax = fig.add_subplot(223)
    pix_map = convert_to_imshow_format(ref, 'x_new', 'y_new')
    ax.imshow(pix_map, interpolation=interpolation)

    plt.tight_layout()
    plt.show()

def convert_to_imshow_format(df, xcol_name, ycol_name):
    """
    PM what it says
    """
    xmin = df[xcol_name].min()
    xmax = df[xcol_name].max()
    ymin = df[ycol_name].min()
    ymax = df[ycol_name].max()
    x_dim = xmax - xmin + 1
    y_dim = ymax - ymin + 1
    df.sort(columns=[ycol_name, xcol_name], inplace=True)
    rgb = df[['R','G','B']].values
    rgb = rgb.reshape((x_dim, y_dim, 3))
    rgb = rgb.astype(np.uint8)
    return rgb

target_path = 'figs/vermeer.jpg'
reference_path = 'figs/dali.jpg'

#reference_path = 'figs/vangogh.jpg'
#reference_path = 'figs/beaux.jpg'
#target_path = 'figs/temp_colorchart.jpg'

tgt = do_all_preprocessing(target_path)
ref = do_all_preprocessing(reference_path)
ref.rearrange_pixels(tgt)
make_plots(tgt.df, ref.df)
stop

tgt = do_all_preprocessing(tgt_dict)
ref = do_all_preprocessing(ref_dict)
ref.rearrange_pixels(tgt)
make_plots(tgt.df, ref.df)

    



