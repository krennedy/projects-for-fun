import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


### OOh
# Or get average color of image, and map distance to that color?
# Also - convert to int8 earlier
fig = plt.figure(figsize=(14,10))


def read_images():
    ref_file = 'figs/vangogh.jpg'
    ref = Image.open(ref_file)
    ref_pix = np.asarray(ref)
    ax.imshow(ref_pix)

     
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
 
        
        R = img_pix[:,:,0].ravel()
        G = img_pix[:,:,1].ravel()
        B = img_pix[:,:,2].ravel()

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
        
    def add_distance_to_black_as_column(self,):
        """
        In this simplest iteration, distance to black is just
        the total R + G + B value. Min = 0,0,0 (pure black)
        """
        df = self.df
        self.df.loc[:, 'dist_to_black'] = df.R + df.G + df.B
  
    def add_dominant_color_as_column(self,):
        """
        Does R/G/B or None dominate?
        """
        idxmax = self.df[['R','G','B']].idxmax()
        index = idxmax.values
        maxvals = idxmax.index
        self.df.loc[index, 'dominant_color'] = maxvals
        self.df.fillna('None', inplace=True)

    def sort_by_distance_to_black(self,):
        """ Ascending, descending, who even cares.
        """
        self.df.sort(columns='dist_to_black', inplace=True)

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
    img.add_distance_to_black_as_column()
    img.add_dominant_color_as_column()
    img.sort_by_distance_to_black()
    return img

def make_plots(tgt, ref):
    """
    This should eventually be animated
    """
    ax = fig.add_subplot(221)
    pix_map = convert_to_imshow_format(tgt, 'x', 'y')
    ax.imshow(pix_map, interpolation='none')

    ax = fig.add_subplot(222)
    pix_map = convert_to_imshow_format(ref, 'x', 'y')
    ax.imshow(pix_map, interpolation='none')

    ax = fig.add_subplot(223)
    pix_map = convert_to_imshow_format(ref, 'x_new', 'y_new')
    ax.imshow(pix_map, interpolation='none')
    
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

target_path = 'figs/vangogh.jpg'
reference_path = 'figs/beaux.jpg'

tgt = do_all_preprocessing(target_path)
ref = do_all_preprocessing(reference_path)
ref.rearrange_pixels(tgt)
make_plots(tgt.df, ref.df)
stop

tgt = do_all_preprocessing(tgt_dict)
ref = do_all_preprocessing(ref_dict)
ref.rearrange_pixels(tgt)
make_plots(tgt.df, ref.df)

    
tgt_dict = dict(
    R = [255, 000, 000, 255],
    G = [000, 255, 000, 255],
    B = [000, 000, 255, 255],
    x = [0, 1, 0, 1],
    y = [0, 0, 1, 1],
)
ref_dict = dict(
    R = [025, 170, 205, 215],
    G = [255, 075, 205, 045],
    B = [125, 210, 205, 145],
    x = [10, 11, 10, 11],
    y = [10, 10, 11, 11],
) # ref doesnt nec. need x & y, unless want to illustrate later... which I do!



