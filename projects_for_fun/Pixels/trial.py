import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_images():
    ref_file = 'figs/vangogh.jpg'
    img_file = 'figs/beaux.jpg'

    ref = Image.open(ref_file)
    img = Image.open(img_file)

    ref_pix = np.asarray(ref)
    img_pix = np.asarray(img)

    # Get miniarture versions
    npix = 5
    ref_mini = ref_pix[:npix]
    img_mini = img_pix[:npix]

    # Maybe bin according to if Red, Yellow, Blue is dominant color

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img_pix)
    print type(img_pix)
    print type(img_pix[0][0][0])
    #plt.show()
    # DataFrame columns:
    # index row col R G B index-of-ref
    # Theta value?

#read_images()
#stop
    
class ImageObj():
    """
    Some notes about this
    Can I make the returned instance just BE a dataframe
    For which I basically append on function calls not native
    to Pandas?
    """
    def __init__(self, image_dict):
        
        """
        Testing to find the format I want for eventual analysis
        image_dict shoudl eventually be an image
        But here as a dict, assuming initial processing done
        """

        self.df  = pd.DataFrame(image_dict)

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
        
        
def do_all_preprocessing(img_dict):
    """
    Returns:
       img: the fully processed/sorted image
    This should probably be subdeffed above too.
    """
    img = ImageObj(img_dict)
    img.add_distance_to_black_as_column()
    img.add_dominant_color_as_column()
    img.sort_by_distance_to_black()
    return img

def make_plots(tgt, ref):
    """
    This should eventually be animated
    """
    fig = plt.figure(figsize=(5,5))

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

tgt = do_all_preprocessing(tgt_dict)
ref = do_all_preprocessing(ref_dict)
ref.rearrange_pixels(tgt)
make_plots(tgt.df, ref.df)


