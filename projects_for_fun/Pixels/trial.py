import Image
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


    # DataFrame columns:
    # index row col R G B index-of-ref
    # Theta value?

    
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


tgt_dict = dict(
    R = [000, 100, 255, 45],
    G = [255, 075, 000, 45],
    B = [125, 210, 000, 45],
    x = [0, 1, 2, 3],
    y = [0, 0, 0, 0],
)
ref_dict = dict(
    R = [025, 170, 205, 215],
    G = [255, 075, 205, 045],
    B = [125, 210, 205, 145],
    x = [0, 1, 2, 3],
    y = [0, 0, 0, 0],
) # ref doesnt nec. need x & y, unless want to illustrate later... which I do!

tgt = do_all_preprocessing(tgt_dict)
ref = do_all_preprocessing(ref_dict)
#ref.rearrange_pixels(tgt_dict)

print tgt.df
print ref.df


