import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as animation

#
# Or get average color of image, and map distance to that color?
# Identify bottlenecks
# Add illustrator
# Convert whole thing to enhanced dataframe object?
# Handle different sized inputs

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
        self.df = sort_by_column(self.df, 'dist_to_black')

    def sort_by_theta_colorwheel(self,):
        self.df = sort_by_column(self.df, 'theta_cwheel')

    def sort_by_fancybins(self,):
        """ Bin by darkness into N bins
        Then sort by theta within those bins
        """
        self.sort_by_distance_to_black()
        df = self.df

        npix = len(df)
        arr = np.arange(npix)
        nbins = 20
        n_per_chunk = npix/nbins + 1  # +1 fudge factor for rounding
        dist_to_black_broad = arr/n_per_chunk
        
        df.loc[:, 'dist_to_black_broad'] = dist_to_black_broad

        # Then sort by theta_cwheel within black
        df = sort_by_column(df, ['dist_to_black_broad', 'theta_cwheel'])
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
        

def sort_by_column(df, col_name):
    """ Ascending, descending, who even cares.
    """
    df.sort(columns=col_name, inplace=True)
    return df

def do_all_preprocessing(path_to_jpg):
    """
    Returns:
       img: the fully processed/sorted image
    This should probably be subdeffed above too.
    """
    img = ImageObj(path_to_jpg)
    img.sort_by_fancybins()
    return img
    
def convert_to_imshow_format(df, xcol_name, ycol_name):
    """
    PM what it says.
    Takes almost no time!
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
    return rgb, df


A_path = 'figs/vermeer.jpg'
B_path = 'figs/dali.jpg'

#reference_path = 'figs/vangogh.jpg'
#target_path = 'figs/beaux.jpg'

A_obj = do_all_preprocessing(A_path)
B_obj = do_all_preprocessing(B_path)
A_obj.rearrange_pixels(B_obj)
B_obj.rearrange_pixels(A_obj)
#make_plots(A_obj.df, B_obj.df)


pix_A, df_A_sorted = convert_to_imshow_format(A_obj.df, 'x', 'y')
pix_B, df_B_sorted = convert_to_imshow_format(B_obj.df, 'x', 'y')


###############
# WOULD BE NICE OBJECT ORIENT ANIMATION STUFF TOO?
###############

#nstep = 500

fig = plt.figure(figsize=(14,10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

pix_B_new = np.ones(pix_B.shape).astype('uint8') * 255
pix_A_new = np.ones(pix_A.shape).astype('uint8') * 255
pix_A_original = pix_A.copy()
pix_B_original = pix_B.copy()


# Initialize
im1 = ax1.imshow(pix_A, animated=True, interpolation='none')
im2 = ax2.imshow(pix_B, animated=True, interpolation='none')
im3 = ax3.imshow(pix_B_new, animated=True, interpolation='none')
im4 = ax4.imshow(pix_A_new, animated=True, interpolation='none')

npix = pix_A.shape[0] * pix_A.shape[1]
nstep = int(npix/10.0)

class Animator():
    def __init__(self,):
        nsteps = 10
        self.npix = pix_A.shape[0] * pix_A.shape[1]
        self.nstep = int(self.npix/float(nsteps))

    def draw(self,):
        ani = animation.FuncAnimation(
            fig, self.updatefig,
            np.arange(0, self.npix, self.nstep),
            interval=100, blit=False, repeat=False)
        plt.show()

    def updatefig(self,j):
        """ Update att 4
        """
        im1.set_array(self.take_out(pix_A, j))
        im2.set_array(self.take_out(pix_B, j))
        im3.set_array(self.put_in(pix_B_new, pix_A_original, df_A_sorted, j))
        im4.set_array(self.put_in(pix_A_new, pix_B_original, df_B_sorted, j))

        # return top row to original state if at end of animation
        if j+nstep >= len(A_obj.df):
            im1.set_array(pix_A_original)
            im2.set_array(pix_B_original)

    def take_out(self,img, i):
        img_shape = img.shape
        img = img.reshape(img_shape[0]* img_shape[1], img_shape[2])
        img[i:i+nstep] = np.array([155,155,155]).astype('uint8')
        img = img.reshape(img_shape)

        return img 

    def put_in(self,img_put, img_pull, df_sorted, i):
        """ Here, you want to put in a few pixels of new image
        But using the values input from another image
        !!! img_pull shouldnt need to have to be flattened each time
        """
        img_shape = img_put.shape
        new_shape = (img_shape[0]* img_shape[1], img_shape[2])


        img_put = img_put.reshape(new_shape)
        img_pull = img_pull.reshape(new_shape)

        x_new = df_sorted.x_new
        y_new = df_sorted.y_new
        x_dim = x_new.max() + 1
        y_dim = y_new.max() + 1
        positions = y_new * x_dim + x_new # or reverse x's and y's
        positions_this_time = positions[i:i+nstep]

        img_put[positions_this_time] = img_pull[i:i+nstep]

        img_put = img_put.reshape(img_shape)
        return img_put



an_example = Animator()
an_example.draw()
#plt.show()
