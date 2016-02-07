import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as animation

#
# Or get average color of image, and map distance to that color?
# Identify bottlenecks
# Convert whole thing to enhanced dataframe object?
# Handle different sized inputs

fig = plt.figure(figsize=(14,10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)


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


class Animator():
    
    def __init__(self, A_df, B_df):
        self.pix_A, self.df_A_sorted = convert_to_imshow_format(A_df, 'x', 'y')
        self.pix_B, self.df_B_sorted = convert_to_imshow_format(B_df, 'x', 'y')
                
        self.pix_B_new = np.ones(self.pix_B.shape).astype('uint8') * 255
        self.pix_A_new = np.ones(self.pix_A.shape).astype('uint8') * 255
        self.pix_A_original = self.pix_A.copy()
        self.pix_B_original = self.pix_B.copy()

        # Initialize
        kwargs = dict(animated=True, interpolation='none')
        self.im1 = ax1.imshow(self.pix_A, **kwargs)
        self.im2 = ax2.imshow(self.pix_B, **kwargs)
        self.im3 = ax3.imshow(self.pix_B_new, **kwargs)
        self.im4 = ax4.imshow(self.pix_A_new, **kwargs)

        nsteps = 10
        self.npix = self.pix_A.shape[0] * self.pix_A.shape[1]
        self.nstep = int(self.npix/float(nsteps))

    def draw(self,):
        ani = animation.FuncAnimation(
            fig, self.updatefig,
            np.arange(0, self.npix, self.nstep),
            interval=100, blit=False, repeat=False)
        plt.show()

    def updatefig(self, j):
        """ Update att 4
        """
        self.im1.set_array(self.take_out(self.pix_A, j))
        self.im2.set_array(self.take_out(self.pix_B, j))
        self.im3.set_array(self.put_in(self.pix_B_new, self.pix_A_original,
                                       self.df_A_sorted, j))
        self.im4.set_array(self.put_in(self.pix_A_new, self.pix_B_original,
                                       self.df_B_sorted, j))

        # return top row to original state if at end of animation
        if j+self.nstep >= len(self.df_A_sorted):
            self.im1.set_array(self.pix_A_original)
            self.im2.set_array(self.pix_B_original)

    def take_out(self,img, i):
        img_shape = img.shape
        img = img.reshape(img_shape[0]* img_shape[1], img_shape[2])
        img[i:i+self.nstep] = np.array([155,155,155]).astype('uint8')
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
        positions_this_time = positions[i:i+self.nstep]

        img_put[positions_this_time] = img_pull[i:i+self.nstep]

        img_put = img_put.reshape(img_shape)
        return img_put


def preprocess(path_to_jpg):
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

    
def main():
    A_path = 'figs/vermeer.jpg'
    B_path = 'figs/dali.jpg'

    A_obj = preprocess(A_path)
    B_obj = preprocess(B_path)

    A_obj.rearrange_pixels(B_obj)
    B_obj.rearrange_pixels(A_obj)

    an_example = Animator(A_obj.df, B_obj.df)
    an_example.draw()


main()


