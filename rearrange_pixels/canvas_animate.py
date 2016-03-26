import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from pixel_tracker import PixelTracker

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

