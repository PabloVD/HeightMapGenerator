#---------------------------
# Height Map Generator
# Generates height maps using gaussian fields from a given power spectrum, with the package powerbox, which are smoothed afterwards with a gaussian filter
# Author: Pablo Villanueva-Domingo
# Last update: 23/12/2022
#---------------------------

import numpy as np
import powerbox as pbox
from scipy import interpolate, ndimage

# Class for generating height maps
class GenerateHeightMap():
    def __init__(self,
                # Number of bins per dimension
                boxsize = 1000,
                # Sigma for the gaussian smoothing
                sigma = 5.,
                # Spectral index for the power spectrum
                indexlaw = -3.
                ):

        self.boxsize = boxsize
        self.sigma = sigma
        self.indexlaw = indexlaw
        self.field = None
        self.name = "heightmap_indexlaw_{:+.1f}_sigma_{:.1f}".format(indexlaw,sigma)

    # Define power spectrum as a power law with an spectral index indexlaw
    # With lower the spectral indexes, small structures are removed
    def powerspec(self, k):

        return k**self.indexlaw

    # Filter the field with a gaussian window
    def smooth_field(self, field):

        x, y = np.linspace(0,field.shape[0],num=field.shape[0]), np.linspace(0,field.shape[1],num=field.shape[1])

        # Interpolation
        f = interpolate.RectBivariateSpline(x,y,field)

        qx = np.linspace(x[0],x[-1], num = self.boxsize)
        qy = np.linspace(y[0],y[-1], num = self.boxsize)

        # Filtering
        smooth = ndimage.gaussian_filter(f(qx,qy),self.sigma)

        return smooth


    # Normalize the values of the field between 0 and 1
    def normalize_field(self, field):

        min, max = np.amin(field), np.amax(field)
        newfield = (field-min)/(max-min)

        return newfield

    # Generate a height map applying different processes:
    # 1. Generate a random gaussian field given a power spectrum
    # 2. Normalize the field between 0 and 1
    # 3. Smooth the field with a gaussian filter
    def generate_hmap(self):
        
        field = pbox.powerbox.PowerBox(self.boxsize, lambda k: self.powerspec(k), dim=2, boxlength=self.boxsize).delta_x()
        field = self.normalize_field(field)
        #field = self.smooth_field(field)
        self.field = field

        return field

    
#--- MAIN ---#

if __name__=="__main__":

    import os
    import matplotlib.pyplot as plt

    outpath = "hmaps"

    if not os.path.exists(outpath):
        os.system("mkdir "+outpath)

    # Uncomment to allow reproducibility
    np.random.seed(seed=1234)

    # Number of height maps to be generatede
    Nmaps = 10

    # Height map parameters
    # Number of bins per dimension
    boxsize = 1000
    # Sigma for the gaussian smoothing
    sigma = 5.
    # Spectral index for the power spectrum
    indexlaw = -3.

    my_dpi=96

    for i in range(Nmaps):

        hmap = GenerateHeightMap(boxsize=boxsize, sigma=sigma, indexlaw=indexlaw)

        field = hmap.generate_hmap()

        fig, ax = plt.subplots(figsize=(boxsize/my_dpi, boxsize/my_dpi), dpi=my_dpi)
        fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        ax.imshow(field,vmin=0.,vmax=1.,cmap="Greys")
        plt.axis('off')
        fig.savefig(outpath+"/"+hmap.name+"_"+str(i)+".png", bbox_inches='tight', pad_inches=0)
        fig.clf()

    print("Done!")
