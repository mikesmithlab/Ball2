
from Generic.fitting import Fit, lorentzian, area_gaussian, gaussian
from Generic.filedialogs import BatchProcess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.ndimage.filters import gaussian_filter
import os


def values_in_xbin(array2bin,x,lower=0,upper=0):
    selection = np.array((x>lower)&(x<upper))
    vals = array2bin[selection]
    freq, binedges = np.histogram(vals, bins=100, range=(-2,3))
    bins = 0.5*(binedges[:-1]+binedges[1:])
    return bins, freq










if __name__ == '__main__':
    file_iterator = BatchProcess(pathfilter='/media/ppzmis/data/BouncingBall_Data/newMovies/SimulationData/new24092019/combined_datasets/impavP80*3.25*combined.dat')

    for filename in file_iterator:#[filename, filename2, filename3]
        print(filename)

        data = np.loadtxt(filename)

        # check whether vx[0] is velocity b4 bounce or after.0.00974381480255445
        b4bounce = np.array(data[:, 5]).astype(bool)
        aftbounce = np.array(data[:, 4]).astype(bool)

        t = np.array(data[:, 0])
        x = np.array(data[:, 1])
        vx = np.array(data[:, 2])
        vrw = np.array(data[:, 3])

        t_b4 = t[b4bounce]

        # t_aft = t[~bounce, 0]
        x = data[b4bounce, 1]
        x_aft = data[aftbounce, 1]

        Distance_from_wall = 0.005

        vx_b4 = np.array(data[b4bounce, 2])

        vx_aft = np.array(data[aftbounce, 2])

        vrwb4 = np.array(data[b4bounce, 3])
        vrwaft = np.array(data[aftbounce, 3])

        dvx = vx_aft - vx_b4

        xbinedges = np.linspace(0.005,0.025,10)
        plt.figure()
        for i in range(np.size(xbinedges)-1):
            bins, freq = values_in_xbin(vrwb4, x, xbinedges[i], xbinedges[i+1])
            plt.plot(bins, freq)
            np.savetxt(filename[:-4] + str(i) + '_vrwforxbins.txt', np.c_[bins[:], freq[:]/np.sum(freq)])
        plt.savefig(filename[:-4] + '_vrwforxbins.png')

