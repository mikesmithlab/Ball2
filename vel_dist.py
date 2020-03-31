
from Generic.fitting import Fit, lorentzian, area_gaussian, gaussian
from Generic.filedialogs import BatchProcess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.ndimage.filters import gaussian_filter
import os
from Generic.fitting import Fit

def hist_plot(variable, numbins=None, marker='r.',fig_num=2, range=None, xlabel='', ylabel='', title=''):
    freq, binedges = np.histogram(variable, bins=numbins, range=range)
    binedges
    freq
    bins = 0.5 * (binedges[1:] + binedges[:-1])
    freq = freq / np.sum(freq)
    plt.figure(fig_num)
    plt.plot(bins, freq, marker)
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.title = title
    return bins, freq


if __name__ == '__main__':

    #pathfilter='/media/ppzmis/data/BouncingBall_Data/NewBouncingBall/P80/traj*g3.25_?.dat'
    pathfilter= '/media/ppzmis/data/BouncingBall_Data/NewBouncingBall/P80new/traj*f1*g2.75*.dat'

    file_iterator = BatchProcess(pathfilter=pathfilter)


    output_path = '/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/'
    output_file = 'P80_3.25'

    t = np.array([])
    x = np.array([])
    vx = np.array([])
    wx = np.array([])

    for index, filename in enumerate(file_iterator):  # [filename, filename2, filename3]

        print(filename)
        data = np.loadtxt(filename)
        # Complete arrays
        t = np.append(t, np.array(data[:, 0]))
        x = np.append(x, np.array(data[:, 1]))
        vx_temp = np.array(data[:, 2])
        vrw = np.array(data[:,3])
        wx_temp = vx_temp / (5*vrw)
        if np.mean(np.abs(vx_temp)) > 0.018:
            vx = np.append(vx, vx_temp)
            wx = np.append(wx, wx_temp)
            print('ok')
            print(index)
        else:
            print('spiked')
            print(index)
        print(np.mean(np.abs(vx_temp)))

    #position_filter = np.abs(x) < (0.5*np.max(x))
    print(np.mean(np.abs(vx)))

    bins, freq = hist_plot(vx, numbins=50, range=(-0.15, 0.15))
    binsw, freqw = hist_plot(wx, numbins=50, range=(-0.03, 0.03))
    print(filename[:-4])
    np.savetxt(filename[:-4] + '_combined_vx.txt',np.c_[bins, freq])
    np.savetxt(filename[:-4] + '_combined_wx.txt', np.c_[binsw, freqw])
    plt.show()