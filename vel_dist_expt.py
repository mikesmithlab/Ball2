from Generic.filedialogs import BatchProcess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Generic.fitting import Fit


def histogram(array, numbins=11, range=None):
    freq, binedges=np.histogram(array, bins=numbins, range=range)
    bins=0.5*(binedges[1:]+binedges[:-1])
    return bins, freq


def plot_hist(bins, freq, title ,norm=True,fignum=1, marker='rx'):
    if norm:
        freq=freq/np.sum(freq)
    plt.figure(fignum)
    plt.plot(bins, freq,marker)
    plt.title(title)


def plot_x_y(x,y,xlabel,ylabel,fignum=2, marker='rx'):
    plt.figure(fignum)
    plt.plot(x,y,marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def fit_data(x,y):
    f=Fit('linear',x=x,y=y)
    f.add_params([0,-1])
    f.fit()
    f.fit_errors(100)
    #f.plot_fit()
    return f.fit_params[0], f.fit_param_errors



if __name__ == '__main__':
    #basepath = '/media/ppzmis/data/BouncingBall_Data/newMovies/RawDataandTracking/P80/'
    #name = '*80_077_data.hdf5'

    basepath = '/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed/'
    name = '*80_077_data_finaldata.hdf5'
    #name = '*240_062_data_finaldata.hdf5'
    print(basepath + name)
    files = BatchProcess(pathfilter=basepath + name)


    vtot = np.array([])
    wtot = np.array([])

    for file in files:
        print(file)
        data = pd.read_hdf(file)

        v = data['xVelMM'].to_numpy()
        w = data['omega_k'].to_numpy()

        vtot = np.append(vtot, v)
        wtot = np.append(wtot, w)

    bins, freq = histogram(vtot, numbins=15, range=(-100,100))
    binsw, freqw = histogram(wtot, numbins=15, range=(-20, 20))
    plot_hist(bins, freq, 'v', fignum=1, marker='rx-')
    plot_hist(binsw, freqw, 'w', fignum=2, marker='rx-')

    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/supplementary_fig/P80_077_vxhist.txt',np.c_[bins,freq])
    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/supplementary_fig/P80_077_wzhist.txt',np.c_[binsw,freqw])

    plt.show()
