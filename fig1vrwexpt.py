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
    f.add_params([0,-1], lower=[None, 'Fixed'], upper=[None, 'Fixed'])
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

    t_bounce_tot = np.array([])
    x_bounce_tot = np.array([])
    v_b4_tot = np.array([])
    v_aft_tot = np.array([])
    dv_tot = np.array([])
    vwr_b4_tot = np.array([])
    vwr_aft_tot = np.array([])
    w_b4_tot = np.array([])
    w_aft_tot = np.array([])

    vtot = np.array([])
    vwrtot = np.array([])

    for file in files:
        print(file)
        data = pd.read_hdf(file)
        print(data.columns)
        #print(list(data.columns))
        t = data.index.to_numpy()/500.0
        x = data['ballXMM'].to_numpy()
        #y = data['ballHeightMM'].to_numpy()
        v = data['xVelMM'].to_numpy()

        #np.savetxt(basepath + 'P80_ytraj.txt', y[:2000])
        #np.savetxt(basepath + 'P80_ttraj.txt', t[:2000])

        bounce=np.argwhere(data['bounce'].to_numpy())
        w = data['omega_k'].to_numpy()
        vwr = -v / (5*w)

        vtot = np.append(vtot, v)
        vwrtot = np.append(vwrtot, vwr)

        len_bounce = np.size(bounce)

        dt_bounce = t[bounce[1:]] - t[bounce[:-1]]
        x_bounce = x[bounce[1:]]
        v_b4_bounce = v[bounce[1:]-1]
        v_aft_bounce = v[bounce[1:]]
        dv = v_aft_bounce - v_b4_bounce
        w_b4_bounce = w[bounce[1:]-1]
        w_aft_bounce = w[bounce[1:]]
        vwr_b4_bounce = vwr[bounce[1:]-1]
        vwr_aft_bounce = vwr[bounce[1:]]

        # b4 bounce is first data point.
        #dt_bounce = t[bounce[1:]] - t[bounce[:-1]]
        #x_bounce = x[bounce[1:]]
        #v_b4_bounce = v[bounce[:-1]]
        #v_aft_bounce = v[bounce[1:]]
        #dv = v_aft_bounce - v_b4_bounce
        #w_b4_bounce = w[bounce[:-1]]
        #w_aft_bounce = w[bounce[1:]]
        #vwr_b4_bounce = -vwr[bounce[:-1]]
        #vwr_aft_bounce = -vwr[bounce[1:]]

        t_bounce_tot = np.append(t_bounce_tot, dt_bounce)
        x_bounce_tot = np.append(x_bounce_tot, x_bounce)
        v_b4_tot = np.append(v_b4_tot, v_b4_bounce)
        v_aft_tot = np.append(v_aft_tot, v_aft_bounce)
        dv_tot = np.append(dv_tot, dv)
        vwr_b4_tot = np.append(vwr_b4_tot, vwr_b4_bounce)
        vwr_aft_tot = np.append(vwr_aft_tot, vwr_aft_bounce)
        w_b4_tot = np.append(w_b4_tot, w_b4_bounce)
        w_aft_tot = np.append(w_aft_tot, w_aft_bounce)


    bins, freq = histogram(vwr_b4_tot, numbins=21, range=(-2,3))
    plot_hist(bins, freq, 'vwr', fignum=6, marker='rx-')
    print(np.sum(freq))

    np.savetxt('/media/ppzmis/data/BouncingBall_Data/PaperDataAndGraphs/P80_077expt_vwrhist.txt',np.c_[bins,freq])

    plt.show()
