from Generic.filedialogs import BatchProcess
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    pathfilter = '/media/ppzmis/data/BouncingBall_Data/newMovies/SimulationData/new24092019/combined_datasets_tbounce/imp*combined.dat'
    files=BatchProcess(pathfilter=pathfilter)

    for file in files:
        print(file)
        data = np.loadtxt(file)

        #read every other row of the time data and work out the difference for time between bounces.
        t_diff = np.array(data[1:-3:2,0]-data[0:-4:2,0])
        #this is the dv following the bounce
        v_diff = np.array(data[2:-2:2,2]-data[1:-3:2,2])

        freq, binedges = np.histogram(v_diff)
        bins = 0.5*(binedges[1:] + binedges[:-1])
        plt.figure()
        plt.plot(bins,freq)
        plt.show()

        print(np.size(t_diff[t_diff <=0.02])/np.size(t_diff))
        print(np.size(t_diff[t_diff > 0.02]))
        freq, bin_edges =np.histogram(t_diff,bins=500, range=(0, 0.05))

        freq = freq/np.sum(freq)
        bins = 0.5*(bin_edges[:-1]+bin_edges[1:])
        tspike=bins[np.argmax(freq[100:])+100]
        print(tspike)
        print(np.size(t_diff[t_diff <= tspike]) / np.size(t_diff))


        plt.figure()
        plt.plot(bins, freq, 'rx')
        plt.plot(bins, freq, 'b-')
        plt.ylabel('Frequency')
        plt.xlabel('Time between bounces (s)')
        plt.ylim([0,0.03])
        plt.title(file.split('imp')[1])

        plt.savefig(file[:-4]+'.png')
        plt.close()
