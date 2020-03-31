import numpy as np
import matplotlib.pyplot as plt

g = 9.81
omega = 2*np.pi*50
gamma = np.array([2,2.25,2.5,2.75,3,3.25,3.5])

t_spikes = []
for i in range(np.size(gamma)):
    t = np.linspace(0,0.04,10000)

    A = g/omega**2
    B = (gamma[i])*g/omega*((1-1/(gamma[i]**2))**0.5)*t
    C = -0.5*g*t**2
    D = -(g*gamma[i]/omega**2)*np.sin(omega*t + np.arcsin(1/gamma[i]))

    dz = A + B + C + D

    root_index = np.argmin(dz[500:]**2)
    root_index = root_index + 500
    t_root = t[root_index]
    print(t_root)
    t_spikes.append(t_root)

t_spikes = np.array(t_spikes)

plt.figure()
plt.plot(gamma,t_spikes,'rx-')
plt.xlabel('Gamma')
plt.ylabel('t spike (s)')
plt.show()