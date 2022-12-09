"""
Kalman Filter for 1D-tracking applications
Simulation: Tracking of a Kartal vehicle using Kalman filter.

@VatozZ
09.12.2022

"""

close_plotting_legacy = True # False shows the previous estimates

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

file = 'kartal.png'
im = plt.imread('kartal.png') # insert local path of the image.
fig, ax = plt.subplots()

zn = 1 #location of measurement
xn = 1 #initial position (state)
pn = 1 #initial deviation of state (covariance)
rn = 1 #measurement noise

n_simulation = 10

resolution_of_x_axis = 0.5

uncertainity_records = []

for i in range(0, n_simulation):


    Kn = pn / (pn + rn) #Kalman Gain

    xn = xn + Kn * (zn - xn) #state update
    pn = (1-Kn)*pn #covariance update(uncertainity of the state)

    zn = zn + 1 #move target

    uncertainity_records.append(pn)

    mu, sigma = xn, pn
    time_axis = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.imshow(im, origin='lower', extent=[xn - pn, xn + pn, 0, 1], aspect=1)
    plt.plot(time_axis, stats.norm.pdf(time_axis, mu, sigma), label='t='+str(i))
    #plt.xlim([0, n_simulation])
    x_t = np.arange(0, n_simulation, resolution_of_x_axis)
    plt.title('estimated position:' + str(round(xn, 1))+'  '+ 'uncertainity of the position:' + str(round(pn, 2)))
    plt.xticks(x_t)
    plt.legend()
    plt.pause(1)

    if close_plotting_legacy:
        plt.clf()

plt.close('all')
plt.figure()
plt.title('Uncertainity changes')
plt.plot(uncertainity_records)
plt.show()

