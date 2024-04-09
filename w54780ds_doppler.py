# -*- coding: utf-8 -*-
"""
Created on Tue Apr 09 2024
_________Title____________

"""
from scipy.optimize import fmin
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt

EMITTED_WAVELENGTH = 656.281  # nanometres
INITIAL_VELOCITY_GUESS = 50
INITIAL_ANGULAR_VELOCITY_GUESS = 0.2
PHASE_GUESS = np.pi
FILE_LIST = [
    "doppler_data_1.csv",
    "doppler_data_2.csv"
]
DELIMITER = ','
SKIP_HEADER = 1
PLOT_FILE_NAME = 'plot.png'
TIME = np.linspace(0, 6, 10000)

# TODO: We can use masked arrays to deal with outliers. The use case where we
#  can remove values close to another value.


def data_getter_and_cleaner():
    """
    Reads from the data files provided in FILE_LIST and combines them into a
    single numpy array. Any invalid data (i.e., nans) are removed.

    Returns
    -------
    array
        Numpy array containing the data combined from the given files.
    """
    temporary_array = np.zeros((0, 3))

    for file_names in FILE_LIST:
        temporary_array = np.vstack((temporary_array,
                                     np.genfromtxt(file_names,
                                                   dtype=float,
                                                   delimiter=DELIMITER,
                                                   skip_header=SKIP_HEADER,
                                                   invalid_raise=False,
                                                   comments='%')))

    array = temporary_array[~np.isnan(temporary_array).any(axis=1), :]

    array = np.delete(array, np.where(np.abs(array[:,1] - EMITTED_WAVELENGTH) >
                                      10), 0)

    return array


def function(v_0, omega,phase):
    doppler_shift = lambda t : 1 + (v_0/constants.c)*np.sin(omega*t + phase)
    doppler_shift = doppler_shift(TIME)*EMITTED_WAVELENGTH

    return doppler_shift


def fun(vwp):
    v_0 = vwp[0]
    omega = vwp[1]
    phase = vwp[2]
    array = data_getter_and_cleaner()
    array = np.delete(array, np.where(array[:,2] == 0),0)
    y_vals = array[:,1]
    time = array[:,0]
    errors = array[:,2]

    with np.errstate(divide='ignore', invalid='ignore'):
        chi = np.sum((np.divide((y_vals - (EMITTED_WAVELENGTH*(1 + (
            v_0/constants.c)*np.sin(omega*time +
                                    phase)))),errors))**2)

    return chi

def data_point_plotter(array,prediction):
    x = array[:,0]
    y = array[:,1]
    error_bars = array[:,2]
   # y = np.delete(y, np.where(np.abs(prediction - y) > 3 * error_bars))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, yerr=error_bars,fmt="bo")
    ax.set_xlabel(r't')
    ax.set_ylabel(r'$\lambda(t)$')
    ax.plot(
        TIME,
        prediction,
        "r-"
    )
    plt.show()

def main() -> None:
    print(fun([INITIAL_VELOCITY_GUESS, INITIAL_ANGULAR_VELOCITY_GUESS, PHASE_GUESS]))

    result = fmin(fun, (INITIAL_VELOCITY_GUESS, INITIAL_ANGULAR_VELOCITY_GUESS,
                     PHASE_GUESS),
                  full_output=True, disp=False)
    print(result)
    print(fun(result[0]))
    predict = function(result[0][0],result[0][1],result[0][2])
    data_point_plotter(data_getter_and_cleaner(),predict)

if __name__ == '__main__':
    main()
