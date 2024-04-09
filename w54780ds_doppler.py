# -*- coding: utf-8 -*-
"""
Created on Tue Apr 09 2024
_________Title____________

"""
from scipy.optimize import fmin
from scipy.stats import iqr, zscore
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt

INCLINATION_ANGLE = 90  # degrees
INCLINATION_FACTOR = np.sin(INCLINATION_ANGLE * (180 / np.pi))
INIT_PARAMETER_ARRAY = [
    50,  # Velocity
    0.3,  # Why does this work and 3e-8 not??????
    0]
EMITTED_WAVELENGTH = 656.281
FILE_LIST = [
    "doppler_data_1.csv",
    "doppler_data_2.csv"
]
DELIMITER = ','
SKIP_HEADER = 1
PLOT_FILE_NAME = 'plot.png'
TIME = np.linspace(0, 6, 10000)


def init_data_filter():
    """
    Initial function which reads data from the data files in the data file
    list. It combines the data from the data files, removing any NaN values,
    and performing initial data filtering by comparing values of wavelength
    to the interquartile range.

    Finer data filtering is performed later when an initial estimate of the
    function is obtained.

    Returns
    -------
    array
        Numpy array containing the data combined from the given files.
    """
    temporary_array = np.zeros((0, 3))

    for file_name in FILE_LIST:
        temporary_array = np.vstack((temporary_array,
                                     np.genfromtxt(file_name,
                                                   dtype=float,
                                                   delimiter=DELIMITER,
                                                   skip_header=SKIP_HEADER,
                                                   comments='%')))
        print(f"File {file_name} processed.")

    array = temporary_array[~np.isnan(temporary_array).any(axis=1), :]
    array = np.delete(array, np.where(np.abs(array[:, 1]
                                             - EMITTED_WAVELENGTH) > iqr(
        array[:, 1])), 0)
    array[np.where(array[:, 2] == 0),2] = np.average(array[np.where(array[:,
                                                                    2] !=
                                                                    0),2])
    number_of_nans = len(np.argwhere(np.isnan(temporary_array)))
    print(f"{number_of_nans} invalid values found in data set.")

    return array


def wavelength_function(initial_velocity, angular_velocity, phase, time=TIME):
    doppler_shift = ((1 + (initial_velocity / constants.c) * np.sin(
        angular_velocity * time + phase) * INCLINATION_FACTOR) *
                     EMITTED_WAVELENGTH)

    return doppler_shift


def chi_squared(parameters, array):
    v_0 = parameters[0]
    angular_velocity = parameters[1]
    phase = parameters[2]
    y_vals = array[:, 1]
    time = array[:, 0]
    errors = array[:, 2]
    chi = np.sum((np.divide(
        (y_vals - wavelength_function(v_0, angular_velocity, phase,
                                      time=time)), errors) ** 2))

    return chi


def reduced_chi_squared(chi, parameters, data_points):
    chi_r = chi / (data_points - parameters)

    return chi_r


def outlier_indices(array, parameter_array):
    times = array[:, 0]
    y_values = array[:, 1]
    errors = array[:, 2]

    indices = np.where(
        np.abs(y_values - wavelength_function(parameter_array[0],
                                              parameter_array[1],
                                              parameter_array[2],
                                              time=times)) > errors * 3)

    return indices


def data_filterer(array, parameters):
    """
    Filters data again.

    """
    #array = np.delete(array, np.where(array[:, 2] == 0), 0)
    zscores = (array[:, 1] - wavelength_function(
        parameters[0],
        parameters[1],
        parameters[2],
        time=array[:, 0]
    )) / array[:, 2]
    threshold = 3
    outliers = np.abs(zscores)
    array = np.delete(array, np.where(outliers > threshold), 0)

    return array


def data_point_plotter(array, prediction, outlier_indices):
    # x_outliers = array[outlier_indices, 0]
    # y_outliers = array[outlier_indices, 1]
    # array = np.delete(array, outlier_indices, 0)
    x = array[:, 0]
    y = array[:, 1]
    error_bars = array[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, yerr=error_bars, fmt="bo")
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\lambda(t)$')
    ax.plot(
        TIME,
        prediction,
        "r-"
    )
    plt.show()


def main() -> None:
    data = init_data_filter()
    chi_r = 5
    parameter_guess = INIT_PARAMETER_ARRAY
    iterations = 0

    # Could try doing this recursively instead.
    while chi_r > 1:
        iterations += 1
        optimised_parameters = fmin(chi_squared,
                                    parameter_guess,
                                    args=(data,),
                                    disp=False)

        chi = chi_squared(optimised_parameters, data)

        chi_r = reduced_chi_squared(chi, len(optimised_parameters),
                                    len(data))

        print(f"Iter {iterations}: {chi_r}")
        print(f"Velocity: {optimised_parameters[0]}\n"
              f"Angular Velocity: {optimised_parameters[1]}\n"
              f"Phase: {optimised_parameters[2]}\n")
        data = data_filterer(data, optimised_parameters)
        parameter_guess = optimised_parameters

    predicted_function = wavelength_function(parameter_guess[0],
                                             parameter_guess[1],
                                             parameter_guess[2])

    data_point_plotter(data, predicted_function,
                       outlier_indices(data, parameter_guess))

    print(f"Reduced chi squared value of {chi_r} obtained after {iterations} "
          f"optimisation iterations.")


if __name__ == '__main__':
    main()
