# -*- coding: utf-8 -*-
"""
Created on Tue Apr 09 2024
______Doppler Shifts______
TODO: Find errors on fitted parameters.
TODO: Calculate m_p and r and extrapolate their uncertainties
TODO: Put calculated values on plot
TODO: Make plot look nice
TODO: Make code readable
TODO: Add more validation checks
TODO: Make a guesser function to make intial estimates of fitted parameters
"""


from scipy.optimize import fmin
from scipy.stats import iqr
from scipy import constants
import numpy as np
import matplotlib.pyplot as plt

INCLINATION_ANGLE = 90  # degrees
INCLINATION_FACTOR = np.sin(INCLINATION_ANGLE * (180 / np.pi))
INIT_PARAMETER_ARRAY = [
    50,  # Velocity
    3e-8 * 365 * 24 * 3600,  # rad/s -> rad/year
    0  # rad
]
EMITTED_WAVELENGTH = 656.281  # nanometres
FILE_LIST = [
    "doppler_data_1.csv",
    "doppler_data_2.csv"
]
DELIMITER = ','
SKIP_HEADER = 1
PLOT_FILE_NAME = 'plot.png'
TIME = np.linspace(0, 6, 10000)  # Years - make this dynamic later
X_LABEL = r'$t$ / s'
Y_LABEL = r'$\lambda(t)$ / nm'
FIT_TOLERANCE = 1  # Minimum reduced chi squared


def data_getter():
    temporary_array = np.zeros((0, 3))

    for file_name in FILE_LIST:
        try:
            temporary_array = np.vstack((temporary_array,
                                         np.genfromtxt(file_name,
                                                       dtype=float,
                                                       delimiter=DELIMITER,
                                                       skip_header=SKIP_HEADER,
                                                       comments='%')))
            print(f"File {file_name} processed.")
        except FileNotFoundError:
            print(f"File {file_name} not found in local path. Skipping.")

    return temporary_array


def wavelength_function(parameters, time=TIME):
    doppler_shift = ((1 + (parameters[0] / constants.c) * np.sin(
        parameters[1] * time + parameters[2]) * INCLINATION_FACTOR) *
                     EMITTED_WAVELENGTH)

    return doppler_shift


def chi_squared_func(parameters, array, function):
    y_vals = array[:, 1]
    time = array[:, 0]
    errors = array[:, 2]
    chi = np.sum((np.divide(
        (y_vals - function(parameters, time=time)), errors) ** 2))

    return chi


def reduced_chi_squared_func(chi, parameters, data_points):
    chi_r = chi / (data_points - parameters)

    return chi_r


def data_filterer(array, parameters, iterations, predicted_function):
    """
    Filters data.

    """

    if iterations == 0:  # Performs initial data filtering
        # Removes invalid values
        array = array[~np.isnan(array).any(axis=1), :]

        # Removes anomalous values
        array = np.delete(array, np.where(np.abs(array[:,
                                                 1] - EMITTED_WAVELENGTH)
                                          > iqr(array[:, 1])), 0)

        # Averages errors where the error = 0
        array[np.where(array[:, 2] == 0), 2] = np.average(
            array[np.where(array[:, 2] != 0), 2])

        return array

    z_scores = ((array[:, 1] - predicted_function(parameters, time=array[:, 0])
                 ) / array[:, 2])
    threshold = 3
    outliers = np.abs(z_scores)
    array = np.delete(array, np.where(outliers > threshold), 0)
    print(f"{len(np.where(outliers > threshold))} outlier(s) found in this "
          f"iteration. Removing.")

    return array


def data_point_plotter(array, predicted_function, parameters):
    x = array[:, 0]
    y = array[:, 1]
    error_bars = array[:, 2]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(211)
    ax.errorbar(x, y, yerr=error_bars, fmt="bo")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.plot(
        TIME,
        predicted_function(parameters),
        "r-"
    )
    ax_residuals = fig.add_subplot(414)
    residuals = y - predicted_function(parameters, x)
    ax_residuals.errorbar(x, residuals, yerr=error_bars, fmt="ro")
    ax_residuals.plot(x, 0 * x, color="r")
    ax_residuals.grid(True)
    ax_residuals.set_title("Residuals", fontsize=14)
    plt.show()


def optimiser(chi_squared, reduced_chi, iterations, parameter_guess,
              data_array, data_filter, predicted_function):
    data = data_filterer(data_array, parameter_guess, iterations,
                         predicted_function)
    chi = chi_squared(parameter_guess, data, predicted_function)
    chi_r = reduced_chi(chi, len(parameter_guess), len(data))
    print(f"Iter {iterations}")
    print(f"Reduced chi squared: {chi_r:.3f}\n"
          f"Velocity: {parameter_guess[0]:.3f} m/s\n"
          f"Angular Velocity: {parameter_guess[1]:.3} rad/year\n"
          f"Phase: {parameter_guess[2]:.3f} rad\n")

    # test = fmin(chi_squared,
    #            parameter_guess,
    #            args=(data,),
    #            disp=False,
    #            retall=True,
    #            )

    # for i in test[1]:
    #    print(f"{chi_squared_func(i, data):.3f} for parameters {i}")

    if chi_r > FIT_TOLERANCE:  # Calls function again if fit isn't good enough
        iteration = iterations + 1
        optimised_parameters = fmin(chi_squared,
                                    parameter_guess,
                                    args=(data, predicted_function),
                                    disp=False)
        return optimiser(chi_squared=chi_squared,
                         reduced_chi=reduced_chi,
                         iterations=iteration,
                         parameter_guess=optimised_parameters,
                         data_array=data,
                         data_filter=data_filter,
                         predicted_function=predicted_function)

    print(f"Chi squared value of {chi:.3f} obtained after"
          f" {iterations} "
          f"optimisation iterations.")
    print(f"Reduced chi squared value of {chi_r:.3f} obtained after"
          f" {iterations} "
          f"optimisation iterations.")
    return parameter_guess, data_array


def main() -> None:
    data = data_getter()

    optimised_parameters, data = optimiser(chi_squared_func,
                                           reduced_chi_squared_func,
                                           0,
                                           INIT_PARAMETER_ARRAY,
                                           data,
                                           data_filterer,
                                           wavelength_function)

    data_point_plotter(data, wavelength_function, optimised_parameters)

    # print(f"Reduced chi squared value of {chi_r:.3f} obtained after"
    #      f" {iterations} "
    #      f"optimisation iterations.")


if __name__ == '__main__':
    main()
