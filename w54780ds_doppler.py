# -*- coding: utf-8 -*-
"""
Created on Tue Apr 09 2024
______Doppler Shifts______
TODO: Calculate m_p and r and extrapolate their uncertainties
TODO: Put calculated values on plot
TODO: Make plot look nice
TODO: Make code readable
TODO: Add more validation checks
TODO: Make a guesser function to make intial estimates of fitted parameters
"""

from math import isclose

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy import constants as const
from scipy import constants as sciconst
from scipy.optimize import fmin
from scipy.stats import iqr

# Constants
SIGMA_TO_CHI_DIFFERENCE = {
    1: 1,
    2: 2.71
}
SIGMA_TO_TOLERANCE = {
    1: 0.15,
    2: 0.25
}
INCLINATION_ANGLE = 90  # degrees
INCLINATION_FACTOR = 1 #np.sin(INCLINATION_ANGLE * (np.pi/180))
INIT_PARAMETER_ARRAY = [[
    50,  # Velocity
    3e-8 * 365 * 24 * 3600,  # rad/s -> rad/year
    3.14  # rad
]]
EMITTED_WAVELENGTH = 656.281  # nanometres
STAR_MASS = 2.78

# Data
FILE_LIST = [
    "doppler_data_1.csv",
    "doppler_data_2.csv"
]
DELIMITER = ','
SKIP_HEADER = 1
Z_SCORE_THRESHOLD = 3  # Minimum Z score
FIT_TOLERANCE = 1  # Minimum reduced chi squared

# Plot
PLOT_FILE_NAME = 'plot.png'
DPI = 400
X_LABEL = r'$t$ / s'
Y_LABEL = r'$\lambda(t)$ / nm'
LEGEND_LABELS = [
    r"Data points",
    r"$\lambda(t) = \frac{c + v_s(t)}{c}\lambda_0$"
]
DATA_POINT_FORMAT = "bo"
LINE_COLOR = "r"
LINE_STYLE = "-"


def data_getter():
    """
    Retrieves data from the file list and converts it to a numpy array.

    Returns
    -------
    array : numpy array
        Array containing data extracted from file list.

    Raises
    ------
    FileNotFoundError
    """
    array = np.empty((0, 3))
    for file_name in FILE_LIST:
        try:
            array = np.vstack((array,
                               np.genfromtxt(file_name,
                                             dtype=float,
                                             delimiter=DELIMITER,
                                             skip_header=SKIP_HEADER,
                                             comments='%')))
            print(f"File {file_name} processed.")
        except FileNotFoundError:
            print(f"File {file_name} not found in local path. Skipping.")

    print(f"{len(array[:])} data points added.")
    return array


def wavelength_function(parameters: list, time):
    """
    Calculates the value of the wavelength function which the data will be
    mapped to.

    Parameters
    ----------
    parameters : list
        A list of parameter values.
    time : numpy array
        Numpy array of time values for which values of the function are
        calculated.

    Returns
    -------
    wavelength : numpy array
        A numpy array of wavelength values corresponding to each time value in
        the 'time' parameter.

    """
    wavelength = ((1 + (parameters[0] / sciconst.c) * np.sin(
        parameters[1] * time + parameters[2]) * INCLINATION_FACTOR) *
        EMITTED_WAVELENGTH)

    return wavelength


def chi_squared_func(parameters, data_array, function):
    """
    Function to calculate the chi squared value for a set of data points being
    mapped to a function.

    Parameters
    ----------
    parameters : list
        List of function parameters.
    data_array : numpy array
        Numpy array of all data points.
    function : function
        The function which is compared to teh data.

    Returns
    -------
    chi_squared : float
        The calculated chi squared value.

    """
    y_vals = data_array[:, 1]
    time = data_array[:, 0]
    errors = data_array[:, 2]
    chi_squared = np.sum(np.divide((y_vals - function(parameters, time)),
                                   errors) ** 2)

    return chi_squared


def data_filterer(data_array, parameters, predicted_function):
    """
    Function to perform data filtering. It will fit data differently intially
    as there is no function to compare the data to. Instead, the difference
    between each data point and the emitted wavelength is taken and then
    compared with the interquartile range. This removes any extreme outliers.
    NaN data points and data points with zero error are also removed.

    In all other iterations, the function calculates the z score of each data
    point, with the value of the fitted function at that point being used as 
    the mean.

    Parameters
    ----------
    data_array : numpy array
        Array of the data currently being worked with.
    parameters : list
        List of function parameters.
    predicted_function : function
        Function to which the data is being mapped to.

    Returns
    -------
    data_array : numpy array
        The filtered numpy array.

    """
    iterations = optimiser.counter

    if iterations == 0:  # Performs initial data filtering
        print("\nInitial data filtering. Removing NaN and extreme outliers.")
        # Removes invalid values
        number_of_nans = data_array[np.where(np.isnan(data_array))].size
        print(f"{number_of_nans} NaN values will be removed.")
        data_array = data_array[~np.isnan(data_array).any(axis=1), :]

        # Removes extreme outliers.
        outlier_condition = (np.abs(data_array[:, 1] - EMITTED_WAVELENGTH)
                             > iqr(data_array[:, 1]))
        number_of_outliers = data_array[np.where(outlier_condition)].size

        print(f"{number_of_outliers} extreme outlier(s) will be removed.")

        data_array = np.delete(data_array, np.where(outlier_condition), 0)

        # Averages errors where the error = 0
        number_of_zero_error = len(data_array[np.where(data_array[:, 2] == 0)])
        print(f"{number_of_zero_error} value(s) with zero error will be "
              "removed.")
        data_array = np.delete(
            data_array, np.where(data_array[:, 2] == 0), 0)

        return data_array

    print(f"\nIteration {iterations}.")
    z_scores = ((data_array[:, 1]
                 - predicted_function(parameters, time=data_array[:, 0]))
                / data_array[:, 2])
    outliers = np.abs(z_scores)
    print(f"{len(data_array[np.where(outliers > Z_SCORE_THRESHOLD)])} outliers"
          " have been found in this iteration.")
    data_array = np.delete(data_array, np.where(outliers
                                                > Z_SCORE_THRESHOLD), 0)

    return data_array


def data_point_plotter(data_array, predicted_function, parameters, time):
    """
    Function to plot the data and the curve which was fitted to the data.

    Parameters
    ----------
    data_array : numpy array
        Array of the data to which the curve was fitted.
    predicted_function : function
        The curve which was fitted to the data.
    parameters : list
        List of function parameters.
    time : numpy array
        A numpy array containing the times for which the curve will be plotted.

    Returns
    -------
    None.

    """
    x_values = data_array[:, 0]
    y_values = data_array[:, 1]
    error_bars = data_array[:, 2]

    # Main plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(211)
    ax.errorbar(x_values,
                y_values,
                yerr=error_bars,
                fmt=DATA_POINT_FORMAT,
                label=LEGEND_LABELS[0]
                )
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True)
    ax.plot(time,
            predicted_function(parameters, time),
            color=LINE_COLOR,
            linestyle=LINE_STYLE,
            label=LEGEND_LABELS[1]
            )

    # Residuals
    ax_residuals = fig.add_subplot(414)
    residuals = y_values - predicted_function(parameters, x_values)
    ax_residuals.errorbar(x_values,
                          residuals,
                          yerr=error_bars,
                          fmt=DATA_POINT_FORMAT)
    ax_residuals.plot(x_values,
                      0 * x_values,
                      color=LINE_COLOR,
                      linestyle=LINE_STYLE)
    ax_residuals.grid(True)
    ax_residuals.set_title("Residuals", fontsize=14)

    ax.legend()
    print(f"Saving figure as {PLOT_FILE_NAME} in local folder.")
    plt.savefig(PLOT_FILE_NAME, dpi=DPI)
    plt.show()


def deviated_parameter_gen(parameter_list, function_list, data_array, sigma):
    """
    Generates a list of parameters which produce a chi squared value which 
    corresponds to a given deviation from the minimum chi squared value found.
    It finds a list of values as it is not likely that a chi squared difference

    Parameters
    ----------
    parameter_list : TYPE
        DESCRIPTION.
    function_list : TYPE
        DESCRIPTION.
    data_array : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.

    Returns
    -------
    deviated_parameters : TYPE
        DESCRIPTION.

    """
    chi_squared = function_list[0]
    predicted_function = function_list[1]
    optimised_parameters = parameter_list[0]

    chi_difference_goal = SIGMA_TO_CHI_DIFFERENCE[sigma]
    chi_tolerance = SIGMA_TO_TOLERANCE[sigma]

    temp_parameter_list = np.empty((0, 3))
    deviated_parameters = np.empty((0, 3))

    for i in parameter_list[-1]:
        chi_difference = np.abs(chi_squared(i, data_array,
                                            predicted_function) -
                                chi_squared(optimised_parameters, data_array,
                                            predicted_function))

        if isclose(chi_difference, chi_difference_goal, abs_tol=chi_tolerance):
            deviated_parameters = np.vstack(
                (deviated_parameters, i, temp_parameter_list))

        temp_parameter_list = i

    return deviated_parameters


def error_list_gen(deviated_parameter_list, optimised_parameters, sigma):
    error_list = np.empty(0)

    if deviated_parameter_list.size == 0:
        return error_list

    for parameters in deviated_parameter_list.T:
        temp_average = np.average(np.abs(parameters))
        error_list = np.append(error_list, temp_average)

    error_list = np.abs(error_list - optimised_parameters) / sigma

    return error_list


def error_list_combiner(sigma_1_error_list, sigma_2_error_list):
    error_list = np.empty((0, 1))

    sigma_1_empty = sigma_1_error_list.size == 0
    sigma_2_empty = sigma_2_error_list.size == 0

    if sigma_1_empty and sigma_2_empty:
        print("No useful parameters found during error calculation! "
              "Re-evaluate initial guess.")
        return np.zeros(3)

    if sigma_1_empty:
        return sigma_2_error_list

    if sigma_2_empty:
        return sigma_1_error_list

    for i, j in zip(sigma_1_error_list, sigma_2_error_list):
        average = np.average([i, j])
        error_list = np.append(error_list, average)

    return error_list


def optimiser(function_list, parameter_guess_list, data_array):
    """
    Fits the curve recursively until the fit tolerance is met, while at the 
    same time filtering the data of any outliers by calling the data filtering
    function, and obtaining errors on the fit parameters by calling the error
    calculator function. Each time the program is called, it increments its
    own counter attribute.

    Parameters
    ----------
    function_list : list
        List of functions available to for use by the function.
    parameter_guess_list : numpy array
        A list of lists of function parameters. The initial guess parameters
        are passed through first.
    data_array : numpy array
        Numpy array containing data currently being worked on.

    Returns
    -------
    optimised_parameters : numpy array
        Numpy array containing the fully optimised, fitted parameters. These
        have also been rounded to 4 significant figures.
    data_array : numpy array
        Numpy array containing fully filtered data.
    error_list : numpy array
        Numpy array containing the errors on the parameters.

    """
    # Functions used in program
    chi_squared = function_list[0]
    data_filter = function_list[1]
    predicted_function = function_list[2]

    # Filter passed data
    data = data_filter(data_array, parameter_guess_list[0], predicted_function)

    # Calculate chi^2 and reduced chi^2 for this iteration
    chi = chi_squared(parameter_guess_list[0], data, predicted_function)
    print(data.shape[0])
    chi_r = chi / (data.shape[0] - len(parameter_guess_list[0]))
    if chi_r > FIT_TOLERANCE:  # Calls function again if fit isn't good enough
        optimiser.counter += 1  # Iterate

        optimised_parameter_list = fmin(chi_squared,
                                        parameter_guess_list[0],
                                        args=(data, predicted_function),
                                        disp=False,
                                        retall=True)  # Outputs every iteration

        return optimiser(function_list=function_list,
                         parameter_guess_list=optimised_parameter_list,
                         data_array=data)

    return parameter_guess_list, data_array, chi, chi_r


def significant_figure_rounder(value: float, significant_figures: int):
    rounded_value = float('{:.{p}g}'.format(value, p=significant_figures))

    return rounded_value


def decimal_place_counter(value: float):
    """
    Counts the number of digits after the decimal point in a floating point 
    number.

    Parameters
    ----------
    value : float
        The value whose digits are being counted.

    Returns
    -------
    decimal_places : int
        The number of digits after the decimal place.

    """
    decimal_places = str(float(value))[::-1].find('.')
    return decimal_places


def period_to_radius(period):
    r = np.cbrt(period ** 2) # AU

    return r


def radius_to_planet_velocity(radius):
    planet_velocity = np.sqrt((sciconst.G * STAR_MASS * const.GM_sun) / (
        radius))

    return planet_velocity


def planet_mass_calculator(star_velocity, planet_velocity):
    planet_mass = (STAR_MASS * star_velocity)/planet_velocity

    return planet_mass


def main() -> None:
    """
    Main function. Calls all other functions and prints main output.

    Returns
    -------
    None
    """
    data = data_getter()

    optimiser.counter = 0  # initialise counter

    optimised_parameters_list, data, chi, chi_r = optimiser(
        function_list=[chi_squared_func, data_filterer, wavelength_function],
        parameter_guess_list=INIT_PARAMETER_ARRAY,
        data_array=data)

    optimised_parameters = optimised_parameters_list[0]

    deviated_parameter_list_1 = deviated_parameter_gen(
        optimised_parameters_list,
        [chi_squared_func, wavelength_function], data, 1)

    deviated_parameter_list_2 = deviated_parameter_gen(
        optimised_parameters_list,
        [chi_squared_func, wavelength_function], data, 2)

    error_list_1 = error_list_gen(deviated_parameter_list_1,
                                  optimised_parameters, 1)
    error_list_2 = error_list_gen(deviated_parameter_list_2,
                                  optimised_parameters, 2)

    error_on_parameters = error_list_combiner(error_list_1, error_list_2)

    for i in range(optimised_parameters.size):
        optimised_parameters[i] = significant_figure_rounder(
            optimised_parameters[i], 4)
        decimal_places = decimal_place_counter(optimised_parameters[i])
        error_on_parameters[i] = np.round(error_on_parameters[i],
                                          decimal_places)

    velocity = optimised_parameters[0]
    angular = optimised_parameters[1]
    phase = optimised_parameters[2]
    error_v = error_on_parameters[0]
    error_w = error_on_parameters[1]
    error_p = error_on_parameters[2]

    print(f"Velocity: ({velocity} +/- {error_v}) m/s\n"
          f"Angular velocity: ({angular} +/- {error_w}) rad/year\n"
          f"Phase: ({phase} +/- {error_p}) rad\n"
          f"Chi^2 = {chi:.3f}\n"
          f"Reduced Chi^2 = {chi_r:.3f}")

    period = 2*np.pi / angular
    radius = period_to_radius(period)
    planet_velocity = radius_to_planet_velocity(radius)
    planet_mass = planet_mass_calculator(velocity, planet_velocity)

    print(f"Orbital radius = {radius}\n"
          f"Planet velocity = {planet_velocity}m/s\n"
          f"Planet mass = {planet_mass}M_s")


    continuous_time = np.linspace(np.floor(data[0, 0]), np.ceil(data[-1, 0]),
                                  len(data) * 1000)

    data_point_plotter(data, wavelength_function,
                       optimised_parameters, continuous_time)


if __name__ == '__main__':
    main()
