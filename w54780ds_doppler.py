# -*- coding: utf-8 -*-
"""
Created on Tue Apr 09 2024
__________________Title___________________
PHYS 10362 - Assignment 2 - Doppler Shifts
------------------------------------------
This calculates the mass of an exoplanet in a binary system using doppler 
spectroscopy. It does this throught the use of chi squared minimisation to fit
a predicted function to the data, while also filtering the data for anomalies
and outliers.

The script filters through a given list of data files, the data from which 
is transferred to a numpy array to be worked with. The fit is optimised by 
running recursively until the goal chi squared value is reached, or the maximum
number of recursions is reached. 


The chi squared minimization is achieved through the use of the scipy 
"minimize" function. Standard error on parameters can be calculated from the 
inverse Hessian matrix, as according to the alogrithm described here:
https://search.r-project.org/CRAN/refmans/HelpersMG/html/SEfromHessian.html
The Hessian matrix can be simply optained through the scipy minimize function.

The program then performs error propogation on values calculated from the 
fitted parameters. The program plots a the data points along with the fitted
curve, while also showing a plot of residuals. The fitted parameters and
calculated values are also displayed on the plot. 
"""

from math import isclose

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import G, c
from scipy.optimize import minimize
from scipy.stats import iqr

# Constants
INCLINATION_ANGLE = 90  # degrees
INCLINATION_FACTOR = np.sin(INCLINATION_ANGLE * (np.pi / 180))
INIT_PARAMETER_ARRAY = [
    50,  # Velocity
    3e-8 * 365 * 24 * 3600,  # rad/s -> rad/year
    3.14  # rad
]
EMITTED_WAVELENGTH = 656.281  # nanometres
STAR_MASS = 2.78 # Solar masses
M_JUP = 1.8981246e+27 # Kilograms
M_SUN = 1.98840987e+30 # Kilgograms
AU = 1.49597871e+11 # Metres

# Data
FILE_LIST = [
    "doppler_data_1.csv",
    "doppler_data_2.csv"
]
DELIMITER = ','
COMMENTS = '%'

# Plot
PLOT_FILE_NAME = 'plot.png'
DPI = 400
X_LABEL = r'$t$ / s'
Y_LABEL = r'$\lambda(t)$ / nm'
LEGEND_LABELS = [
    r"Data points",
    r"$\lambda(t) = \frac{c + v_s(t)}{c}\lambda_0$"
]
PLOT_STYLE = 'default'
DATA_POINT_COLOR = "#f25544"
DATA_POINT_FORMAT = "o"
LINE_COLOR = "#28c989"
LINE_STYLE = "-"

# Program behaviour
SKIP_HEADER = 1
Z_SCORE_THRESHOLD = 3  # Minimum Z score
FIT_TOLERANCE = 1  # Minimum reduced chi squared
MAX_ITER = 50
CHI_TOLERANCE = 0.09
SIGNIFICANT_FIGURES = 4


def data_getter():
    """
    Retrieves data from the file list and converts it to a numpy array.

    Returns
    -------
    array : array like
        Array containing data extracted from file list.

    Raises
    ------
    FileNotFoundError
    SystemExit
    """
    data_array = np.empty((0, 3))
    for file_name in FILE_LIST:
        try:
            data_array = np.vstack((data_array,
                                    np.genfromtxt(file_name,
                                                  dtype=float,
                                                  delimiter=DELIMITER,
                                                  skip_header=SKIP_HEADER,
                                                  comments=COMMENTS)))
            print(f"File {file_name} processed.")
        except FileNotFoundError:
            print(f"File {file_name} not found in local path. Skipping.")

    if len(data_array[:]) == 0:
        raise SystemExit('No data points added. Quitting.')

    print(f"{len(data_array[:])} data points added.")

    return data_array

def initial_data_filtering(data_array):
    """
    Performs initial data filtering by removing extreme outliuers, NaNs, and
    0-error values. 
    
    Extreme outliers are identified by comparing their 
    absolute difference with the interquartile range of the data. This works as
    the "regular" shifted wavelengths fluctuate about the emitted wavelength
    within the interquartile range of the data.

    Parameters
    ----------
    data_array : array like
        Data which is going to be filtered.

    Returns
    -------
    data_array : array like
        Filtered data.

    """
    print("\nInitial data filtering. Removing NaN and extreme outliers.")

    # Removes invalid values
    number_of_nans = data_array[np.where(np.isnan(data_array))].size
    print(f"{number_of_nans} NaN values will be removed.")
    data_array = data_array[~np.isnan(data_array).any(axis=1), :]
    
    # Removes extreme outliers.
    iq = iqr(data_array[:,1], rng=(25, 75))
    print(iq)
    outlier_condition = (np.abs(data_array[:, 1] - EMITTED_WAVELENGTH)
                         > iq)
    number_of_outliers = data_array[np.where(outlier_condition)].size
    print(f"{number_of_outliers} extreme outlier(s) will be removed.")
    data_array = np.delete(data_array, np.where(outlier_condition), 0)
    
    # Removes rows with error = 0
    number_of_zero_error = len(data_array[np.where(data_array[:, 2] == 0)])
    print(f"{number_of_zero_error} row(s) with zero error will be removed.")
    data_array = np.delete(data_array, np.where(data_array[:, 2] == 0), 0)

    return data_array

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
    wavelength = ((1 + (parameters[0] / c) * np.sin(
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


def data_filterer(data_array, parameters, predicted_function, 
                  z_threshold=Z_SCORE_THRESHOLD):
    """
    Performs further data filtering by comparing the z score of each data
    point, using the currently fitte function as a value for the mean. This
    process removes more subtle outliers.

    Parameters
    ----------
    data_array : array like
        Data being filtered.
    parameters : array like
        The fitted parameters.
    predicted_function : function
        Function the data is being fitted to.

    Returns
    -------
    data_array : array like
        Filtered data.

    """
    z_scores = ((data_array[:, 1]
                 - predicted_function(parameters, time=data_array[:, 0]))
                / data_array[:, 2])

    outlier_conditon = np.abs(z_scores) > z_threshold

    print(f"{len(data_array[np.where(outlier_conditon)])} outliers"
          " have been found in this iteration.")
    data_array = np.delete(data_array, np.where(outlier_conditon), 0)

    return data_array


def optimiser(function_list, data_array, hess_matrix=None,
              parameter_guess_list=INIT_PARAMETER_ARRAY):
    """
    Optimises the curve fit by recursively fitting the function to the data
    until a goal chi squared value is met within a given range.

    Parameters
    ----------
    function_list : array like
        Contains functions used by the optimiser.
    data_array : array like
        The data being fitted.
    hess_matrix : array like, optional
        A matrix generated by the minimize function from which the standard
        error on the fitted parameters can be found. The default is None.
    parameter_guess_list : array like, optional
        A list of the fitted paramters. The default is INIT_PARAMETER_ARRAY.

    Returns
    -------
    parameter_guess_list : array like
        List of fully optimised, fitted parameters.
    data_array : array like
        Contains the fitted data which has been further filtered by the
        optimisation process.
    hess_matrix : array like
        Inverse of the Hessian matrix which will be used to find the standard
        error on the fitted parameters.


    """
    print(f"\nIteration {optimiser.counter}.")

    # Functions used in program
    chi_squared = function_list[0]
    data_filter = function_list[1]
    predicted_function = function_list[2]

    # Calculate chi^2 and reduced chi^2 for this iteration
    chi = chi_squared(parameter_guess_list, data_array, predicted_function)
    chi_r = chi / (data_array.shape[0] - len(parameter_guess_list))
    print(f"Chi^2 = {chi:.3f}")
    print(f"Reduced chi^2 = {chi_r:.3f}")

    if optimiser.counter == MAX_ITER:
        print("Maximum number of iterations reached! Giving up. Consider re-"
              "evaluating intial guesses or chi squared tolerance.")
        return parameter_guess_list, data_array, hess_matrix

    # Calls function again if fit isn't good enough.
    if not isclose(chi_r, 1, abs_tol=CHI_TOLERANCE):

        optimised_parameter_list = (minimize(chi_squared,
                                             parameter_guess_list,
                                             args=(data_array,
                                                   predicted_function))).x

        hess_inv_matrix = (minimize(chi_squared, parameter_guess_list,
                               args=(data_array, predicted_function),
                               )).hess_inv

        data = data_filter(data_array, optimised_parameter_list,
                           predicted_function)

        optimiser.counter += 1  # Iterate

        return optimiser(function_list=function_list,
                         parameter_guess_list=optimised_parameter_list,
                         data_array=data,
                         hess_matrix=hess_inv_matrix)

    return parameter_guess_list, data_array, hess_matrix


def significant_figure_rounder(value: float,
                               significant_figures=SIGNIFICANT_FIGURES):
    """
    Rounds a given value to a number of significant figures and returns it as
    a string to preserve zeros.

    Parameters
    ----------
    value : float
        Value being rounded.
    significant_figures : int
        Number of significant figures to which the value is rounded. The
        default is SIGNIFICANT_FIGURES.

    Returns
    -------
    rounded_value : string
        The rounded value.

    """
    before_decimal = str(float(value))[::1].find('.')

    if str(float(value))[0] == '0':
        rounded_value = f'{value:.{significant_figures}g}'

        return rounded_value

    if before_decimal > significant_figures:
        rounded_value = f'{value:.{significant_figures}g}'
        return rounded_value

    decimal_places = significant_figures - before_decimal
    rounded_value = f'{value:.{decimal_places}f}'

    return rounded_value


def decimal_place_counter(value):
    """
    Counts the number of digits after the decimal point in a floating point
    number. The argument must be a string or have must be able to be converted
    to a string.

    Parameters
    ----------
    value : string
        The value whose digits are being counted.

    Returns
    -------
    decimal_places : int
        The number of digits after the decimal place.

    """
    decimal_places = str(value)[::-1].find('.')

    return decimal_places


def error_propogator(value_list, error_list,
                     function_output: float, constant=1.0, power=1.0):
    """
    Performs standard error propagation. Only considers cases where there is
    multiplication between values, a value is raised to a power, and/or is
    raised to a power, as the formulas used in this problem only perform
    those operations.

    Parameters
    ----------
    value_list : list
        A list of values with errors on them.
    error_list : list
        A list of errors corresponding to each value.
    function_output : float, optional
        Value of the function for which the error is being propagated.
    constant : float, optional
        Constant values without error which the values are being multiplied by.
        The default is 1.0.
    power : float, optional
        Power to which the values are being raised to. The default is 1.0.

    Returns
    -------
    final_error : float
        The final propagated error.

    Raises
    ------
    SystemExit

    """
    value_list = np.array(value_list)
    error_list = np.array(error_list)

    if value_list.size != error_list.size:
        raise SystemExit("Fatal error! Mismatched error and value size.")

    combined = np.sqrt(np.sum((error_list / value_list) ** 2))

    final_error = constant * power * combined * function_output

    return final_error


def data_point_plotter(data_array, predicted_function, parameters, time,
                       display_values) -> None:
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
    display_values : numpy array
        Contains strings of values which will be printed on the plot.

    Returns
    -------
    None.


    """
    x_values = data_array[:, 0]
    y_values = data_array[:, 1]
    error_bars = data_array[:, 2]

    # Main plot
    fig = plt.figure(figsize=(8, 8))
    plt.style.use(PLOT_STYLE)

    axes = fig.add_subplot(211)
    axes.errorbar(x_values,
                  y_values,
                  yerr=error_bars,
                  color=DATA_POINT_COLOR,
                  fmt=DATA_POINT_FORMAT,
                  label=LEGEND_LABELS[0]
                  )
    axes.set_xlabel(X_LABEL)
    axes.set_ylabel(Y_LABEL)
    axes.grid(True)
    axes.plot(time,
              predicted_function(parameters, time),
              color=LINE_COLOR,
              linestyle=LINE_STYLE,
              label=LEGEND_LABELS[1]
              )

    chi = display_values[5]
    degrees_of_freedom = len(data_array - len(parameters))
    reduced_chi = display_values[6]

    velocity = display_values[0]
    angular_velocity = display_values[1]
    phase = display_values[2]
    radius = display_values[3]
    planet_mass = display_values[4]

    velocity_error = display_values[7]
    angular_velocity_error = display_values[8]
    phase_error = display_values[9]
    radius_error = display_values[10]
    planet_mass_error = display_values[11]

    axes.annotate(('Fit data'),
                  (1, 0), (-40, -35),
                  xycoords='axes fraction', textcoords='offset points',
                  va='top', fontsize='10')
    axes.annotate((rf'$\chi^2 = {chi}$'),
                  (1, 0), (-60, -55),
                  xycoords='axes fraction', textcoords='offset points',
                  va='top', fontsize='10')
    axes.annotate((rf'Degrees of Freedom = ${degrees_of_freedom}$'),
                  (1,0), (-132, -75),
                  xycoords='axes fraction', textcoords='offset points',
                  va='top', fontsize='10')
    axes.annotate((rf'Reduced $\chi^2 = {reduced_chi}$'),
                  (1, 0), (-104, -95), #
                  xycoords='axes fraction', textcoords='offset points',
                  va='top', fontsize='10')

    axes.annotate(('Fitted function'),
                  (0, 0), (-60, -35),
                  xycoords='axes fraction', textcoords='offset points',
                  va='top', fontsize='10')
    axes.annotate((rf'$v_s(t) = v_0\sin(\omega t '
                   rf'+ \phi)\sin({INCLINATION_ANGLE})$'),
                  (0, 0), (-60, -55),
                  xycoords='axes fraction',  textcoords='offset points',
                  va='top', fontsize='10')
    axes.annotate((rf'$v_0 = ({velocity} \pm {velocity_error})'
                   r'\mathrm{m s}^{-1}$'),
                  (0, 0),
                  (-60, -75),
                  xycoords='axes fraction', va='top',
                  textcoords='offset points', fontsize='10')
    axes.annotate((rf'$\omega = ({angular_velocity} \pm'
                   rf'{angular_velocity_error}$)'
                   r'rad year$^{-1}$'),
                  (0, 0), (-60, -95),
                  xycoords='axes fraction', textcoords='offset points',
                  va='top', fontsize='10')
    axes.annotate((rf'$\phi = ({phase} \pm'
                   rf'{phase_error})$rad'),
                  (0, 0), (-60, -115),
                  xycoords='axes fraction', textcoords='offset points',
                  va='top', fontsize='10')

    axes.annotate(('Calculated values'),
                  (0.5, 0), (-60, -35),
                  xycoords='axes fraction', textcoords='offset points',
                  va='top', fontsize='10')
    axes.annotate((rf'$r = ({radius} \pm {radius_error})$AU'),
                  (0.5,0), (-60,-55),
                  xycoords='axes fraction', textcoords='offset points',
                  va='top', fontsize='10')
    axes.annotate((rf'$m_p = ({planet_mass} \pm {planet_mass_error})m_J$'),
                  (0.5,0), (-60,-75),
                  xycoords='axes fraction', textcoords='offset points',
                  va='top', fontsize='10')


    # Residuals
    residuals_axes = fig.add_subplot(414)
    residuals = y_values - predicted_function(parameters, x_values)
    residuals_axes.errorbar(x_values,
                            residuals,
                            yerr=error_bars,
                            color=DATA_POINT_COLOR,
                            fmt=DATA_POINT_FORMAT)
    residuals_axes.plot(x_values,
                        0 * x_values,
                        color=LINE_COLOR,
                        linestyle=LINE_STYLE)
    residuals_axes.grid(True)
    residuals_axes.set_title("Residuals", fontsize=14)
    residuals_axes.set_xlabel(X_LABEL)
    residuals_axes.set_ylabel("Standardised Residual")
    
    

    axes.legend()
    print(f"Saving figure as {PLOT_FILE_NAME} in local folder.")
    plt.savefig(PLOT_FILE_NAME, dpi=DPI)
    plt.show()


def main() -> None:
    """
    Main function. Calls all other functions and prints main output.

    Returns
    -------
    None
    """
    # Gets data
    data = data_getter()
    data = initial_data_filtering(data)

    optimiser.counter = 1  # initialise counter

    # Optimises data and fit
    optimised_parameters_list, data, hessian_matrix = optimiser(
        function_list=[chi_squared_func, data_filterer, wavelength_function],
        data_array=data)
    
    # Fit information
    chi = np.round(chi_squared_func(optimised_parameters_list,
                                    data, wavelength_function), decimals=3)
    reduced_chi = np.round(chi / (len(data) - len(optimised_parameters_list)),
                           decimals=3)

    # Standard errors on parameters found as described in the header.
    error_on_parameters = np.sqrt(np.diag(hessian_matrix))
    
    # Fitted parameters
    velocity = optimised_parameters_list[0]  # m/s
    angular = optimised_parameters_list[1]  # rad / year
    phase = optimised_parameters_list[2]  # rad

    # Errors on fitted parameters
    error_v = error_on_parameters[0]
    error_w = error_on_parameters[1]
    error_p = error_on_parameters[2]

    # Calculated values
    period = 2 * np.pi / angular
    period_error = error_propogator([angular], [error_w], period)

    radius = np.cbrt(period ** 2)
    radius_error = error_propogator([period], [period_error], radius, (3 / 2))

    # Not displayed, only needed for further calculation.
    planet_velocity = np.sqrt((G * STAR_MASS * M_SUN) / (radius * AU))
    planet_v_error = error_propogator([radius], [radius_error],
                                      planet_velocity, power=(1 / 2),
                                      constant=np.sqrt(G * STAR_MASS))


    planet_mass =((STAR_MASS * M_SUN * velocity) / planet_velocity) / M_JUP
    planet_mass_error = error_propogator([planet_velocity, velocity],
                                         [planet_v_error, error_v],
                                         planet_mass, constant=STAR_MASS)

    radius_error = np.round(radius_error, decimal_place_counter(radius))
    planet_mass_error = np.round(planet_mass_error,
                                 decimal_place_counter(planet_mass))

    display_list = np.empty(0)

    # Rounds all values to appropriate number of significant figures. Stored
    # as strings to preserve end zeros.
    velocity = significant_figure_rounder(velocity)
    angular = significant_figure_rounder(angular)
    phase = significant_figure_rounder(phase)
    planet_mass = significant_figure_rounder(planet_mass)
    radius = significant_figure_rounder(radius)

    # Creates a list of all displayed values to be passed through to the
    # plotting function
    display_list = np.append(display_list,
                             (velocity, angular, phase, radius, planet_mass,
                              f'{chi:.3f}', f'{reduced_chi:.3f}',
                              f'{error_v:.{decimal_place_counter(velocity)}f}',
                              f'{error_w:.{decimal_place_counter(angular)}f}',
                              f'{error_p:.{decimal_place_counter(phase)}f}',
                              f'{radius_error:.{decimal_place_counter(radius)}f}',
                              f'{planet_mass_error:.{decimal_place_counter(planet_mass)}f}'))

    print(f"\nBelow values found after {optimiser.counter} iterations.\n"
          f""
          f"Chi^2 = {chi:.3f}\n"
          f""
          f"Reduced Chi^2 = {reduced_chi:.3f}"
          f"\nVelocity: ({velocity} +/- "
          f"{error_v:.{decimal_place_counter(velocity)}f}) m/s\n"
          f""
          f"Angular velocity: ({angular} +/- "
          f"{error_w:.{decimal_place_counter(angular)}f}) rad/year\n"
          f""
          f"Phase: ({phase} +/- "
          f"{error_p:.{decimal_place_counter(phase)}f}) rad\n"
          f"Orbital radius=({radius} + /- "
          f"{radius_error:.{decimal_place_counter(radius)}f}) AU\n"
          f""
          f"Planet mass=({planet_mass} + /- "
          f"{planet_mass_error:.{decimal_place_counter(planet_mass)}f}) M_jup")

    
    continuous_time = np.linspace(np.floor(data[0, 0]), np.ceil(data[-1, 0]),
                                  num=data.size)

    data_point_plotter(data, wavelength_function,
                       optimised_parameters_list, continuous_time,
                       display_list)
    

if __name__ == '__main__':
    main()
