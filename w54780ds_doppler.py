# -*- coding: utf-8 -*-
"""
Created on Tue Apr 09 2024
__________________Title___________________
PHYS 10362 - Assignment 2 - Doppler Shifts
------------------------------------------
This calculates the mass of an exoplanet in a binary system using doppler
spectroscopy. It does this through the use of chi squared minimisation to fit
a predicted function to the data, while also filtering the data for anomalies
and outliers.

The script filters through a given list of data files, the data from which
is transferred to a numpy array to be worked with. Once any invalid rows and
extreme outliers are removed, program guesses initial parameters by cursory
analysis of the data. The fit is optimised by running recursively until the
goal chi squared value is reached, or the maximum number of recursions is
reached.

The scipy "minimize" function performs chi squared minimisation. Standard
error on parameters can be calculated from the inverse Hessian matrix,
as according to the algorithm described here:
https://search.r-project.org/CRAN/refmans/HelpersMG/html/SEfromHessian.html
The Hessian matrix can be simply obtained through the scipy minimize function.

The program then performs error propagation on values calculated from the
fitted parameters. The program plots the data points along with the fitted
curve, while also showing a plot of residuals. The fitted parameters and
calculated values are also displayed on the plot.

The most time-intensive aspect of this code is saving and showing the generated
figure.

Last Updated: 23/04/2024
@author: Dominik Szablonski, UID: 11310146
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import G, c
from scipy.optimize import minimize
from scipy.stats import iqr

# Constants
INCLINATION_ANGLE = 90  # degrees
INCLINATION_FACTOR = np.sin(INCLINATION_ANGLE * (np.pi / 180))
EMITTED_WAVELENGTH = 656.281  # nanometres
STAR_MASS = 2.78  # Solar masses
M_JUP = 1.8981246e+27  # Kilograms
M_SUN = 1.98840987e+30  # Kilograms
AU = 1.49597871e+11  # Metres

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
X_LABEL = r'$t$ / years'
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
Z_SCORE_THRESHOLD = 3
FIT_TOLERANCE = 1
MAX_ITER = 50
CHI_TOLERANCE = 0.1
SIGNIFICANT_FIGURES = 4


def data_getter():
    """
    Retrieves data from the file list and converts it to a numpy array.

    Returns
    -------
    array : array_like
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
                                                  comments=COMMENTS,
                                                  autostrip=True
                                                  )))
            print(f"File {file_name} processed.")
        except FileNotFoundError:
            print(f"File {file_name} not found in local path. Skipping.")

    if len(data_array[:]) == 0:
        raise SystemExit('No data points added. Quitting.')

    print(f"{len(data_array[:])} data points added.")

    return data_array


def initial_data_filtering(data_array):
    """
    Performs initial data filtering by removing extreme outliers, NaNs, and
    0-error values.

    Extreme outliers are identified by comparing their absolute difference
    with the range of the 15th and 85th percentile of the data. This works
    as the "regular" shifted wavelengths fluctuate about the emitted wavelength
    within this range.

    Initial data filtering is required as initial guesses

    Parameters
    ----------
    data_array : array_like
        Data which is going to be filtered.

    Returns
    -------
    data_array : array_like
        Filtered data.

    """
    print("\nInitial data filtering.")

    # Removes invalid values
    number_of_nans = data_array[np.where(np.isnan(data_array))].size
    print(f"{number_of_nans} NaN values will be removed.")
    data_array = data_array[~np.isnan(data_array).any(axis=1), :]

    # Removes rows with error = 0
    number_of_zero_error = len(data_array[np.where(data_array[:, 2] == 0)])
    print(f"{number_of_zero_error} row(s) with zero error will be removed.")
    data_array = np.delete(data_array, np.where(data_array[:, 2] == 0), 0)

    # Removes extreme outliers.
    per_range = iqr(data_array[:, 1], rng=(15, 85))
    outlier_condition = (np.abs(data_array[:, 1] - EMITTED_WAVELENGTH) >
                         per_range)
    number_of_outliers = data_array[np.where(outlier_condition)].size
    print(f"{number_of_outliers} extreme outlier(s) will be removed.")
    data_array = np.delete(data_array, np.where(outlier_condition), 0)

    return data_array


def wavelength_function(parameters: list, time):
    """
    Calculates the value of the wavelength function which the data will be
    mapped to.

    Parameters
    ----------
    parameters : array_like
        Parameter values.
    time : array_like
        Time values for which values of the function are calculated.

    Returns
    -------
    wavelength : array_like
        Wavelength values corresponding to each time value in the 'time'
        parameter.

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
    parameters : array_like
        Function parameters.
    data_array : array_like
        Data points being considered.
    function : function
        The function which is compared to the data.

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
    point, using the currently fitted function as a value for the mean. This
    process removes more subtle outliers.

    Parameters
    ----------
    data_array : array_like
        Data being filtered.
    parameters : array_like
        The fitted parameters.
    predicted_function : function
        The curve data is being fitted to.
    z_threshold : float, optional
        The Z score threshold for which outliers are identified. The default is
        Z_SCORE_THRESHOLD

    Returns
    -------
    data_array : array_like
        Filtered data.

    """
    z_scores = ((data_array[:, 1]
                 - predicted_function(parameters, time=data_array[:, 0]))
                / data_array[:, 2])

    outlier_condition = np.abs(z_scores) > z_threshold

    print(f"{len(data_array[np.where(outlier_condition)])} outliers"
          " have been found in this iteration.")
    data_array = np.delete(data_array, np.where(outlier_condition), 0)

    return data_array


def optimiser(function_list, data_array, hess_matrix=None,
              parameter_guess_list=None):
    """
    Optimises the curve fit by recursively fitting the function to the data
    until a goal chi squared value is met within a given range.

    Parameters
    ----------
    function_list : array_like
        Contains functions used by the optimiser.
    data_array : array_like
        The data being fitted.
    hess_matrix : array_like, optional
        A matrix generated by the minimize function from which the standard
        error on the fitted parameters can be found. The default is None.
    parameter_guess_list : array_like, optional
        A list of the fitted parameters. The default is None.

    Returns
    -------
    parameter_guess_list : array_like
        List of fully optimised, fitted parameters.
    data_array : array_like
        Contains the fitted data which has been further filtered by the
        optimisation process.
    hess_matrix : array_like
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
              "evaluating initial guesses or chi squared tolerance.")
        return parameter_guess_list, data_array, hess_matrix

    # Calls function again if fit isn't good enough.
    if not np.isclose(chi_r, 1, atol=CHI_TOLERANCE) and chi_r > 1:
        minimisation_output = minimize(chi_squared, parameter_guess_list,
                                       args=(data_array, predicted_function))

        optimised_parameter_list = minimisation_output.x

        hess_inv_matrix = minimisation_output.hess_inv

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
        rounded_value = f'{value:{significant_figures}g}'
        return rounded_value

    decimal_places = significant_figures - before_decimal
    rounded_value = f'{value:.{decimal_places}f}'

    return rounded_value


def propagator(value_list, error_list,
               function_output: float, constant=1.0, power=1.0):
    """
    Performs standard error propagation. Only considers cases where there is
    multiplication between values, a value is raised to a power, and/or is
    raised to a power, as the formulas used in this problem only perform
    those operations.

    Parameters
    ----------
    value_list : array_like
        A list of values with errors on them.
    error_list : array_like
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


def data_point_plotter(data_array, predicted_function, parameters,
                       display_values) -> None:
    """
    Function to plot the data and the curve which was fitted to the data.

    Parameters
    ----------
    data_array : numpy array
        Array of the data to which the curve was fitted.
    predicted_function : function
        The curve which was fitted to the data.
    parameters : array_like
        List of function parameters.
    display_values : numpy array
        Contains strings of values which will be printed on the plot.

    Returns
    -------
    None.


    """
    x_values = data_array[:, 0]
    y_values = data_array[:, 1]
    error_bars = data_array[:, 2]

    time = np.linspace(np.min(x_values), np.max(x_values), data_array.size)
    
    # Main plot
    fig = plt.figure(figsize=(8, 8))
    plt.style.use(PLOT_STYLE)

    main_axes = fig.add_subplot(211)
    main_axes.errorbar(x_values,
                       y_values,
                       yerr=error_bars,
                       color=DATA_POINT_COLOR,
                       fmt=DATA_POINT_FORMAT,
                       label=LEGEND_LABELS[0]
                       )
    main_axes.set_xlabel(X_LABEL)
    main_axes.set_ylabel(Y_LABEL)
    main_axes.grid(True)
    main_axes.plot(time,
                   predicted_function(parameters, time),
                   color=LINE_COLOR,
                   linestyle=LINE_STYLE,
                   label=LEGEND_LABELS[1]
                   )

    degrees_of_freedom = data_array.shape[0] - parameters.size

    main_axes.annotate('Fit data',
                       (1, 0), (-40, -35),
                       xycoords='axes fraction', textcoords='offset points',
                       va='top', fontsize='10')
    main_axes.annotate(rf'$\chi^2 = {display_values[0][0]}$',
                       (1, 0), (-60, -55),
                       xycoords='axes fraction', textcoords='offset points',
                       va='top', fontsize='10')
    main_axes.annotate(rf'Degrees of Freedom = ${degrees_of_freedom}$',
                       (1, 0), (-132, -75),
                       xycoords='axes fraction', textcoords='offset points',
                       va='top', fontsize='10')
    main_axes.annotate(rf'Reduced $\chi^2 = {display_values[0][1]}$',
                       (1, 0), (-104, -95),  #
                       xycoords='axes fraction', textcoords='offset points',
                       va='top', fontsize='10')

    main_axes.annotate('Fitted parameters',
                       (0, 0), (-60, -35),
                       xycoords='axes fraction', textcoords='offset points',
                       va='top', fontsize='10')
    main_axes.annotate((rf'$v_s(t) = v_0\sin(\omega t '
                        rf'+ \phi)\sin({INCLINATION_ANGLE})$'),
                       (0, 0), (-60, -55),
                       xycoords='axes fraction', textcoords='offset points',
                       va='top', fontsize='10')
    main_axes.annotate((rf'$v_0 = ({display_values[1][0]} \pm'
                        rf'{display_values[1][1]})'
                        r'\mathrm{m s}^{-1}$'),
                       (0, 0), (-60, -75),
                       xycoords='axes fraction', va='top',
                       textcoords='offset points', fontsize='10')
    main_axes.annotate((rf'$\omega = ({display_values[2][0]} \pm'
                        rf'{display_values[2][1]}$)'
                        r'rad year$^{-1}$'),
                       (0, 0), (-60, -95),
                       xycoords='axes fraction', textcoords='offset points',
                       va='top', fontsize='10')
    main_axes.annotate((rf'$\phi = ({display_values[3][0]} \pm'
                        rf'{display_values[3][1]})$rad'),
                       (0, 0), (-60, -115),
                       xycoords='axes fraction', textcoords='offset points',
                       va='top', fontsize='10')

    main_axes.annotate('Calculated values',
                       (0.5, 0), (-60, -35),
                       xycoords='axes fraction', textcoords='offset points',
                       va='top', fontsize='10')
    main_axes.annotate(rf'$r = ({display_values[4][0]} \pm'
                       rf'{display_values[4][1]})$AU',
                       (0.5, 0), (-60, -55),
                       xycoords='axes fraction', textcoords='offset points',
                       va='top', fontsize='10')
    main_axes.annotate(rf'$m_p = ({display_values[5][0]} \pm'
                       rf'{display_values[5][1]})m_J$',
                       (0.5, 0), (-60, -75),
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
    

    main_axes.legend()
    plt.savefig(PLOT_FILE_NAME, dpi=DPI)
    
    plt.show()


def guesser(data, function):
    """
    Function to generate initial estimates for fitted parameters.

    A guess for velocity is made by rearranging the fitted equation for when
    the peak doppler shift is observed,

        v_0 = ((lambda_max/lambda_0) - 1)*(c/sin(inclination angle)).

    An estimate for the period is found by finding times when the observed
    wavelength is close to the emitted wavelength, i.e, times when,

        sin(omega t + phi) = 0.

    The difference betweens these times results in obtaining a half-period,
    which can be used to calculate an estimate angular velocity. This works
    even when more than 1 cycle of oscillation is observed as the differences
    are averaged.

    An estimate for the phase is found by generating time values of a
    "pure sin" wave, i.e., with the function with 0 phase difference. The time
    of maximum amplitude of this wave is then subtracted from the time of the
    maximum value of the data (delta t), and the estimated phase difference is
    found by,

        phi = (2 * pi * delta t) / period.

    Parameters
    ----------
    data : array_like
        Contains data being fitted.
    function : function
        Curve to which the data will be fitted.

    Returns
    -------
    guess_list : array_like
        Initial parameter guesses.

    """
    # Velocity estimate
    velocity_guess = ((np.max(data[:, 1]) / EMITTED_WAVELENGTH)
                      - 1) * (c / INCLINATION_FACTOR)

    # Angular velocity estimate
    central_times = data[np.where(np.isclose(EMITTED_WAVELENGTH,
                                             data[:, 1], rtol=1e-9)), 0]
    half_period = np.average(np.diff(central_times))
    angular_velocity_guess = np.pi / half_period

    # Phase estimate
    pure_sin = function([velocity_guess, angular_velocity_guess, 0],
                        data[:, 0])
    pure_max_time = data[np.where(np.isclose(np.max(data[:, 1]), pure_sin,
                                             rtol=1e-9)), 0]
    time_diff = np.abs(pure_max_time - data[np.where(data[:, 1] == np.max(
        data[:, 1])), 0])
    phase_guess = ((np.pi * time_diff) / half_period)[0][0]

    guess_list = [velocity_guess, angular_velocity_guess, phase_guess]

    return guess_list


def display_list_gen(value_list, error_list, rounding_function):
    """
    Rounds and converts fitted and calculated values to strings so that they
    may be displayed to the user in the terminal and on the plot. The order
    of values in value_list must match the order of errors in error_list.

    Parameters
    ----------
    value_list : array_like
        List of values (fitted and calculated).
    error_list : array_like
        List of errors corresponding to each value.
    rounding_function : function
        Required function to round

    :param value_list:
    :param error_list:
    :param rounding_function:
    :return:
    """
    display_list = np.empty((0, 2))

    for error, value in zip(error_list, value_list):
        rounded_value = rounding_function(value)
        decimal_places = str(rounded_value)[::-1].find(".")
        rounded_error = f'{error:.{decimal_places}f}'
        display_list = np.append(display_list,
                                 [[rounded_value, rounded_error]], axis=0)

    return display_list


def displayer(display_list) -> None:
    """
    Prints output to terminal.

    Parameters
    ----------
    display_list : array_like
        List of values which were calculated and are now displayed to the user.

    Returns
    -------
    None
    """
    print(f"\nBelow values found after {optimiser.counter} iterations.\n"
          f""
          f"Chi^2 = {display_list[0][0]}\n"
          f""
          f"Reduced Chi^2 = {display_list[0][0]}"
          f"\n"
          f"\nVelocity: ({display_list[1][0]} +/- {display_list[1][1]}) m/s\n"
          f""
          f"Angular velocity: ({display_list[2][0]} +/- "
          f"{display_list[2][1]}) rad/year\n"
          f""
          f"Phase: ({display_list[3][0]} +/- {display_list[3][1]}) rad\n"
          f"\n"
          f"Orbital radius=({display_list[4][0]} + /- "
          f"{display_list[4][1]}) AU\n"
          f""
          f"Planet mass=({display_list[5][0]} + /- {display_list[5][1]}) M_jup"
          )


def main() -> None:
    """
    Main function. Calls all other functions.

    Returns
    -------
    None
    """
    data = initial_data_filtering(data_getter())  # Retrieves and filters data

    optimiser.counter = 1  # initialise counter

    guess_list = guesser(data, wavelength_function)

    # Optimises data and fit
    optimised_parameters_list, data, hessian_matrix = optimiser(
        function_list=[chi_squared_func, data_filterer, wavelength_function],
        data_array=data, parameter_guess_list=guess_list)

    # Fit information
    chi = np.round(chi_squared_func(optimised_parameters_list,
                                    data, wavelength_function), decimals=3)
    reduced_chi = np.round(chi / (len(data) - len(optimised_parameters_list)),
                           decimals=3)

    # Standard errors on parameters found as described in the header.
    error_on_parameters = np.sqrt(np.diag(hessian_matrix))

    # Fitted parameters. Assigned to variables for calculation.
    velocity = optimised_parameters_list[0]  # m/s
    angular = optimised_parameters_list[1]  # rad / year

    # Errors on fitted parameters. Assigned to variables for calculation.
    error_v = error_on_parameters[0]
    error_w = error_on_parameters[1]

    # Calculated values
    period = 2 * np.pi / angular
    period_error = propagator([angular], [error_w], period)

    radius = np.cbrt(period ** 2)
    radius_error = propagator([period], [period_error], radius, (3 / 2))

    # Not displayed, only needed for further calculation.
    planet_velocity = np.sqrt((G * STAR_MASS * M_SUN) / (radius * AU))
    planet_v_error = propagator([radius], [radius_error],
                                planet_velocity, power=(1 / 2),
                                constant=np.sqrt(G * STAR_MASS))

    planet_mass = ((STAR_MASS * M_SUN * velocity) / planet_velocity) / M_JUP
    planet_mass_error = propagator([planet_velocity, velocity],
                                   [planet_v_error, error_v],
                                   planet_mass, constant=STAR_MASS)

    value_list = np.append(optimised_parameters_list, (planet_mass, radius))
    error_list = np.append(error_on_parameters,
                           (planet_mass_error, radius_error))

    display_list = np.array([[f'{chi:.3f}', f'{reduced_chi:.3f}']])
    display_list = np.append(display_list,
                             display_list_gen(value_list, error_list,
                                              significant_figure_rounder),
                             axis=0)

    displayer(display_list)
    
    data_point_plotter(data, wavelength_function, optimised_parameters_list,
                       display_list)


if __name__ == '__main__':
    main()
    
