import ising_functions as fn
import numpy as np



energy_with_err, magnetization_with_err, specific_heat_with_error, magnetic_susc_with_error, corr_time_with_error = fn.run_simulation()

temperature = np.arange(1, 4.1, 0.2)

fn.plot_corr_time(temperature, corr_time_with_error[0], corr_time_with_error[1])

fn.plot_specific_heat(temperature, specific_heat_with_error[0], specific_heat_with_error[1])

fn.plot_energy(temperature, energy_with_err[0], energy_with_err[1])

fn.plot_average_spin(temperature, magnetization_with_err[0], magnetization_with_err[1])

fn.plot_magnetic_susc(temperature, magnetic_susc_with_error[0], magnetic_susc_with_error[1])


