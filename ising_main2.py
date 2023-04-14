import ising_functions2 as fn
import numpy as np



energy, energy_error, magnetization, magnetization_error, specific_heat, specific_heat_error, magnetic_susc, magnetic_susc_error, corr_time, corr_time_error = fn.run_simulation()

temperature = np.arange(1, 1.1, 0.2)

fn.plot_corr_time(temperature, corr_time, corr_time_error)

fn.plot_specific_heat(temperature, specific_heat, specific_heat_error)

fn.plot_energy(temperature, energy, energy_err)

fn.plot_average_spin(temperature, magnetization, magnetization_err)

fn.plot_magnetic_susc(temperature, magnetic_susc, magnetic_susc_error)