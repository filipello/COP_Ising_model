import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def initial_spins(N, temperature):
    ''' Setting the initial lattice, with a different probability distribution for different temperatures in order to reach equilibrium faster '''
    spin_0 = np.random.choice([1,-1])
    if temperature <= 2.2:
        lattice = np.random.choice([spin_0, -spin_0], size=(N,N), p=[0.9, 0.1])
    if temperature > 2.2 and temperature <= 2.4:
        lattice = np.random.choice([spin_0, -spin_0], size=(N,N), p=[0.8, 0.2])
    if temperature > 2.4:
        lattice = np.random.choice([spin_0, -spin_0], size=(N,N), p=[0.5, 0.5])
    return lattice


def system_energy(lattice, J, magn_field):
    '''Calculating the energy of the lattice'''
    energy = -J * np.sum(lattice * (np.roll(lattice, -1, axis=0) + np.roll(lattice, -1, axis=1))) - magn_field * np.sum(lattice)
    return energy



def monte_carlo(lattice, coupling_constant, temperature, magn_field):
    '''Function to implement a Monte-Carlo method to swap a single spin, it returns the new lattice and energy difference'''
    beta = 1/temperature
    size = len(lattice[0,:])
    random_x, random_y = np.random.randint(size, size=2)
    delta_E = 2 * coupling_constant * lattice[random_x, random_y] * (lattice[(random_x-1)%size, random_y] 
                                                              + lattice[(random_x+1)%size, random_y] 
                                                              + lattice[random_x, (random_y-1)%size] 
                                                              + lattice[random_x, (random_y+1)%size]) + 2* magn_field * lattice[random_x, random_y]
    if delta_E <= 0:
        lattice[random_x, random_y] *= -1
        return lattice, delta_E
    if np.random.rand() < np.exp(-delta_E * beta):
        lattice[random_x, random_y] *= -1
        return lattice, delta_E
    else:
        return lattice, 0





def show_snapshot(lattice, timestep):
    
    plt.imshow(lattice, cmap='magma')
    plt.title(f'Timestep {timestep}')
    plt.show()


def ising_simulation(size, coupling_constant, temperature, timesteps, magn_field):
    '''This method implements a Metropolis algorithm to evolve the system for the desired timesteps. Before taking measurements the system runs for t_to_eq timesteps in order to reach equilibrium. It returns the energy and magnetization of the system at each timestep (after equilibrium) as numpy arrays'''
    lattice = initial_spins(size, temperature)
    energies = np.zeros((timesteps))
    magnetizations =  np.zeros((timesteps))
    if temperature >= 1.8 and temperature <= 2.6:
        t_to_eq = 300000
    else:
        t_to_eq = 30000
    for i in range(timesteps + t_to_eq):
        lattice, delta_E = monte_carlo(lattice, coupling_constant, temperature, magn_field) 
        if i >= t_to_eq:
            energies[i- t_to_eq] = (system_energy(lattice, coupling_constant, magn_field)/size**2)
            magnetizations[i - t_to_eq] = (np.sum(lattice)/size**2)
                
    return energies, magnetizations

def auto_correlation(magnetization, timestep):
    '''This methd calculates the integral for the auto-corelation function at each timestep, in a discretized form. It returns the auto-correlation function at each timestep'''
    xi = np.zeros(timestep)
    for t in range(timestep):
        xi[t] = (np.sum(magnetization[0:timestep-t]*magnetization[t:timestep])) * (timestep-t)**(-1) -  ((np.sum(magnetization[0:timestep-t]) * (timestep-t)**(-1) ) * (np.sum(magnetization[t:timestep]) * (timestep-t)**(-1)))
        
    return xi

def corr_time(chi):
    '''Thiis method integrates the normalized auto-correlation function until it reaches zero in order to return the auto-correlation time'''
    i = 0
    integral = 0
    while chi[i] >= 0 and i < len(chi):
        integral += chi[i]/chi[0]
        i += 1
    return integral


def average_chi(size, coupling_constant, temperature, timesteps, magn_field): 
    '''This method repeats the measurement for the auto-correlation function in order to reduce the noise. It returns the correlation function after noise reduction and the measured auto-correlation time at each iteration'''
    iterations = 10
    average_chi = np.zeros((timesteps))
    tau_noised = np.zeros((iterations))
    for i in range(iterations):
        energies, magnetizations = ising_simulation(size, coupling_constant, temperature, timesteps, magn_field)
        chi_noised = auto_correlation(magnetizations, timesteps)
        tau_noised[i] = corr_time(chi_noised)
        average_chi += chi_noised
    return average_chi/iterations, tau_noised

def average_tau(size, coupling_constant, temperature, timesteps, magn_field):
    '''This method evaluates a de-noised correlation function and then returns a correlation time without noise and its error (the standard deviation of all the measured  correlation times'''
    time_correlation, tau_noise = average_chi(size, coupling_constant, temperature, timesteps, magn_field)
    average_tau = corr_time(time_correlation)
    std_tau = np.std(tau_noise)
    return average_tau, std_tau



def magnetic_susceptibility(magnetization, tau, temperature, n_spins, t_max):
    """Computes the magnetic susceptibility per spin for equal blocks of size 16*tau"""
    block_length = int(16*tau)
    beta = 1/temperature
    magn_susc = []    
    for j in np.arange(0, int(t_max/block_length)):
        magn_susc.append((beta/n_spins**2)*(np.mean(magnetization[j*block_length:(j+1)*block_length]**2)-np.mean(magnetization[j*block_length:(j+1)*block_length])**2))
    mean_magn_susc = np.mean(magn_susc)
    stdev_susceptibility = np.std(magn_susc)
    return mean_magn_susc, stdev_susceptibility

def specific_heat(energy, tau, temperature, n_spins, t_max):
    """Computes the specific heat per spin for equal blocks of size 16*tau"""
    block_length = int(16*tau)
    k_B = 1
    spec_heat = []
    for h in np.arange(0, int(t_max/block_length)):
        spec_heat.append((1/(k_B*temperature**2*n_spins**2))*(np.mean(energy[h*block_length:(h+1)*block_length]**2)-np.mean(energy[h*block_length:(h+1)*block_length])**2))
    mean_spec_heat = np.mean(spec_heat)
    stdev_specific_heat = np.std(spec_heat)
    return mean_spec_heat, stdev_specific_heat

def avg_magnetization_and_stdev(magnetization, tau, t_max):
    '''Computes the average absolute value of spins and its error'''
    mean_magnetization = np.mean(np.abs(magnetization)) 
    stdev_magnetization = np.sqrt((2*tau/t_max)*(np.mean(magnetization**2)-np.mean(magnetization)**2))
    return mean_magnetization, stdev_magnetization
                                  
def avg_energy_and_stdev(energy, tau, t_max):
    '''Computes the average energy value per spins and its error'''
    mean_energy = np.mean(energy)
    stdev_energy = np.sqrt((2*tau/t_max)*(np.mean(energy**2)-np.mean(energy)**2))
    return mean_energy, stdev_energy
                           


def run_simulation(magn_field):
    '''This method takes the value of the external field from the user and perform the whole simulation of a 50x50 lattice for 16 different temperature values. It calculates all required measurements and their error and save them in .zip file using a pandas DataFrame. '''
    n_spins = 50
    coupling_constant = 1
    n_steps = 100000
    if magn_field != 0:
        temperature = np.array([1.4, 1.8, 2.0, 2.2, 2.4 ,2.6, 3.0, 4.0])
    else:
        temperature = np.arange(1.0, 4.1, 0.2)
    
    
    energies = np.zeros(len(temperature))
    energies_error = np.zeros(len(temperature))
    magnetizations = np.zeros(len(temperature))
    magnetizations_error = np.zeros(len(temperature))
    spec_heat = np.zeros(len(temperature))
    spec_heat_error = np.zeros(len(temperature))
    magnetic_susc = np.zeros(len(temperature))
    magnetic_susc_error = np.zeros(len(temperature))
    correlation_time = np.zeros(len(temperature))
    correlation_time_error = np.zeros(len(temperature))
    
    for i in range(len(temperature)):
        print('currently measuring T=', f'{temperature[i]:.1f}')
        correlation_time[i], correlation_time_error[i] = average_tau(n_spins, coupling_constant, temperature[i], n_steps, magn_field)
        t_max = int(10*16*correlation_time[i])
        energy, magnetiz = ising_simulation(n_spins, coupling_constant, temperature[i], t_max, magn_field)
        spec_heat[i], spec_heat_error[i] = specific_heat(energy, correlation_time[i], temperature[i], n_spins, t_max)
        magnetic_susc[i], magnetic_susc_error[i] = magnetic_susceptibility(magnetiz, correlation_time[i], temperature[i], n_spins, t_max)
        energies[i], energies_error[i] = avg_energy_and_stdev(energy, correlation_time[i], t_max)
        magnetizations[i], magnetizations_error[i] = avg_magnetization_and_stdev(magnetiz, correlation_time[i], t_max)
    
    measurements_to_dataframe = {'temperature':temperature, 'energy':energies, 'error en.':energies_error, 'av spin':magnetizations, 
                                 'error spin':magnetizations_error, 'spec. heat':spec_heat, 'error spec. heat': spec_heat_error, 'magn. susc': magnetic_susc,                                      'error magn. susc': magnetic_susc_error, 'tau': correlation_time, 'error tau': correlation_time_error}
    measurements_df = pd.DataFrame(data = measurements_to_dataframe)

    measurements_df.to_csv('ising_measure.zip')
    

def plot_corr_time(temperature, tau, tau_error):
    '''Method to plot the correlation time (normalized by number of spins) and its error as a function of temperature'''
    plt.figure()
    plt.errorbar(temperature, tau/2500, yerr = tau_error/2500, fmt='x:b')
    plt.xlabel('Temperature')
    plt.ylabel(r'$\tau$')
    plt.title('Auto-correlation time')
    plt.savefig('corr_time.png', dpi = 200)
    plt.show()


def plot_specific_heat(temperature, spec_heat, spec_heat_err):
    '''Method to plot the specific heat and its error as a function of temperature'''
    plt.figure()
    plt.errorbar(temperature, spec_heat, spec_heat_err, fmt='x:r')
    plt.xlabel('Temperature')
    plt.ylabel(r'$C$')
    plt.title('Specific heat')
    plt.tight_layout()
    # plt.ticklabel_format(style='plain')
    plt.savefig('sp_heat.png', bbox_inches = "tight", dpi = 200)
    plt.show()
    
def plot_magnetic_susc(temperature, magn_susc, magn_susc_err):
    '''Method to plot the magnetic susceptibility and its error as a function of temperature'''
    plt.figure()
    plt.errorbar(temperature, magn_susc, magn_susc_err, fmt='x:g')
    plt.xlabel('Temperature')
    plt.ylabel(r'$\chi_M$')
    plt.title('Magnetic susceptibility')
    plt.ticklabel_format(style='sci')
    plt.savefig('magn_susc.png', bbox_inches = "tight", dpi = 200)
    plt.show()
    
def plot_average_spin(temperature, magnetization, magnetization_err):
    '''Method to plot the average absolute spin and its error as a function of temperature'''
    plt.figure()
    plt.errorbar(temperature, magnetization, magnetization_err, fmt='x:c')
    plt.xlabel('Temperature')
    plt.ylabel(r'$m$')
    plt.title('Mean absolute spin')
    plt.savefig('magnetization.png', dpi = 200)
    plt.show()
    
def plot_energy(temperature, energy, energy_err):
    '''Method to plot the average energy per spin and its error as a function of temperature'''
    plt.figure()
    plt.errorbar(temperature, energy, energy_err, fmt='x:m')
    plt.xlabel('Temperature')
    plt.ylabel(r'$e$')
    plt.title('Energy per spin')
    plt.savefig('energy.png', dpi = 200)
    plt.show()
    