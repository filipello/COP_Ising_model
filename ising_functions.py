import numpy as np
import matplotlib.pyplot as plt

def initial_spins(N, temperature):
    spin_0 = np.random.choice([1,-1])
    if temperature <= 2.2:
        lattice = np.random.choice([spin_0, -spin_0], size=(N,N), p=[0.85, 0.15])
    if temperature > 2.2 and temperature <= 2.4:
        lattice = np.random.choice([spin_0, -spin_0], size=(N,N), p=[0.7, 0.3])
    else:
        lattice = np.random.choice([spin_0, -spin_0], size=(N,N), p=[0.5, 0.5])
    return lattice


def system_energy(lattice, J):
#     this method is way faster the the double for loop

    energy = -J * np.sum(lattice * (np.roll(lattice, -1, axis=0) + np.roll(lattice, -1, axis=1)))
    return energy



def monte_carlo(lattice, coupling_constant, temperature):
    beta = 1/temperature
    size = len(lattice[0,:])
    random_x, random_y = np.random.randint(size, size=2)
    delta_E = 2 * coupling_constant * lattice[random_x, random_y] * (lattice[(random_x-1)%size, random_y] 
                                                              + lattice[(random_x+1)%size, random_y] 
                                                              + lattice[random_x, (random_y-1)%size] 
                                                              + lattice[random_x, (random_y+1)%size])
    if delta_E <= 0:
        lattice[random_x, random_y] *= -1
        return lattice, delta_E
    if np.random.rand() < np.exp(-delta_E * beta):
        lattice[random_x, random_y] *= -1
        return lattice, delta_E
    else:
        return lattice, 0

def show_snapshot(lattice, timestep):
    
    plt.imshow(lattice, cmap='binary')
    plt.title(f'Timestep {timestep}')
    plt.show()


def ising_simulation(size, coupling_constant, temperature, timesteps):
#     two metropolis algorithm starting from the same initial lattice
    lattice = initial_spins(size, temperature)
    energies = []
    magnetizations = []
    if temperature >= 2.0 and temperature <= 2.6:
        t_to_eq = 250000
    else:
        t_to_eq = 70000
    for i in range(timesteps + t_to_eq):
        lattice, delta_E = monte_carlo(lattice, coupling_constant, temperature) 
        if i >= t_to_eq:
            energies.append(system_energy(lattice, coupling_constant)/size**2)
            magnetizations.append(np.sum(lattice)/size**2)
            
#             if i%1000 == 0:
#                 show_snapshot(lattice, i-t_to_eq)
                
    return np.array(energies), np.array(magnetizations)

def auto_correlation(magnetization, timestep):
    xi = np.zeros(timestep)
    for t in range(timestep):
        xi[t] = (np.sum(magnetization[0:timestep-t]*magnetization[t:timestep])) * (timestep-t)**(-1) -  ((np.sum(magnetization[0:timestep-t]) * (timestep-t)**(-1) ) * (np.sum(magnetization[t:timestep]) * (timestep-t)**(-1)))
        
    return xi

def corr_time(chi):
    i = 0
    integral = 0
    while chi[i] >= 0:
        integral += chi[i]/chi[0]
        i += 1
    return integral


def average_chi(size, coupling_constant, temperature, timesteps): #test
    iterations = 15
    average_chi = np.zeros((timesteps))
    tau_noised = np.zeros((iterations))
    for i in range(iterations):
        energies, magnetizations = ising_simulation(size, coupling_constant, temperature, timesteps)
        chi_noised = auto_correlation(magnetizations, timesteps)
        tau_noised[i] = corr_time(chi_noised)
        average_chi += chi_noised
    return average_chi/iterations, tau_noised

def average_tau(size, coupling_constant, temperature, timesteps):
    time_correlation, tau_noise = average_chi(size, coupling_constant, temperature, timesteps)
    average_tau = corr_time(time_correlation)
    std_tau = np.std(tau_noise)
    return average_tau, std_tau



def magnetic_susceptibility(magnetization, tau, temperature, n_spins, t_max):
    """Computes the magnetic susceptibility per spin for equal blocks of size 16*tau"""
    #block_length = 16*tau
    block_length = int(16*tau)
    beta = 1/temperature
    magn_susc = []    
    for j in np.arange(0, int(t_max/block_length)):
        magn_susc.append((beta/n_spins**2)*(np.mean(magnetization[j*block_length:(j+1)*block_length]**2)-np.mean(magnetization[j*block_length:(j+1)*block_length])**2))
    mean_magn_susc = np.mean(magn_susc)
#     stdev_susceptibility = np.sqrt((2*tau/t_max)*(np.mean(magn_susc**2)-np.mean(magn_susc)**2))
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
#     stdev_specific_heat = np.sqrt((2*tau/t_max)*(np.mean(spec_heat**2)-np.mean(spec_heat)**2))
    stdev_specific_heat = np.std(spec_heat)
    return mean_spec_heat, stdev_specific_heat

def avg_magnetization_and_stdev(magnetization, tau, t_max):
    #t_max = sweeps
    mean_magnetization = np.mean(np.abs(magnetization)) 
    stdev_magnetization = np.sqrt((2*tau/t_max)*(np.mean(magnetization**2)-np.mean(magnetization)**2))
    return mean_magnetization, stdev_magnetization
                                  
def avg_energy_and_stdev(energy, tau, t_max):
    mean_energy = np.mean(energy)
    stdev_energy = np.sqrt((2*tau/t_max)*(np.mean(energy**2)-np.mean(energy)**2))
    return mean_energy, stdev_energy
                           



def plot_corr_time(temperature, tau, tau_error):
    plt.figure()
    plt.errorbar(temperature, tau, yerr = tau_error, fmt='x:b')
    plt.xlabel('Temperature')
    plt.ylabel(r'$\tau$')
    plt.title('Auto-correlation time')
    plt.savefig('corr_time.png')
    plt.show()


def plot_specific_heat(temperature, spec_heat, spec_heat_err):
    plt.figure()
    plt.errorbar(temperature, spec_heat, spec_heat_err, fmt='x:r')
    plt.xlabel('Temperature')
    plt.ylabel(r'$C$')
    plt.title('Specific heat')
    plt.savefig('sp_heat.png')
    plt.show()
    
def plot_magnetic_susc(temperature, magn_susc, magn_susc_err):
    plt.figure()
    plt.errorbar(temperature, magn_susc, magn_susc_err, fmt='x:g')
    plt.xlabel('Temperature')
    plt.ylabel(r'$\chi_M$')
    plt.title('Magnetic susceptibility')
    plt.savefig('magn_susc.png')
    plt.show()
    
def plot_average_spin(temperature, magnetization, magnetization_err):
    plt.figure()
    plt.errorbar(temperature, magnetization, magnetization_err, fmt='x:c')
    plt.xlabel('Temperature')
    plt.ylabel(r'$m$')
    plt.title('Mean absolute spin')
    plt.savefig('magnetization.png')
    plt.show()
    
def plot_energy(temperature, energy, energy_err):
    plt.figure()
    plt.errorbar(temperature, energy, energy_err, fmt='x:m')
    plt.xlabel('Temperature')
    plt.ylabel(r'$e$')
    plt.title('Energy per spin')
    plt.savefig('energy.png')
    plt.show()
    
def run_simulation():
    n_spins = 50
    coupling_constant = 1
    n_steps = 100000
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
        correlation_time[i], correlation_time_error = average_tau(n_spins, coupling_constant, temperature[i], n_steps)
        t_max = int(20*16*correlation_time[i])
        energy, magnetiz = ising_simulation(n_spins, coupling_constant, temperature[i], t_max)
        spec_heat[i], spec_heat_error[i] = specific_heat(energy, correlation_time[i], temperature[i], n_spins, t_max)
        magnetic_susc[i], magnetic_susc_error[i] = magnetic_susceptibility(magnetiz, correlation_time[i], temperature[i], n_spins, t_max)
        energies[i], energies_error[i] = avg_energy_and_stdev(energy, correlation_time[i], t_max)
        magnetizations[i], magnetizations_error[i] = avg_magnetization_and_stdev(magnetiz, correlation_time[i], t_max)
        print(i)
    return energies, energies_error, magnetizations, magnetizations_error, spec_heat, spec_heat_error, magnetic_susc, magnetic_susc_error, correlation_time, correlation_time_error
        
    
