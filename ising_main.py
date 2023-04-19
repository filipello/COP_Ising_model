import ising_functions as fn



initialisation = True
while initialisation == True:
    print('This is a simulation of a 2-D Ising model. Please choose one of the following options by pressing the corresponding number')
    print('1. Take all the physical measurements')
    print('2. Plot the snapshots of the evoluting lattice \n')
    get_input = input()
    
    if get_input == '1':
        print('Input the value of the external magentic field (float) \n')
        get_extetnal_field = float(input())
        fn.run_simulation(get_extetnal_field)
        break
