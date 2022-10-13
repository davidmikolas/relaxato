import numpy as np
import matplotlib.pyplot as plt
from grand_minimizer_v15 import Island, E_hexi, Island_from_HOC    

kwargs = {'polarity_1': 'n', 'scale': 1.0} # more like a honeycomb

t_eval = np.linspace(0, 100, 1000)

E_bond_alpha = 2

if True:
    final_energies = []
    for i in range(10):
        print(i)
        island = Island(R=30, a=1.30, nmax=5, randomize='sq', sig_xy=0.1,
                    E_surf=E_hexi, xy_offset=[0.0, 0], # iseed=42, 
                    E_surf_kwargs=kwargs, E_bond_alpha=E_bond_alpha) # nmax=5, # E_bond_alpha=30
        island.solve_as_ivp(t_eval=t_eval, damping=5, rtol=1E-08, dense_output=True) # damping=5,
        final_energies.append(island.total_energy_final)
    final_energies = np.array(final_energies)
    
if True:
    fig, ax = plt.subplots(1, 1)
    ax.plot(final_energies)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.set_ylim(0, final_energies.max() + 1)
    plt.show()
