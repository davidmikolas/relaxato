import numpy as np
import matplotlib.pyplot as plt
from grand_minimizer_v15 import Island, E_hexi, Island_from_HOC    

kwargs = {'polarity_1': 'n', 'scale': 1.0} # more like a honeycomb

t_eval = np.linspace(0, 30, 300)

rotations = np.linspace(0, 30, 31)

E_bond_alpha = 1.
damping = 1.

if True:
    final_energies = []
    final_rotations = []
    final_rotations_weighted = []
    final_mean_bond_length = []
    for R in rotations:
        print(R)
        island = Island(R=R, a=1.35, nmax=5, randomize='sq', sig_xy=0.0,
                    E_surf=E_hexi, xy_offset=[0.0, 0], # iseed=42, 
                    E_surf_kwargs=kwargs, E_bond_alpha=E_bond_alpha) # nmax=5, # E_bond_alpha=30
        island.solve_as_ivp(t_eval=t_eval, damping=damping, rtol=1E-08, dense_output=True) # damping=5,
        final_rotations.append(island.mean_angle_final)
        final_rotations_weighted.append(island.weighted_mean_angle_final)
        final_energies.append(island.total_energy_final)
        final_mean_bond_length.append(island.bond_lengths_final.mean())
    final_energies = np.array(final_energies)
    final_rotations = np.array(final_rotations)
    final_rotations_weighted = np.array(final_rotations_weighted)
    final_mean_bond_length = np.array(final_mean_bond_length)
    
if True:
    fig, axes = plt.subplots(1, 3)
    ax1, ax2, ax3 = axes
    titles = ('total energy', 'mean rotation', 'mean bond length')
    ax1.plot(rotations, final_energies)
    ax2.plot(rotations, final_rotations)
    ax2.plot(rotations, final_rotations_weighted, '--')
    ax3.plot(rotations, final_mean_bond_length)
    for ax, title in zip(axes, titles):
        ax.set_title(title)
    # ax.get_yaxis().get_major_formatter().set_useOffset(False)
    # ax.set_ylim(0, final_energies.max() + 1)
    plt.show()
