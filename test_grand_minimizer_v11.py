import numpy as np
import matplotlib.pyplot as plt
from grand_minimizer_v11 import Island, E_hexi, Island_from_HOC    

kwargs = {'polarity_1': 'n'} # more like a honeycomb

island = Island(R=0, a=1.34, nmax=5, E_surf=E_hexi,
                E_surf_kwargs=kwargs, E_bond_alpha=10) # nmax=5, 

t_eval = np.linspace(0, 20, 201)

island.solve_as_ivp(t_eval=t_eval, damping=1)

angles, mean_angle, weighted_mean_angle = island.get_angles()

island.show(show_arrows=True, show_final=True, border=5, figsize=[12, 8]) # 0.5

for vxy in island.velocities:
    v = np.sqrt((vxy**2).sum(axis=0))
    plt.plot(v)
plt.show()


if True:
    results = []
    rotations = np.linspace(0, 60, 61)
    for R in rotations:
        island = Island(R=R, a=1.35, nmax=5, E_surf=E_hexi,
                        E_surf_kwargs=kwargs, E_bond_alpha=10)
        island.solve_as_ivp(t_eval=t_eval, damping=1)

        E_initial, E_final = [island.get_total_energy(xy) for xy in
                              (island.xy, island.xy_final)]
        angles, mean_angle, weighted_mean_angle = island.get_angles()
        results.append([E_initial, E_final, weighted_mean_angle])
        print(R, )

    things = [np.array(thing) for thing in zip(*results)]
    E_initial, E_final, weighted_mean_angle = things

if True:
    plt.figure()
    plt.plot(rotations, E_initial)
    plt.plot(rotations, E_final)
    plt.plot(rotations, weighted_mean_angle)
    plt.show()


"""
print('angles.shape, np.nanmin(angles), np.nanmax(angles): ',
      angles.shape, np.nanmin(angles), np.nanmax(angles))
print('final mean_angle, weighted_mean_angle: ', mean_angle, weighted_mean_angle)
"""

"""
# FROM 10
kwargs = {'polarity_1': 'n'} # more like a honeycomb

ijkl = (7, 1, 8, 1)
ijkl = (7, 0, 3, 3)

i1 = Island_from_HOC(ijkl, E_surf=E_hexi, E_surf_kwargs=kwargs,
                       E_bond_alpha=1)

i1.set_alpha(10)

t_eval = np.linspace(0, 10, 101)

i1.solve_as_ivp(t_eval=t_eval, damping=3)


i1.show(show_arrows=True, show_final=True, border=.5)

i2 = Island_from_HOC(ijkl, nmax=5, E_surf=E_hexi, E_surf_kwargs=kwargs,
                       E_bond_alpha=1)

i2.set_alpha(10)

i2.solve_as_ivp(t_eval=t_eval, damping=3)

i2.show(show_arrows=True, show_final=True, border=2.5)
"""

"""
# FROM 9
kwargs = {'polarity_1': 'n'} # more like a honeycomb

i3 = Island(R=20, a=1.3, strain=0.02, kind='h', nmax=5,
            E_surf=E_hexi, E_surf_kwargs=kwargs, E_bond_alpha=1)


t_eval = np.linspace(0, 10, 101)

alphas = np.logspace(0, 2, 3)

i3.set_alpha(10)

i3.solve_as_ivp(t_eval=t_eval, damping=3)

i3.show(show_arrows=True, show_final=True, border=2)

print('i3.velocities.shape: ', i3.velocities.shape)
for vxy in i3.velocities:
    v = np.sqrt((vxy**2).sum(axis=0))
    plt.plot(v)
plt.show()

"""



