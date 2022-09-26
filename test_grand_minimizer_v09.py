import numpy as np
import matplotlib.pyplot as plt
from grand_minimizer_v09 import Island, E_hexi     

kwargs = {'polarity_1': 'n'} # more like a honeycomb

i3 = Island(R=20, a=1.3, strain=0.02, kind='h', nmax=10,
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





