import numpy as np
import matplotlib.pyplot as plt
from grand_minimizer_v08a import Island, E_hexi     

if False: 
    kwargs = {'polarity_1': 'p', 'polarity_2': 'p', 'power': 0.8} # more like a honeycomb

    i1 = Island(R=20, kind='h', nmax=2, E_surf=E_hexi, E_surf_kwargs=kwargs)

    i1.show(title='+/-20 degrees high density coincidence sites')

    i1.initialize_positions(R=0, strain=-0.2)

    i1.show(title='0 degrees, crazy strain=0.5')

    kwargs = {'polarity_1': 'n', 'polarity_2': 'p', 'power': 1.5} # more like a honeycomb

    i1.set_E_surf(E_surf_kwargs=kwargs)

    i1.show(title='a new potential', colors='blue')


if False: 
    kwargs = {'polarity_1': 'n'} # more like a honeycomb

    i2 = Island(R=20, a=1.3, kind='h', nmax=2, E_surf=E_hexi, E_surf_kwargs=kwargs)
    i2.show()

if False:
    for i, ((x, y), (gx, gy)) in enumerate(zip(i2.xy.T, i2.grad_surf.T)):
        print(i, [round(thing, 6) for thing in (x, y, gx, gy)])

# set it on fire!

kwargs = {'polarity_1': 'n'} # more like a honeycomb

i3 = Island(R=20, a=1.3, strain=0.02, kind='h', nmax=2,
            E_surf=E_hexi, E_surf_kwargs=kwargs, E_bond_alpha=10.)

i3.show(show_arrows=True)

t_eval = np.linspace(0, 2, 11)

# i3.set_alpha(10)

i3.solve_as_ivp(t_eval=t_eval)


fig, ax = plt.subplots(1, 1)
x, y = i3.xy
ax.plot(x, y, 'ok')
x, y = i3.xy_final
ax.plot(x, y, 'or')
ax.set_aspect('equal')
plt.show()



