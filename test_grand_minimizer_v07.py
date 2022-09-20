import numpy as np
import matplotlib.pyplot as plt
from grand_minimizer_v07 import Island, E_hexi    

if True:
    kwargs = {'polarity_1': 'p', 'polarity_2': 'p', 'power': 0.8} # more like a honeycomb

    i1 = Island(R=20, kind='h', nmax=2, E_surf=E_hexi, E_surf_kwargs=kwargs)

    i1.show(title='+/-20 degrees high density coincidence sites')

    i1.initialize_positions(R=0, strain=-0.2)

    i1.show(title='0 degrees, crazy strain=0.5')

    kwargs = {'polarity_1': 'n', 'polarity_2': 'p', 'power': 1.5} # more like a honeycomb

    i1.set_E_surf(E_surf_kwargs=kwargs)

    i1.show(title='a new potential', colors='blue')

kwargs = {'polarity_1': 'n'} # more like a honeycomb

i2 = Island(R=20, a=1.3, kind='h', nmax=2, E_surf=E_hexi, E_surf_kwargs=kwargs)
i2.show()

for i, ((x, y), (gx, gy)) in enumerate(zip(i2.xy.T, i2.grad_surf_xy.T)):
    print(i, [round(thing, 6) for thing in (x, y, gx, gy)])

# set it on fire!

i3 = Island(R=20, a=1.3, strain=0.2, kind='h', nmax=2,
            E_surf=E_hexi, E_surf_kwargs=kwargs)
i3.show(show_arrows=True)
# i3.FIRE_Island(N_steps=100)


