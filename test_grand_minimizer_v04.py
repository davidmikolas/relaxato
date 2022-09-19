import numpy as np
import matplotlib.pyplot as plt
from grand_minimizer_v04 import Island

def E_hexi(xy, polarity_1='p', polarity_2='p', power=None):
    r3o2 = 3**0.5 / 2.
    kay = 2 * np.pi * np.array([[r3o2, -0.5], [r3o2, 0.5], [0, 1]]) / r3o2
    if xy.ndim == 3:
        three = np.cos((kay[..., None, None] * xy).sum(axis=1))
    elif xy.ndim == 2:
        three = np.cos((kay[..., None] * xy).sum(axis=1))
    else:
        print('uhoh!')
        three = None
    E = None
    if polarity_1.lower() in ('n', 'neg', 'negative'):
        E = 1 - (1.5 + three.sum(axis=0)) / 4.5
    elif polarity_1.lower() in ('p', 'pos', 'positive'):
        E = (1.5 + three.sum(axis=0)) / 4.5
    else:
        print('unclear polarity_1')
    if isinstance(power, (int, float)):
        E = E ** power
    if polarity_2.lower() in ('n', 'neg', 'negative'):
        E = 1.0 - E
    elif polarity_1.lower() in ('p', 'pos', 'positive'):
        pass
    else:
        print('unclear polarity_2')
    return E

    

kwargs = {'polarity_1': 'p', 'polarity_2': 'p', 'power': 0.8} # more like a honeycomb

i5 = Island(R=20, kind='h', nmax=5, E_surf=E_hexi, E_surf_kwargs=kwargs)

i5.show(title='+/-20 degrees high density coincidence sites')

i5.update(R=0, strain=0.5)

i5.show(title='0 degrees, crazy strain=0.5')



