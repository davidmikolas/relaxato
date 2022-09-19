import numpy as np
import matplotlib.pyplot as plt
from grand_minimizer_v05 import Island, E_hexi    

kwargs = {'polarity_1': 'p', 'polarity_2': 'p', 'power': 0.8} # more like a honeycomb

i1 = Island(R=20, kind='h', nmax=5, E_surf=E_hexi, E_surf_kwargs=kwargs)

i1.show(title='+/-20 degrees high density coincidence sites')

i1.update(R=0, strain=0.5)

i1.show(title='0 degrees, crazy strain=0.5')

kwargs = {'polarity_1': 'n', 'polarity_2': 'p', 'power': 1.5} # more like a honeycomb

i1.set_E_surf(E_surf_kwargs=kwargs)

i1.show(title='a new potential', colors='blue')


