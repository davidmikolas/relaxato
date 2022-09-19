import numpy as np
import matplotlib.pyplot as plt

### Check Bonds Plot
### Calculate Bond Energies (strain)

class Island():
    def __init__(self, a=1, R=0, strain=0, kind='hex', nmax=3, nmin=None,
                 E_surf=None, E_surf_kwargs=None, E_bond_alpha=None):
        self.a = float(a)
        self.R = float(R)
        self.strain = float(strain)
        self.kind = str(kind)
        self.nmax = int(nmax)
        if nmin != None:
            self.nmin = min(max(int(nmin), 0), self.nmax)
        else:
            self.nmin = self.nmax
        if E_surf != None:
            self.set_E_surf(E_surf, E_surf_kwargs=E_surf_kwargs)
        self.getvecs()
        self.get_raw_ij()
        self.prune_ij()
        self.get_bonds()
        self.update()
        if isinstance(E_bond_alpha, float):
            self.set_alpha(E_bond_alpha)

    def get_raw_ij(self):
        ij = np.mgrid[-self.nmax:self.nmax+1, -self.nmax:self.nmax+1]
        keep = np.abs(ij.sum(axis=0)) <= self.nmax
        self.ij_raw = ij[:, keep]

    def prune_ij(self):
        if self.kind.lower() in ('h', 'hex', 'hexagon', 'hexagonal'):
            self.ij = self.ij_raw.copy()
        elif self.kind.lower() in ('t', 'tri', 'triangle', 'triangular'):
            i, j = self.ij_raw
            keep = (i >= 0) * (j >= 0)
            self.ij = self.ij_raw[:, keep].copy()
        elif self.kind.lower() in ('b', 'bar'):
            i, j = self.ij_raw
            keep = j <= self.nmin
            self.ij = self.ij_raw[:, keep].copy()
        else:
            print('uhoh, unsupported kind')
            self.ij = None

    def get_bonds(self):
        six = np.array([[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]])
        ijs = self.ij.T
        bob = [(tuple(thing), i) for i, thing in enumerate(ijs)]
        self.atom_dict = dict(bob)
        self.bonds_list = []
        for i, j in ijs:
            bonds = []
            self.bonds_list.append(bonds)
            for di, dj in six:
                try:
                    bonds.append(self.atom_dict[(i+di, j+dj)])
                except:
                    pass        

    def update(self, R=None, strain=None):
        if R != None:
            self.R = float(R)
        if strain != None:
            self.strain = float(strain)
        xy = (self.ij[:, None] * self.vecs[:, :, None]).sum(axis=0)
        xy *= (1.0 + self.strain)
        s, c = [f(np.radians(self.R)) for f in (np.sin, np.cos)]
        rotm = np.array([[c, -s], [s, c]])
        xyr = (rotm[..., None] * xy).sum(axis=1)
        self.xy = xyr

    def set_E_surf(self, E_surf=None, E_surf_kwargs=None):
        if (E_surf != None):
            if  callable(E_surf):
                self.E_surf = E_surf
            else:
                print("E_surf not set because it wasn't callable")
        if E_surf_kwargs != None:
            self.E_surf_kwargs = E_surf_kwargs

    def set_alpha(self, E_bond_alpha):
        self.E_bond_alpha = E_bond_alpha

    def surface_energy(self, xy):
        return self.E_surf(xy, **self.E_surf_kwargs)

    def grad_bonds(self):
        grad_x, grad_y = [], []
        for (x, y), bonds in zip(self.xy.T, self.bonds_list):
            gx, gy = 0.0, 0.0
            for n_atom in bonds:
                xb, yb = self.xy.T[n_atom]
                dx, dy = (xb - x) / self.a, (yb - y) / self.a
                thing = (2 * ((dx**2 + dy**2)**0.5 - 1.) * 0.5 *
                         (dx**2 + dy**2)**-0.5 * 2)
                gx += thing * dx
                gy += thing * dy
            grad_x.append(gx)
            grad_y.append(gy)
        self.grad_x = np.array(grad_x)
        self.grad_y = np.array(grad_y)
        
    def getvecs(self):
        uvecs = np.array([[1, 0], [0.5, 3**0.5/2]])
        self.vecs = self.a * uvecs

    def show(self, nper=20, marker='o', marker_size=120, colors='red', title=None):
        x, y = self.xy
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        xmean = 0.5 * (xmin + xmax)
        ymean = 0.5 * (ymin + ymax)
        xhw, yhw = 0.5 * (xmax - xmin), 0.5 * (ymax-ymin)
        hw = max(xhw, yhw) + 1. 
        extent = [xmean-hw, xmean+hw, ymean-hw, ymean+hw]
        nhw = int(nper * hw)
        xy_plot = np.mgrid[extent[2]:extent[3]:(2*nhw+1)*1j,
                           extent[0]:extent[1]:(2*nhw+1)*1j]
        self.xy_plot = xy_plot[::-1].copy()  # make it xy instead of yx
        self.Es = self.surface_energy(self.xy_plot)
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(self.Es, origin='lower', extent=extent)
        clb = fig.colorbar(im, ax=ax)
        ax.scatter(x, y, marker=marker, linewidth=0, c=colors, s=marker_size)
        # for now draw 1/3 bonds
        f = 0.4
        for (x, y), bonds in zip(self.xy.T, self.bonds_list):
            for n_atom in bonds:
                xb, yb = self.xy.T[n_atom]
                dx, dy = f * (xb - x), f * (yb - y)
                ax.plot([x, x+dx], [y, y+dy], '-')
        if title != None:
            ax.set_title(title)
        plt.show()



if False:
    xmin, xmax, ymin, ymax = -8, 12, -5, 5
    nper = 20
    xmean = 0.5 * (xmin + xmax)
    ymean = 0.5 * (ymin + ymax)
    xhw, yhw = 0.5 * (xmax - xmin), 0.5 * (ymax-ymin)
    hw = max(xhw, yhw) + 1. 
    extent = [xmean-hw, xmean+hw, ymean-hw, ymean+hw]
    nhw = int(nper * hw)
    xy_plot = np.mgrid[extent[2]:extent[3]:(2*nhw+1)*1j,
                       extent[0]:extent[1]:(2*nhw+1)*1j] # hey! transpose
    xy_plot = xy_plot[::-1].copy()
    print('xy_plot.shape: ', xy_plot.shape)
    xp, yp = xy_plot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    im1 = ax1.imshow(xp, origin='lower')
    ax1.set_title('xp')
    clb = fig.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(yp, origin='lower')
    ax2.set_title('yp')
    clb = fig.colorbar(im2, ax=ax2)
    plt.show()
        

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





# https://stackoverflow.com/a/4915964/3904031
"""
while thing in some_list: some_list.remove(thing)    

with suppress(ValueError):
    while True:
        some_list.remove(thing)


is_not_thing = lambda x: x is not thing
cleaned_list = filter(is_not_thing, some_list)


cleaned_list = [ x for x in some_list if x is not thing ]
"""
# BUT SEE ALSO https://stackoverflow.com/questions/4211209/remove-all-the-elements-that-occur-in-one-list-from-another
# AND ALSO https://stackoverflow.com/questions/2793324/is-there-a-simple-way-to-delete-a-list-element-by-value
