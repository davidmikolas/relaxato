import numpy as np
import matplotlib.pyplot as plt

### DONE Check Bonds Plot and gradients   SEEMS OKAY 
### DONE Calculate Bond Energies and Gradients (strain)   SEEMS OKAY

### add FIRE to Islands

### add HOC
### add FIRE to apodized HOC

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
        if isinstance(E_surf_kwargs, dict):
            self.E_surf_kwargs = E_surf_kwargs
        else:
            self.E_surf_kwargs = dict()
        if callable(E_surf):
            self.set_E_surf(E_surf)
        self.initialize_vecs()
        self.calculate_raw_ij()
        self.prune_ij()
        self.define_bonds()
        self.initialize_positions()
        if isinstance(E_bond_alpha, float):
            self.set_alpha(E_bond_alpha)

    def initialize_vecs(self):
        uvecs = np.array([[1, 0], [0.5, 3**0.5/2]])
        self.vecs = self.a * uvecs

    def calculate_raw_ij(self):
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

    def define_bonds(self):
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
        self.get_unique_bonds()

    def get_unique_bonds(self):
        pairs = [ [(i, j) for j in bonds] for (i, bonds) in enumerate(self.bonds_list)]
        pairs = [tuple(sorted(pair)) for pair in sum(pairs, [])]
        self.unique_bond_pairs = [list(pair) for pair in set(pairs)]

    def initialize_positions(self, R=None, strain=None):
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

    def update_surface_energies(self):
        self.surface_energies = self.E_surf(self.xy, **self.E_surf_kwargs)
        self.grad_surf()

    def get_surface_energies(self, xy):
        return self.E_surf(xy, **self.E_surf_kwargs)

    def update_bond_energies(self):
        bond_energies = []
        for (a, b) in self.unique_bond_pairs:
            r = np.sqrt(((self.xy[:, b] - self.xy[:, a])**2).sum())
            bond_energies.append( (r / self.a - 1)**2)
        self.bond_energies = np.array(bond_energies)
        self.grad_bonds()

    def grad_surf(self, d=0.0001):
        self.grad_surf_d = d
        offsets = np.array([[d, 0], [-d, 0], [0, d], [0, -d]])
        Es = [self.get_surface_energies(self.xy + offset[:, None]) for offset in offsets]
        self.grad_surf_x = (Es[1] - Es[0]) / (2 * d)
        self.grad_surf_y = (Es[3] - Es[2]) / (2 * d)
        self.grad_surf_xy = np.vstack((self.grad_surf_x, self.grad_surf_y))

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
        self.grad_bond_x = np.array(grad_x)
        self.grad_bond_y = np.array(grad_y)
        self.grad_bond_xy = np.vstack((self.grad_bond_x, self.grad_bond_y))

    """
    def FIRE_Island(self, N_steps, alpha_start=0.25, f_alpha=0.99,
                    delta_t_start=0.01, delta_t_max=10*0.01, delta_t_min=0.02*0.01,
                    delta_t_fdec=0.5, N_delay=20): # asdf

        # initialize x(t) and F(x(t))
        x0 = self.xy.flatten().copy()   # actual starting positions
        v0 = np.zeros_like(x0)

        # initialize other stuff
        alpha_FIRE = alpha_start
        delta_t = delta_t_start
        f_delta_t_grow = 1.1
        Npgt0 = 0

        results = [[x0.copy(), v0.copy()]]
        asdf
        energy = [Energy(x0, a, L_total, omega, alpha)] 
        delta_ts = [delta_t_start]
        t = 0.
        for i in range(N_max):
            F = Force(x, L_total, omega, alpha)
            P = np.dot(F, v) # force you feel *dot* where you are going
            if P > 0:
                Npgt0 += 1
                F = Force(x, L_total, omega, alpha)
                F_norm = F / np.sqrt((F**2).sum())  # or np.linalg.norm()
                v_norm = v / np.sqrt((v**2).sum())
                v = (1-alpha_FIRE) * v + alpha_FIRE * F * (v_norm / F_norm) # This is it!
                if Npgt0 > N_delay:
                    delta_t = min(f_delta_t_grow * delta_t, delta_t_max)
                    alpha *= f_alpha_FIRE
            else: # P <= 0
                Npgt0 = 0
                v[:] = 0.   # stop! literally!
                delta_t = delta_t_fdec * delta_t
                alpha_FIRE = alpha_FIRE_start
            # now use https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
            # get new x and v
            x += v * delta_t + 0.5 * F * delta_t**2
            FF = Force(x, L_total, omega, alpha) # force at t + delta_t
            v += 0.5 * (F + FF) * delta_t
            # then
            results.append([x.copy(), v.copy()])
            energy.append(Energy(x, a, L_total, omega, alpha))
            delta_ts.append(delta_t)
            t += delta_t
            # check for convergence and break if you like
            # for example, has the energy stopped decreasing significantly?

            positions, velocities = np.swapaxes(np.array(list(zip(*results))), 1, 2)
            energy = np.array(energy)
            delta_ts = np.array(delta_ts)
    """
    
    def show(self, nper=20, marker='o', marker_size=120, colors='red',
             arrow_width=0.1, title=None, show_arrows=False):
        twopi = 2 * np.pi
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
        self.E_plot = self.get_surface_energies(self.xy_plot)
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(self.E_plot, origin='lower', extent=extent)
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
        self.update_surface_energies() # and gradients
        self.update_bond_energies()  # and gradients
        if show_arrows:
            for (x, y), (gx, gy) in zip(self.xy.T, self.grad_bond_xy.T):
                ax.arrow(x, y, gx, gy, width=arrow_width, color='k')
            for (x, y), (gx, gy) in zip(self.xy.T, self.grad_surf_xy.T):
                ax.arrow(x, y, gx, gy, width=arrow_width, color='r')
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
    elif polarity_2.lower() in ('p', 'pos', 'positive'):
        pass
    else:
        print('unclear polarity_2')
    return E

