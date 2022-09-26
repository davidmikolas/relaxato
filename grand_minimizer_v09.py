import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time


### DONE Check Bonds Plot and gradients   SEEMS OKAY 
### DONE Calculate Bond Energies and Gradients (strain)   SEEMS OKAY

### add FIRE to Islands

### add HOC
### add FIRE to apodized HOC 

class Island():
    def __init__(self, a=1, R=0, strain=0, kind='hex', nmax=3, nmin=None,
                 E_surf=None, E_surf_kwargs=None, E_bond_alpha=None,
                 grad_surf_d=0.00001):
        self.a = float(a)
        self.R = float(R)
        self.strain = float(strain)
        self.kind = str(kind)
        self.nmax = int(nmax)
        self.grad_surf_d = float(grad_surf_d)
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
        print('E_bond_alpha: ', E_bond_alpha)
        if isinstance(E_bond_alpha, float):
            self.set_alpha(E_bond_alpha)
            print('self.E_bond_alpha: ', self.E_bond_alpha)

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
            self.n_atoms = self.ij.shape[1]
        elif self.kind.lower() in ('t', 'tri', 'triangle', 'triangular'):
            i, j = self.ij_raw
            keep = (i >= 0) * (j >= 0)
            self.ij = self.ij_raw[:, keep].copy()
            self.n_atoms = self.ij.shape[1]
        elif self.kind.lower() in ('b', 'bar'):
            i, j = self.ij_raw
            keep = j <= self.nmin
            self.ij = self.ij_raw[:, keep].copy()
            self.n_atoms = self.ij.shape[1]
        else:
            print('uhoh, unsupported kind')
            self.ij = None
            self.n_atoms = None
        print('n_atoms: ', self.n_atoms)

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
        self.n_bonds = len(self.unique_bond_pairs)
        print('n_bonds: ', self.n_bonds)

    def initialize_positions(self, R=None, strain=None):
        if R != None:
            self.R = float(R)
        if strain != None:
            self.strain = float(strain)
        xy = (self.ij[:, None] * self.vecs[:, :, None]).sum(axis=0)
        xy *= (1.0 + self.strain)
        s, c = [f(np.radians(self.R)) for f in (np.sin, np.cos)]
        rotm = np.array([[c, -s], [s, c]])
        self.xy = (rotm[..., None] * xy).sum(axis=1)
        self.vxy = np.zeros_like(self.xy)

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
        self.surface_energy = self.surface_energies.sum()
        self.update_grad_surf()

    def get_surface_energies(self, xy):
        return self.E_surf(xy, **self.E_surf_kwargs)

    def update_bond_energies(self):
        self.bond_energies = self.get_bond_energies(self.xy)
        self.bond_energy = self.bond_energies.sum()
        self.update_grad_bonds()

    def get_bond_energies(self, xy):
        bond_energies = []
        for (a, b) in self.unique_bond_pairs:
            r = np.sqrt(((xy[:, b] - xy[:, a])**2).sum())
            bond_energies.append( (r / self.a - 1)**2)
        return self.E_bond_alpha * np.array(bond_energies)

    def update_grad_surf(self):
        self.grad_surf = self.get_grad_surf(self.xy, d=self.grad_surf_d)

    def get_grad_surf(self, xy, d):
        offsets = np.array([[d, 0], [-d, 0], [0, d], [0, -d]])
        Es = [self.get_surface_energies(xy + offset[:, None]) for offset in offsets]
        return np.vstack((Es[1] - Es[0], Es[3] - Es[2])) / (2 * d)

    def update_grad_bonds(self):
        self.grad_bonds = self.get_grad_bonds(self.xy)

    def get_grad_bonds(self, xy):
        grad_x, grad_y = [], []
        for (x, y), bonds in zip(xy.T, self.bonds_list):
            gx, gy = 0.0, 0.0
            for n_atom in bonds:
                xb, yb = xy.T[n_atom]
                dx, dy = (xb - x) / self.a, (yb - y) / self.a
                thing = (2 * ((dx**2 + dy**2)**0.5 - 1.) * 0.5 *
                         (dx**2 + dy**2)**-0.5 * 2)
                gx += thing * dx
                gy += thing * dy
            grad_x.append(gx)
            grad_y.append(gy)
        return self.E_bond_alpha * np.vstack((grad_x, grad_y))

    def solve_as_ivp(self, t_eval, rtol=1E-03, method='DOP853', damping=1.0): # asdf

        def deriv(t, state_vector, damping):
            xyf, vxyf = state_vector.reshape(2, -1)
            xy = xyf.reshape(2, -1)
            acc_surf = self.get_grad_surf(xy, d=self.grad_surf_d).flatten()
            acc_bonds = self.get_grad_bonds(xy).flatten()
            acc_damping = -damping * vxyf
            return np.hstack((vxyf, acc_surf + acc_bonds + acc_damping))
             

        # initialize x(t) and v(t)
        xyf = self.xy.flatten().copy()
        vxyf = np.zeros_like(xyf)
        state = np.hstack([xyf, vxyf])

        t_span = t_eval.min(), t_eval.max()
        t_start = time.process_time()
        args = (damping, )
        answer = solve_ivp(deriv, t_span=t_span, y0=state, method=method,
                           args=args, t_eval=t_eval, rtol=rtol,
                           dense_output=False, events=None)
        self.process_time = time.process_time() - t_start
        print('process time: ', self.process_time)

        self.state_vectors = answer['y']
        self.n_points = self.state_vectors.shape[1]
        view = self.state_vectors.reshape(4, -1, self.n_points)
        self.trajectories = np.moveaxis(view[:2], 0, 1)
        self.velocities = np.moveaxis(view[2:], 0, 1)
        self.xy_final, self.vxy_final = answer['y'][:, -1].reshape(2, 2, -1)
        self.ivp_answer = dict(answer)
        print('message: ', answer['message'])
        print('success: ', answer['success'])
        print('nfev: ', answer['nfev'])
        

    def FIRE_Island(self, N_steps, alpha_start=0.25, f_alpha=0.99,
                    delta_t_start=0.01, delta_t_max=10*0.01, delta_t_min=0.02*0.01,
                    delta_t_fdec=0.5, N_delay=20): # asdf

        # initialize x(t) and v(t)
        xy = self.xy.copy()
        vxy = np.zeros_like(xy)

        # initialize other stuff
        alpha = alpha_start
        delta_t = delta_t_start
        f_delta_t_grow = 1.1
        Npgt0 = 0

        # collect the trajectories just in case someone finds them interesting
        self.FIRE_results = [[xy.copy(), vxy.copy()]]
        self.FIRE_energies = [[self.get_surface_energies(xy),
                               self.get_bond_energies(xy)]]
        delta_ts = [delta_t_start]
        t = 0.
        for i in range(N_steps):
            ### GET FORCE
            Fxy = (self.get_grad_surf(xy, d=self.grad_surf_d) +
                   self.get_grad_bonds(xy))
            print('Fxy.shape: ', Fxy.shape)
            print('vxy.shape: ', vxy.shape)
            P = (Fxy * vxy).sum(axis=0) # dot product; force you feel *dot* where you are going
            if P > 0:
                Npgt0 += 1
                ### WAIT am I normalizing correctly?
                Fxy_norm = Fxy / np.sqrt((Fxy**2).sum(axis=0))  # 2 x N (or np.linalg.norm())
                vxy_norm = vxy / np.sqrt((vxy**2).sum(axis=0))  # 2 x N
                vxy = (1. - alpha) * v + alpha * Fxy * (vxy_norm / Fxy_norm) # This is it!
                if Npgt0 > N_delay:
                    delta_t = min(f_delta_t_grow * delta_t, delta_t_max)
                    alpha *= f_alpha
            else: # P <= 0
                Npgt0 = 0
                v[:] = 0.   # stop! literally!
                delta_t = delta_t_fdec * delta_t
                alpha = alpha_start
            # now use https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
            # get new x and v
            xy += vxy * delta_t + 0.5 * Fxy * delta_t**2
            FFxy = self.get_grad_surf(xy, d=self.d) + self.get_grad_bonds(xy) # force at t + delta_t
            vxy += 0.5 * (Fxy + FFxy) * delta_t
            # then
            self.results.append([xy.copy(), vxy.copy()])
            self.energies.append([self.get_surface_energies(xy),
                                  self.get_bond_energies(xy)])
            delta_ts.append(delta_t)
            t += delta_t
            # check for convergence and break if you like
            # for example, has the energy stopped decreasing significantly?
        self.xy_final, self.vxy_final = results[-1]
        print("I'm done playing with FIRE for now")

    
    def show(self, nper=20, marker='o', marker_size=120, colors='red',
             arrow_width=0.1, title=None, show_arrows=False,
             show_final=False, show_colorbars=False, border=0.2):
        twopi = 2 * np.pi
        x, y = self.xy
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        xmean = 0.5 * (xmin + xmax)
        ymean = 0.5 * (ymin + ymax)
        xhw, yhw = 0.5 * (xmax - xmin), 0.5 * (ymax-ymin)
        hw = max(xhw, yhw) + border 
        extent = [xmean-hw, xmean+hw, ymean-hw, ymean+hw]
        nhw = int(nper * hw)
        xy_plot = np.mgrid[extent[2]:extent[3]:(2*nhw+1)*1j,
                           extent[0]:extent[1]:(2*nhw+1)*1j]
        self.xy_plot = xy_plot[::-1].copy()  # make it xy instead of yx
        self.E_plot = self.get_surface_energies(self.xy_plot)
        if show_final:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            # original configuration plot (left)
            im1 = ax1.imshow(self.E_plot, origin='lower', extent=extent)
            if show_colorbars:
                clb1 = fig.colorbar(im1, ax=ax1)
            x, y = self.xy
            ax1.scatter(x, y, marker=marker, linewidth=0, c=colors, s=marker_size)
            # for now draw 1/3 bonds
            f = 0.4
            if title != None:
                ax1.set_title(title)
            for (x, y), bonds in zip(self.xy.T, self.bonds_list):
                for n_atom in bonds:
                    xb, yb = self.xy.T[n_atom]
                    dx, dy = f * (xb - x), f * (yb - y)
                    ax1.plot([x, x+dx], [y, y+dy], '-')
            self.update_surface_energies() # and gradients ### NEEDED?
            self.update_bond_energies()  # and gradients ### NEEDED?
            if show_arrows:
                for (x, y), (gx, gy) in zip(self.xy.T, self.grad_bonds.T):
                    ax1.arrow(x, y, gx, gy, width=arrow_width, color='k')
                for (x, y), (gx, gy) in zip(self.xy.T, self.grad_surf.T):
                    ax1.arrow(x, y, gx, gy, width=arrow_width, color='r')

            # relaxed configuration plot (right)
            im2 = ax2.imshow(self.E_plot, origin='lower', extent=extent)
            if show_colorbars:
                clb2 = fig.colorbar(im2, ax=ax2)
            x, y = self.xy
            ax2.scatter(x, y, marker=marker, linewidth=0, c='k', s=marker_size)
            x, y = self.xy_final
            ax2.scatter(x, y, marker=marker, linewidth=0, c='r', s=marker_size)
            for x, y in self.trajectories:
                plt.plot(x, y)
            if title != None:
                ax2.set_title(title + 'relaxed')
            plt.show()
        else:
            fig, ax = plt.subplots(1, 1)
            im = ax.imshow(self.E_plot, origin='lower', extent=extent)
            if show_colorbars:
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
                for (x, y), (gx, gy) in zip(self.xy.T, self.grad_bonds.T):
                    ax.arrow(x, y, gx, gy, width=arrow_width, color='k')
                for (x, y), (gx, gy) in zip(self.xy.T, self.grad_surf.T):
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

