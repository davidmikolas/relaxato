import numpy as np
import matplotlib.pyplot as plt
from grand_minimizer_v22 import Island, E_hexi, Island_from_HOC
import json, yaml, time #, pickle
from datetime import datetime

kwargs = {'polarity_1': 'n', 'scale': 1.0} # honeycomb with "holes"

dt_check = 4.0
n_check_max = 100 # this is pretty long
n_eval_per = 100 # this is pretty long

grad_surf_d = 1E-10
rtol = 1E-03 

v_stop = 1E-04 # is 1E-04 low enough?
# damping = 4 # 1.5 # is this okay?

### Island variables ###

nmax = 3

# E_bond_alpha = 60   # 25 meV * ((d-d0)/0.1A)^2 and 250 meV E_surfac amplitude 

E_bond_alphas = [30, 60, 120]   # 25 meV * ((d-d0)/0.1A)^2 and 250 meV E_surfac amplitude 
# E_bond_alphas = E_bond_alphas[:2] #######  QQQ

rotations = np.arange(31)
# rotations = rotations[-2:] #######  QQQ

lattice_constants = np.linspace(1.32, 1.38, 7)
# lattice_constants = lattice_constants[:2] #######

### Page variables ###

sig_xy = 0.0

# DAMPING
dampings = [2, 4, 8]
# dampings = dampings[:2] ####### QQQ

# OFFSETS
v1, v2 = np.array([[1, 0], [0.5, 3**0.5/2]])
top = np.zeros(2)
bridge = np.array([0.5, 0])
hollow = np.array([0.5, 3**0.5/6])
half_bridge = 0.5 * bridge
half_hollow = 0.5 * hollow
xy_offsets = np.vstack([top, bridge, hollow, half_bridge, half_hollow])
print('xy_offsets: ', xy_offsets)
print('xy_offsets.shape: ', xy_offsets.shape)

xy_offsets = xy_offsets[-1:] ###### QQQ


### Get ready to calculate! ###

# https://docs.python.org/3/library/datetime.html truncate to milliseconds

begin_time_string = datetime.today().strftime('%Y-%m-%d at %H.%M.%S.%f')

job_name = 'happy' + ' ' + begin_time_string

total_process_time_start = time.process_time()
total_time_start = time.time()

positionz, velocitiez = [], []
i_page = 0
n_pages = len(lattice_constants) * len(rotations) * len(E_bond_alphas)

for a in lattice_constants:
    for R in rotations:
        for E_bond_alpha in E_bond_alphas:
            # BEGIN PAGE
            summary_dict = dict()
            start_dict = dict()
            positionz, velocitiez = [], []
            page_name = ('p' + str(1000000 + i_page)[1:] +
                        '_a=' + str(round(a, 3)) +
                        '_R=' + str(round(R, 1)) +
                        '_Eba=' + str(round(E_bond_alpha, 1)))
            page_process_time_start = time.process_time()
            page_total_time_start = time.time()
            i_calculations = 0
            for damping in dampings:
                for xy_offset in xy_offsets:
                    island = Island(R=R, a=a, nmax=nmax, randomize='sq', sig_xy=sig_xy,
                                    E_surf=E_hexi, xy_offset=xy_offset, iseed=42, 
                                    E_surf_kwargs=kwargs, E_bond_alpha=E_bond_alpha,
                                    grad_surf_d=grad_surf_d)  # E_bond_alpha=30

                    island.solve_as_ivp_smart(dt_check=dt_check, n_check_max=n_check_max,
                                              v_stop=v_stop, n_eval_per=n_eval_per, damping=damping,
                                              rtol=rtol, dense_output=False) # damping=5,


                    for A, B in island.report_dict['summary'].items():
                        try:
                            summary_dict[A].append(B) # {'a': 1.32}
                        except:
                            summary_dict[A] = [B] # {'a': [1.32]} # {'a': [1.32, 1.34]}

                    for A, B in island.report_dict['start'].items():
                        try:
                            start_dict[A].append(B)
                        except:
                            start_dict[A] = [B]
             

                    positions = island.trajectory_dict['positions']
                    positionz.append(positions.copy())
                    velocities = island.trajectory_dict['velocities']
                    velocitiez.append(velocities.copy())
                    i_calculations += 1

            savename = job_name + '_' + page_name
            # save trajectory arrays as .npz (zipped .npy files)
            np.savez(savename + '_pos_.npz', *positionz)
            np.savez(savename + '_vel_.npz', *velocitiez)

            page_process_time = (time.process_time() -
                                  page_process_time_start)
            page_total_time = (time.process_time() -
                                  page_process_time_start)
            message = (str(i_page+1) + ' of ' + str(n_pages) + ' with ' +
                       str(i_calculations) + ' calculations: process time: ' +
                       str(round(page_process_time, 3)) + ' sec.')
            print(message)
            
            with open(savename + '_start_.json', 'w') as outfile:
                json.dump(start_dict, outfile)
                
            with open(savename + '_summary_.json', 'w') as outfile:
                json.dump(summary_dict, outfile)

            print('results saved under: ', savename)

            i_page += 1


# pip install Send2Trash
# from send2trash import send2trash
# send2trash('qwerty.txt')

if True:
    for ps, vs in zip(positionz, velocitiez):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for x, y in ps:
            ax1.plot(x, y)
        for vx, vy in vs:
            ax2.plot(np.sqrt(vx**2 + vy**2))
        plt.show()

if True:
    R_final = summary_dict['angles_final_mean']
    R_final = np.array(R_final).reshape(len(lattice_constants), -1)
    R_initial = start_dict['R']
    R_initial = np.array(R_initial).reshape(len(lattice_constants), -1)
    if True:
        plt.figure()
        for Ri, Rf in zip(R_initial, R_final):
            plt.plot(Ri, Rf, '-')
        plt.show()

"""
# sadly yaml doesn't work for anything that even smells of numpy
with open('start_dict ' + job_name + '.yaml', 'w') as outfile:
    yaml.dump(start_dict, outfile)
with open('summary_dict ' + job_name + '.yaml', 'w') as outfile:
    yaml.dump(summary_dict, outfile)
"""

if False:
    for vs in velocities:
        fig, ax = plt.subplots(1, 1, figsize=[14, 7.5])
        speeds = np.sqrt((vs**2).sum(axis=1))
        for speed in speeds:
            ax.plot(speed, linewidth=0.5)
        plt.show()


"""
    results_dicts.append(island.report_dict)
    
    # island.show(show_arrows=True, marker_size=10, show_final=True, border=5, figsize=[12, 8]) # 0.5

    xys = np.swapaxes(island.trajectories, 2, 0)
    vxys = np.swapaxes(island.velocities, 2, 0)
    bes = np.array([island.get_bond_energies(xy).sum() for xy in xys])
    ses = np.array([island.get_surface_energies(xy).sum() for xy in xys])
    kes = np.array([0.5 * (vxy**2).sum() for vxy in vxys])
    tes = bes + ses + kes
    t_eval = island.t_eval
    plt.plot(t_eval, bes)
    plt.plot(t_eval, ses)
    plt.plot(t_eval, kes)
    plt.plot(t_eval, tes, '--')
    plt.show()

    island.update_bond_energies()
    island.update_surface_energies()
    be, se = island.bond_energy, island.surface_energy
    print('initial energy: ', round(be, 3), round(se, 3), round(be + se, 3))
    bef = island.get_bond_energies(island.xy_final).sum()
    sef = island.get_surface_energies(island.xy_final).sum()
    print('final energy:   ', round(bef, 3), round(sef, 3), round(bef + sef, 3))
    print('')
    print('initial mean bond: ', round(island.get_bond_distances(island.xy).mean(), 3))
    print('final mean bond:   ', round(island.get_bond_distances(island.xy_final).mean(), 3))
    print('')

    if True:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[12, 5])
        vsqs = []
        for vxy in island.velocities:
            vsq = (vxy**2).sum(axis=0)
            vsqs.append(vsq)
            v = np.sqrt(vsq)
            ax1.plot(v, linewidth=0.5)
            ax2.plot(v, linewidth=0.5)
        ax2.set_yscale('log')
        ax2.set_ylim(1E-07, None)
        E = sum(vsqs)
        ax3.plot(E)
        ax3.set_yscale('log')
        ax3.set_ylim(1E-14, None)
        ax1.set_title('|v|')
        ax2.set_title('|v|')
        ax3.set_title('sum(|v|^2)')
        plt.show()

with open(pickle_filename, 'wb') as outfile:
    pickle.dump(results_dicts, outfile)

rd = island.report_dict
print('rd.keys(): ', rd.keys())
sd = rd['summary']
print('sd.keys(): ', sd.keys())

# angles, mean_angle, weighted_mean_angle = island.get_angles()
# print('mean_angle, weighted_mean_angle: ', mean_angle, weighted_mean_angle) 

if False:
    ijkl = (3, 4, 5, 2)
    ijkl = (3, 1, 2, 1)
    i3 = Island_from_HOC(ijkl, E_surf=E_hexi, E_surf_kwargs=kwargs,
                           E_bond_alpha=1)

    bl, bhl = i3.bonds_list, i3.bonds_hoc_list
    print(bl[0])
    for n, (x, y) in bhl[0]:
        print(n, round(x, 2), round(y, 2))
    fig, axes = plt.subplots(2, 2)
    titles = ('internal bonds', 'boundary bonds', 'boundary bond targets', 'proxies')
    axes = axes.flatten()
    ax1, ax2, ax3, ax4 = axes
    ms1, ms2 = 12, 6
    lw0, lw1, lw2 = 0.5, 1.0, 1.5
    grayish = [0.7, 0.7, 0.7, 1]
    x, y = i3.xy
    # plot atoms in both
    for ax, title in zip(axes, titles):
        _ = ax.plot(x, y, 'ok', ms=ms1)
        ax.set_aspect('equal')
        ax.set_title(title)
    # now plot eight more!
    for ax in axes:
        for ij in i3.bigij:
            xy_off = (ij[:, None] * i3.vecs_rot).sum(axis=0)
            x, y = i3.xy + xy_off[:, None]
            _ = ax.plot(x, y, 'o', ms=ms2)
    # and enumerate them:
    xy_off = np.array([0.25, 0.05])
    for i, xy in enumerate(i3.xy.T):
        _ = ax1.annotate(str(i), xy + xy_off)
    # plot normal bonds on the left
    for i, (xc, yc) in enumerate(i3.xy.T):
        for n in i3.bonds_list[i]:
            xb, yb = i3.xy[:, n]
            _ = ax1.plot([xc, xb], [yc, yb], '-k', lw=lw1)
    # plot HOC bonds on the right
    for i, (xc, yc) in enumerate(i3.xy.T):
        for n, (xo, yo) in i3.bonds_hoc_list[i]:
            xa, ya = i3.xy[:, n]
            xb, yb = xa - xo, ya - yo
            if i == 0:
                _ = ax2.plot([xc, xb], [yc, yb], '-r', lw=lw2)
                _ = ax3.plot([xc, xa], [yc, ya], '-r', lw=lw2)
                _ = ax4.plot([xb, xa], [yb, ya], '-r', lw=lw2)
            else:
                _ = ax2.plot([xc, xb], [yc, yb], '-', color=grayish, lw=lw0)
                _ = ax3.plot([xc, xa], [yc, ya], '-', color=grayish, lw=lw0)
                _ = ax4.plot([xb, xa], [yb, ya], '-', color=grayish, lw=lw0)
    for ax in axes[1:]:
        _ = ax.set_xlim(*ax1.get_xlim())
        _ = ax.set_ylim(*ax1.get_ylim())
        ax.set_aspect('equal')
    plt.show()


d = island.report_dict

# wow = [a

# 3, 1, 5, 4, 6, 2



if False:
    hw = 10.
    N = 501
    xyp = np.mgrid[-hw:hw:N*1j, -hw:hw:N*1j]
    A = E_hexi(xyp, **kwargs)
    kwargs_mod = {'polarity_1': 'p', 'scale': 6.0}
    B = E_hexi(xyp, **kwargs_mod)
    C = A * B
    fig, axes = plt.subplots(1, 3)
    extent = [-hw, hw, -hw, hw]
    for ax, thing in zip(axes, (A, B, C)):
        ax.imshow(thing, origin='lower', extent=extent)
    plt.show()


    island.update_bond_energies()
    island.update_surface_energies()
    print('initial bond energy: ', island.bond_energy)
    print('initial surface energy: ', island.surface_energy)
    print('initial energy: ', island.bond_energy + island.surface_energy)
    print('initial mean bond: ', island.get_bond_distances(island.xy).mean())

    t_eval = np.linspace(0, 20, 401)

    island.solve_as_ivp(t_eval=t_eval, damping=3, dense_output=True, tol=1E-06) # damping=5,

    final_bond_energy = island.get_bond_energies(island.xy_final).sum()
    final_surface_energy = island.get_surface_energies(island.xy_final).sum()
    print('')
    print('final bond energy: ', final_bond_energy)
    print('final surface energy: ', final_surface_energy)
    print('final energy: ', final_bond_energy + final_surface_energy)
    print('final mean bond: ', island.get_bond_distances(island.xy_final).mean())


    # xy_final_solve_as_ivp = island.xy_final


if False:
    island.show(show_arrows=True, marker_size=10, show_final=True, border=5, figsize=[12, 8]) # 0.5

if False:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    vsqs = []
    for vxy in island.velocities:
        vsq = (vxy**2).sum(axis=0)
        vsqs.append(vsq)
        v = np.sqrt(vsq)
        ax1.plot(v, linewidth=0.5)
        ax2.plot(v, linewidth=0.5)
    ax2.set_yscale('log')
    E = sum(vsqs)
    ax3.plot(E)
    ax3.set_yscale('log')
    ax1.set_title('|v|')
    ax2.set_title('|v|')
    ax3.set_title('sum(|v|^2)')
    plt.show()




if False:
    ijkl = (7, 1, 8, 1)
    i1 = Island_from_HOC(ijkl, E_surf=E_hexi, E_surf_kwargs=kwargs,
                           E_bond_alpha=1)

    ijkl = (7, 0, 3, 3)
    i2 = Island_from_HOC(ijkl, E_surf=E_hexi, E_surf_kwargs=kwargs,
                           E_bond_alpha=1)

    ijkl = (3, 4, 5, 2)
    i3 = Island_from_HOC(ijkl, E_surf=E_hexi, E_surf_kwargs=kwargs,
                           E_bond_alpha=1)

    ijkl = (4, 1, 3, 1)
    i4 = Island_from_HOC(ijkl, E_surf=E_hexi, E_surf_kwargs=kwargs,
                           E_bond_alpha=1)

    print(i4.ij.T.tolist())

    for i, (b, bh) in enumerate(zip(i4.bonds_list, i4.bonds_hoc_list)):
        print(i, b, bh)
        print('')

    i4.show(annotate=True)

    # islands = (i1, i2, i3)
    islands = (i4, i4)
    fig, axes = plt.subplots(1, len(islands))
    for ax, isl in zip(axes, islands):
        x0, y0 = isl.xy
        ax.plot(x0, y0, 'ok', ms=6)
        ax.set_aspect('equal')
        for ij in isl.bigij:
            xy_off = (ij[..., None] * isl.vecs_rot).sum(axis=0)
            x, y = isl.xy + xy_off[:, None]
            ax.plot(x, y, 'o', ms=4)        
    plt.show()

if False:
    island.show(show_arrows=True, marker_size=10, show_final=True, border=5, figsize=[12, 8]) # 0.5

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    vsqs = []
    for vxy in island.velocities:
        vsq = (vxy**2).sum(axis=0)
        vsqs.append(vsq)
        v = np.sqrt(vsq)
        ax1.plot(v, linewidth=0.5)
        ax2.plot(v, linewidth=0.5)
    ax2.set_yscale('log')
    E = sum(vsqs)
    ax3.plot(E)
    ax3.set_yscale('log')
    ax1.set_title('|v|')
    ax2.set_title('|v|')
    ax3.set_title('sum(|v|^2)')
    plt.show()

# print('Now use minimize')

#island.solve_minimize(tol=1E-03, method='Nelder-Mead')

# island.show(show_arrows=True, marker_size=10, show_final=True, border=5, figsize=[12, 8]) # 0.5

if False:
    import cProfile
    import pstats
    import sys

    def profile2():
        pr = cProfile.Profile()
        pr.enable()
        # function(int(sys.argv[1]))
        island.get_grad_bonds(island.xy)
        pr.disable()
        return pstats.Stats(pr)

    profile2().dump_stats('bob.txt')

    loop = False

    if loop:
        results = []
        rotations = np.linspace(0, 60, 61)
        for R in rotations:
            island = Island(R=R, a=1.30, nmax=5, E_surf=E_hexi,
                            E_surf_kwargs=kwargs, E_bond_alpha=30)
            island.solve_as_ivp(t_eval=t_eval, damping=1)

            E_initial, E_final = [island.get_total_energy(xy) for xy in
                                  (island.xy, island.xy_final)]
            angles, mean_angle, weighted_mean_angle = island.get_angles()
            results.append([E_initial, E_final, weighted_mean_angle])
            print(R, )

        things = [np.array(thing) for thing in zip(*results)]
        E_initial, E_final, weighted_mean_angle = things

    if loop:
        plt.figure()
        plt.plot(rotations, E_initial)
        plt.plot(rotations, E_final)
        plt.plot(rotations, weighted_mean_angle)
        plt.show()


"""

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



