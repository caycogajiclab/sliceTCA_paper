from figures.Fig2.code.hand_decoding import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as grid
import scipy
import seaborn as sns
import os


def import_models_data():
    """
    :return: dict
    Keys:
    raw : raw data
    0120: sliceTCA models with trial=0, neuron=12, time=0 slicing components
    1200, 1201, 0120 : other sliceTCA models
    12 : 12-components TCA model
    lfads : LFADS model

    Each model has a 'reconstruction' (time, trial neuron) and 'components' key. lfads and raw components are None.
    """

    return np.load('../files/models.npy', allow_pickle=True).flat[0]


def import_movement():
    """
    :return: ndarray (time, trial, x-y)
    """

    return np.load('../files/movement.npy')


def import_target_pos():
    """
    :return: ndarray (trial, x-y)
    """

    return np.load('../files/target_pos.npy')


def import_target_angle():
    """
    :return: ndarray (trial,)
    """

    return np.load('../files/target_angle.npy')


def import_condition():
    """
    :return: ndarray (trial,)
    """

    return np.load('../files/condition.npy')


def import_decoding_r2():
    """
    :return:  dict of the R^2 decoding performance over different seeds.
    Keys:
    1200 : trial=12, neuron=0, time=0 slicing components.
    1201 : trial=12, neuron=0, time=1
    lfads : LFADS

    Each model has a straight and curved key.
    """

    return np.load('../files/decoding_r2.npy', allow_pickle=True).flat[0]


def get_movement_avg(data, movement, angles, split=-1):

    unique_angles = np.unique(angles)
    avg_data, avg_movement = np.stack([np.mean(data[:,:split][:,angles[:split]==i], axis=1) for i in unique_angles], axis=1), \
                             np.stack([np.mean(movement[:,:split][:,angles[:split]==i], axis=1) for i in unique_angles], axis=1)
    avg_estimated_movement, decoder_avg = functions.velocity_decoding(avg_data, avg_movement)

    avg_data_test, avg_movement_test = np.stack([np.mean(data[:,split:][:,angles[split:]==i], axis=1) for i in unique_angles], axis=1), \
                             np.stack([np.mean(movement[:,split:][:,angles[split:]==i], axis=1) for i in unique_angles], axis=1)

    estimated_movement_avg_test = decoder_avg(avg_data_test, avg_movement_test[0])

    return avg_movement, avg_estimated_movement, avg_movement_test, estimated_movement_avg_test


def get_r2(data, movement, angles, sample_size=20):

    nb_trials_over_2 = data.shape[1] // 2

    r2s = []
    for i in range(sample_size):
        permutation = np.random.permutation(data.shape[1])
        data, movement, angles = data[:,permutation], movement[:,permutation], angles[permutation]

        _, _, avg_movement_test, estimated_movement_avg_test = get_movement_avg(data, movement, angles, nb_trials_over_2)
        r2_fold1, _ = functions.trial_wise_r2(avg_movement_test, estimated_movement_avg_test, mean=False)

        _, _, avg_movement_test, estimated_movement_avg_test = get_movement_avg(np.flip(data, axis=1), np.flip(movement, axis=1),
                                                                        np.flip(angles, axis=0), nb_trials_over_2)
        r2_fold2, _ = functions.trial_wise_r2(avg_movement_test, estimated_movement_avg_test, mean=False)

        r2s.append([r2_fold1, r2_fold2])

    return np.array(r2s)


def get_r2_all(data, movement, angles, sample_size=20):
    trial_start_curved, trial_end_curved = 265, 511
    trial_start_straight, trial_end_straight = 0, 265
    time_start, time_end = 55, -5

    r2_straight = get_r2(data[time_start:time_end-10, trial_start_straight:trial_end_straight],
                                    movement[time_start+10:time_end, trial_start_straight:trial_end_straight],
                                    angles[trial_start_straight:trial_end_straight], sample_size=sample_size)
    r2_curved = get_r2(data[time_start:time_end-10, trial_start_curved:trial_end_curved],
                                        movement[time_start+10:time_end, trial_start_curved:trial_end_curved],
                                        angles[trial_start_curved:trial_end_curved], sample_size=sample_size)

    return r2_straight, r2_curved


def plot_movement(data, movement, target_angles, curved=True):
    """
    Plots the movement, rescaled along the x-axis.
    """

    if curved:
        trial_start, trial_end = 265, 511
    else:
        trial_start, trial_end = 0, 265
    time_start, time_end = 55, -5

    movement_temp = movement[time_start+10:time_end, trial_start:trial_end]

    data_temp = {}

    for t in data.keys():
        data_temp[t] = {}
        data_temp[t]['reconstruction'] = data[t]['reconstruction'][time_start: time_end - 10, trial_start: trial_end]

    target_angles = target_angles[trial_start:trial_end]

    estimated_movements = [functions.velocity_decoding(data_temp[t]['reconstruction'], movement_temp)[0] for t in data_temp.keys()]

    avg_movement, estimated_avg_movement, avg_movement_test, estimated_movement_avg_test = get_movement_avg(data_temp['raw']['reconstruction'], movement_temp, target_angles, split=130)

    movements = [movement_temp, estimated_avg_movement]+estimated_movements
    angles = [target_angles, np.unique(target_angles)] + [target_angles for i in range(len(data))]

    for m in movements:
        m[...,0] *= 1.5/2
        m = m[:-5]

    fig = plt.figure(figsize=((len(data)+2)*3, 3), constrained_layout=True)
    axs = [fig.add_subplot(*[1,len(data)+2,i]) for i in range(1,len(data)+3)]
    plotting.plot_multiple_hand_movements(axs, movements, angles, alpha=0.6)

    avg_movement = avg_movement[:-5]
    avg_movement[...,0]  *= 1.5/2
    plotting.plot_hand_movement(axs[1], avg_movement, np.unique(angles[0]), alpha=0.6, linestyle='--', set_ax_lim=False)

    plt.savefig(plot_directory+'/hand_decoding_curved2_'+str(curved)+'.pdf')


def sort_by_condition(data, condition):
    trial_start_curved, trial_end_curved = 265, 511
    trial_start_straight, trial_end_straight = 0, 265
    data_straight = data[:, trial_start_straight:trial_end_straight]
    data_curved = data[:, trial_start_curved:trial_end_curved]
    condition_straight = condition[trial_start_straight:trial_end_straight]
    condition_curved = condition[trial_start_curved:trial_end_curved]

    cond_sorted_straight = [20, 22, 10, 34, 18, 39, 32, 5, 37, 35, 26, 31]
    cond_sorted_curved = [20, 22, 10, 34, 18, 39, 32, 5, 37, 35, 26, 31]

    data_straight_sorted = [data_straight[:,condition_straight==i] for i in cond_sorted_straight]
    data_curved_sorted = [data_curved[:,condition_curved==i] for i in cond_sorted_curved]

    cuts_straight = np.cumsum(np.array([i.shape[1] for i in data_straight_sorted]))
    cuts_curved = np.cumsum(np.array([i.shape[1] for i in data_curved_sorted]))

    return np.concatenate(data_straight_sorted+data_curved_sorted, axis=1), cuts_straight, cuts_curved


def wilcoxon(decoding_r2):

    print('sliceTCA1200 straight:', decoding_r2['1200']['straight'].mean(),
          'sliceTCA1201 straight:', decoding_r2['1201']['straight'].mean(),
          'LFADS straight:', decoding_r2['lfads']['straight'].mean())
    print('sliceTCA1200 curved:', decoding_r2['1200']['curved'].mean(),
          'sliceTCA1201 curved:', decoding_r2['1201']['curved'].mean(),
          'LFADS curved:', decoding_r2['lfads']['curved'].mean())

    print('p-value straight - Wilcoxon (LFADS vs 1201):', scipy.stats.wilcoxon(decoding_r2['lfads']['straight'], decoding_r2['1201']['straight']))
    print('p-value curved - Wilcoxon (LFADS vs 1201):', scipy.stats.wilcoxon(decoding_r2['lfads']['curved'], decoding_r2['1201']['curved']))

    print('p-value straight - Wilcoxon (1200 vs 1201):', scipy.stats.wilcoxon(decoding_r2['1200']['straight'], decoding_r2['1201']['straight']))
    print('p-value curved - Wilcoxon (1200 vs 1201):', scipy.stats.wilcoxon(decoding_r2['1200']['curved'], decoding_r2['1201']['curved']))


def plot_correlation(data, condition, M1=True, PMd=True):

    if PMd and M1:
        slice = data['1201']['components'][2][1][0][:,:]
    elif PMd:
        slice = data['1201']['components'][2][1][0][:,92:]
    else:
        slice = data['1201']['components'][2][1][0][:,:92]

    vmin = np.percentile(np.abs(np.corrcoef(slice)), 5)
    vmax = np.percentile(np.abs(np.corrcoef(slice)), 95)

    trial_start_curved, trial_end_curved = 265, 511
    trial_start_straight, trial_end_straight = 0, 265
    slice_sorted, cuts_straight, cuts_curved = sort_by_condition(slice[np.newaxis], condition)
    correlation_matrix = np.corrcoef(slice_sorted[0])

    correlation_matrix_straight_straight = correlation_matrix[trial_start_straight:trial_end_straight, trial_start_straight:trial_end_straight]
    correlation_matrix_curved_curved = correlation_matrix[trial_start_curved:trial_end_curved, trial_start_curved:trial_end_curved]
    correlation_matrix_curved_straight = correlation_matrix[trial_start_curved:trial_end_curved, trial_start_straight:trial_end_straight]

    fig = plt.figure(figsize=(16, 5))
    axs = [fig.add_subplot(1,3,i) for i in range(1,4)]
    plotting.plot_correlation_matrix(axs[0], fig, correlation_matrix_straight_straight, vmin=vmin, vmax=vmax)
    axs[0].set_xticks(cuts_straight)
    axs[0].set_yticks(cuts_straight)

    plotting.plot_correlation_matrix(axs[1], fig, correlation_matrix_curved_curved, vmin=vmin, vmax=vmax)
    axs[1].set_xticks(cuts_curved)
    axs[1].set_yticks(cuts_curved)

    plotting.plot_correlation_matrix(axs[2], fig, correlation_matrix_curved_straight, vmin=vmin, vmax=vmax)
    axs[2].set_xticks(cuts_straight)
    axs[2].set_yticks(cuts_curved)

    plt.savefig(plot_directory+'/correlation_m1'+str(M1)+'_pmd'+str(PMd)+'.pdf')


def plot_k_fold(data, movement, angles):
    """
    Plots the k-fold (and q permutations) cross-validated decoding performance for each model.
    """

    trial_start_curved, trial_end_curved = 265, 511
    trial_start_straight, trial_end_straight = 0, 265
    time_start, time_end = 55, -5

    r2s_straight = np.array([functions.k_fold_cross_validated_r2(data[k]['reconstruction'][time_start:time_end - 10, trial_start_straight:trial_end_straight],
            movement[time_start+10:time_end,trial_start_straight:trial_end_straight], functions.velocity_decoding, folds=5, sample_size=20, mean=False, seed=0)[0]
                             for k in data.keys()])

    r2s_curved = np.array([functions.k_fold_cross_validated_r2(data[k]['reconstruction'][time_start:time_end - 10, trial_start_curved:trial_end_curved],
            movement[time_start+10:time_end,trial_start_curved:trial_end_curved], functions.velocity_decoding, folds=5, sample_size=20, mean=False, seed=0)[0]
                           for k in data.keys()])

    r2s = list(map(list, zip(*[r2s_straight, r2s_curved])))

    r2_s, r2_c = get_r2_all(data['raw']['reconstruction'], movement, angles, sample_size=20)

    r2s = r2s[0:1] + [[r2_s, r2_c]] + r2s[1:]

    labels = ['raw'] + ['avg'] + [i for i in list(data.keys())[1:]] #
    bar_colors = np.array([(117, 47, 184), (254, 199, 105)])/255

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    plotting.plot_bar_dots_per_condition(ax, r2s, label_model=labels, label_condition=['straight', 'curved'], colors=bar_colors, dots=True)
    ax.set_yticks([0.1*i for i in range(-3,10)])

    plt.savefig(plot_directory+'/bar_r2.pdf')


def plot_movement_condition(movement, condition, angle, curved=True):
    """
    Plots the movement per condition.
    """

    trial_start_curved, trial_end_curved = 265, 511
    trial_start_straight, trial_end_straight = 0, 265

    if not curved:
        movement = movement[:, trial_start_straight:trial_end_straight]
        condition = condition[trial_start_straight:trial_end_straight]
        angle = angle[trial_start_straight:trial_end_straight]

    else:
        movement = movement[:, trial_start_curved:trial_end_curved]
        condition = condition[trial_start_curved:trial_end_curved]
        angle = angle[trial_start_curved:trial_end_curved]

    max_x, max_y = np.max(np.abs(movement[...,0])), np.max(np.abs(movement[...,1]))

    unique_conditions = np.unique(condition)
    movement_per_condition = [movement[:,condition==i] for i in unique_conditions]
    angle_per_condition = [angle[condition==i][0] for i in unique_conditions]

    fig = plt.figure(figsize=(2*len(unique_conditions), 2.2), constrained_layout=True)
    axs = [fig.add_subplot(1,len(unique_conditions),i+1) for i in range(len(unique_conditions))]

    cmap = matplotlib.colormaps['gist_rainbow']
    for i, m in enumerate(movement_per_condition):

        axs[i].plot(m[...,0], m[...,1], color=cmap(angle_per_condition[i]/2/np.pi))
        axs[i].set_title(unique_conditions[i])

        axs[i].set_xlim(-max_x*1.1, max_x*1.1)
        axs[i].set_ylim(-max_y*1.1, max_y*1.1)

    plt.savefig(plot_directory+'/condition_curved_'+str(curved)+'.pdf')


def plot_lda(data, condition, target_pos, angle, orth=False):
    """
    Linear discriminant analysis.
    """

    trial_start_curved, trial_end_curved = 265, 511

    target_pos = target_pos - target_pos.mean(axis=0)

    trial_start, trial_end = trial_start_curved, trial_end_curved

    cond_CCW = [5, 18, 20, 22, 26, 35, 39]
    slice = data['1201']['components'][2][1][0][trial_start:trial_end]
    target_pos = target_pos[trial_start:trial_end]
    angle = angle[trial_start:trial_end]
    condition = np.array([0 if condition[i] in cond_CCW else 1 for i in range(trial_start, trial_end)])

    if orth:
        decoded_initial_state = slice @ scipy.linalg.orth(analysis_functions.lda_pos(slice, condition, target_pos).T)
    else:
        decoded_initial_state = slice @ analysis_functions.lda_pos(slice, condition, target_pos).T

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection='3d')
    plotting.plot_lda_pos(ax, decoded_initial_state, condition, angle / 2 / np.pi)
    ax.view_init(elev=25, azim=70)

    plt.savefig(plot_directory+'/lda.pdf')


def plot_target_decoding(data, target_pos, angle, curved=True, ax=None):
    """
    Decodes the position of the target and plots it.
    """

    trial_start_curved, trial_end_curved = 265, 511
    trial_start_straight, trial_end_straight = 0, 265

    target_pos = target_pos - target_pos.mean(axis=0)
    slice = data['1201']['components'][2][1][0]
    if not curved:
        slice = slice[trial_start_straight:trial_end_straight]
        target_pos = target_pos[trial_start_straight:trial_end_straight]
        angle = angle[trial_start_straight:trial_end_straight]

    else:
        slice = slice[trial_start_curved:trial_end_curved]
        target_pos = target_pos[trial_start_curved:trial_end_curved]
        angle = angle[trial_start_curved:trial_end_curved]

    decoded_target_pos, _ = functions.position_decoding(slice[np.newaxis], target_pos[np.newaxis])
    decoded_target_pos = decoded_target_pos[0]

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot()
    ax.axes.set_aspect('equal')

    unique_target_pos = np.unique(target_pos, axis=0)
    angle_unique_target = np.arctan2(unique_target_pos[...,1], unique_target_pos[...,0])+np.pi

    cmap = matplotlib.colormaps['gist_rainbow']
    ax.scatter(decoded_target_pos[...,0]* 1.5/2, decoded_target_pos[...,1], color=[cmap(i/np.pi/2) for i in angle], alpha=0.3, s=30)
    for i in range(len(unique_target_pos)):
        ax.scatter(unique_target_pos[i,0]* 1.5/2, unique_target_pos[i,1], color=cmap(angle_unique_target[i]/np.pi/2), s=100)

    ax.set_title('curved: '+str(curved))
    ax.scatter(0,0, color='black', s=50)


def plot_target_decoding_both(data, target_pos, angle):

    fig = plt.figure(figsize=(10,5))
    ax_straight = fig.add_subplot(1,2,1)
    plot_target_decoding(data, target_pos, angle, curved=False, ax=ax_straight)

    ax_curved = fig.add_subplot(1, 2, 2)
    plot_target_decoding(data, target_pos, angle, curved=True, ax=ax_curved)

    max_x = max((abs(ax_straight.get_xlim()[0]),abs(ax_straight.get_xlim()[1]),abs(ax_curved.get_xlim()[0]),abs(ax_curved.get_xlim()[1])))
    max_y = max((abs(ax_straight.get_ylim()[0]),abs(ax_straight.get_ylim()[1]),abs(ax_curved.get_ylim()[0]),abs(ax_curved.get_ylim()[1])))

    ax_straight.set_xlim(-max_x, max_x)
    ax_straight.set_ylim(-max_y, max_y)
    ax_curved.set_xlim(-max_x, max_x)
    ax_curved.set_ylim(-max_y, max_y)

    plt.savefig(plot_directory+'/target_decoding.pdf')


def plot_component(data, target_pos, colorbar=False):
    """
    Plots sliceTCA components.
    -Neurons sorted according to the first slice for trial slices.
    -Trials sorted according to condition.
    """

    trial_start_curved, trial_end_curved = 265, 511

    x, y = target_pos[:, 0], target_pos[:, 1]

    x1, x2 = x[:trial_start_curved], x[trial_start_curved:]
    y1, y2 = y[:trial_start_curved], y[trial_start_curved:]

    a1 = np.arctan2(y1, x1)
    a2 = np.arctan2(y2, x2)
    a = np.concatenate([a1, a2])

    tidx = np.concatenate([np.argsort(a[:len(a1)]), len(a1) + np.argsort(a[len(a1):])])
    angles = np.concatenate([np.sort(a[:len(a1)]), len(a1) + np.sort(a[len(a1):])])
    cuts = np.where(angles[:-1] - angles[1:] != 0)[0] + 1
    time = np.linspace(-1, 0.5, 150)

    comp = data['1201']['components']
    cmap = matplotlib.colormaps['gist_rainbow']
    colors = np.array([cmap(np.mod((i + np.pi + np.pi + np.pi) / (np.pi * 2), 1))[:3] for i in a])
    ncomp = comp[0][0].shape[0]

    fig = plt.figure(figsize=[6, ncomp * 4], dpi=300)
    gs = grid.GridSpec(ncomp * 2, 2, height_ratios=[1.5, 4] * ncomp)

    vmin, vmax = 0, .3

    part = 0
    fact = 1

    sort_by = 0

    if len(comp[part]) != 0:
        vecs = comp[part][0]
        mats = comp[part][1]

        vecmax = []
        for vi, v in enumerate(vecs):
            vecmax.append(v.max())

            ax = fig.add_subplot(gs[vi * 4])
            ax.scatter(np.arange(len(v)), fact * (v[tidx] / vecmax[vi]), lw=.3, c=colors[tidx], s=4)
            ax.plot([len(x1), len(x1)], [0, 1], 'k-', alpha=.2)
            ax.set_xlabel('Trials')
            ax.set_ylabel('Weight')
            ax.set_xlim([0, len(v)])
            ax.set_ylim(0, 1.05)
            ax.set_xticks([140, 380])
            ax.set_xticklabels(['Mo maze', 'Maze'])

        for mi, m in enumerate(mats):
            if mi == sort_by:
                t = m.copy()
                t = np.array([ti - ti.min() for ti in t])
                t = np.array([ti / ti.max() for ti in t])
                tidx2 = np.argsort(t.argmax(axis=-1))

        for mi, m in enumerate(mats):
            t = m.copy()
            t = np.array([ti - ti.min() for ti in t])
            t = np.array([ti / ti.max() for ti in t])

            ax = fig.add_subplot(gs[mi * 4 + 2])
            im = ax.imshow((t[tidx2] * vecmax[mi]) / fact, vmin=vmin, vmax=vmax, extent=[time[0], time[-1], 0, len(m)],
                      aspect='auto', cmap='magma', origin='lower')
            ax.plot([0, 0], [0, len(m)], 'w--', lw=1)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Neurons')
            if colorbar:
                plt.colorbar(im, ax=ax)

    part = 2
    fact = 1

    if len(comp[part]) != 0:
        vecs = comp[part][0]
        mats = comp[part][1]

        vecmax = []
        for vi, v in enumerate(vecs):
            vecmax.append(v.max())

            ax = fig.add_subplot(gs[vi * 4 + 1])
            ax.plot(time, fact * (v / vecmax[vi]), color='k')
            ax.set_xlabel('Time')
            ax.set_ylabel('Weight')
            ax.set_xlim(time[0], time[-1])
            ax.set_ylim(0, 1.05)

        for mi, m in enumerate(mats):
            # if mi==0:
            t = m.copy()
            t = np.array([ti - ti.min() for ti in t.T]).T
            t = np.array([ti / (2 * ti.mean()) for ti in t.T]).T

            nidx = np.argsort(np.mean(t[tidx][:cuts[0]], 0))[::-1]

            ax = fig.add_subplot(gs[mi * 4 + 3])
            im =ax.imshow((t[tidx][:, nidx] * vecmax[mi]) / fact, vmin=vmin, vmax=vmax, extent=[0, m.shape[1], 0, len(m)],
                      aspect='auto', cmap='magma', origin='lower')

            ax.plot([0, m.shape[1]], [x1.shape, x1.shape], 'w-', lw=1.5)
            ax.set_yticks([140, 380])
            ax.set_yticklabels(['No maze', 'Maze'])
            ax.set_xlabel('Neurons')
            ax.set_ylabel('Trials')
            if colorbar:
                plt.colorbar(im, ax = ax)

    plt.tight_layout()
    sns.despine()

    if colorbar:
        plt.savefig(plot_directory+'/components-colorbar.pdf')
    else:
        plt.savefig(plot_directory+'/components.pdf')


if __name__=='__main__':

    data = import_models_data()
    movement = import_movement()
    target_pos = import_target_pos()
    target_angle = import_target_angle()
    condition = import_condition()
    decoding_r2 = import_decoding_r2()

    plot_directory = './plots/'
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    #plot_movement_condition(movement, condition, target_angle, curved=True)
    #plot_lda(data, condition, target_pos, target_angle)
    plot_k_fold(data, movement, target_angle)
    #plot_correlation(data, condition, PMd=True, M1=True)
    #plot_movement(data, movement, target_angle, curved=True)
    #plot_target_decoding_both(data, target_pos, target_angle)
    #plot_component(data, target_pos)
    #wilcoxon(decoding_r2)
