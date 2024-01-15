import matplotlib
import numpy as np
import matplotlib.patches as mpatches


def plot_hand_movement(ax, movement, angle, linestyle='-', alpha=1.0, linewidth=1.0, set_ax_lim=True):
    """
    Plots hand movement with trajectories colored as angle
    in:
        movement (ndarray): time x trial x XY
        angle (ndarray): len(angle) = trial, angle must be between 0 and 2pi
    """

    cmap = matplotlib.cm.get_cmap('gist_rainbow')

    for i in range(movement.shape[1]):
        ax.plot(movement[:, i, 0], movement[:, i, 1], color=cmap(angle[i]/(2*np.pi)),
                linestyle=linestyle, alpha=alpha, linewidth=linewidth)

    max_movement = np.abs(movement).max()

    if set_ax_lim:
        ax.set_xlim(-max_movement*1.05, max_movement*1.05)
        ax.set_ylim(-max_movement*1.05, max_movement*1.05)

    ax.set_aspect('equal')


def plot_multiple_hand_movements(axs, movements, angles, linestyle='-', alpha=1.0, linewidth=1.0, set_ax_lim=True):
    """
    Plots hand movement with trajectories colored as angle, puts in separate axes
    in:
        movement (ndarray): time x trial x XY
        angle (ndarray): len(angle) = trial, angle must be between 0 and 2pi
    """

    max_x = np.max(np.array([np.abs(m[...,0]).max() for m in movements]))
    max_y = np.max(np.array([np.abs(m[...,1]).max() for m in movements]))

    max_movement = max((max_x, max_y))

    for i in range(len(movements)):
        plot_hand_movement(axs[i], movements[i], angles[i],
                linestyle=linestyle, alpha=alpha, linewidth=linewidth, set_ax_lim=False)

        if set_ax_lim:
            axs[i].set_xlim(-max_movement, max_movement)
            axs[i].set_ylim(-max_movement, max_movement)


def plot_lda_pos(ax, decoded_initial_state, condition, angle, label=False):
    """
    Plots decoded initial state
    in:
        decoded_init_state (ndarray): trial x 3. The first dimension is LDA decoded condition and the other two the target pos
        condition (ndarray): trial. The condition decoded by LDA
        angle (ndarray): trial. The angle of the decoded target.
    """

    cmap = matplotlib.cm.get_cmap('gist_rainbow')

    z = decoded_initial_state[...,2].min()-(decoded_initial_state[...,2].max()-decoded_initial_state[...,2].min())*0.2

    for mi, m in enumerate(decoded_initial_state):
        if condition[mi] == 0:
            marker = '^'
        else:
            marker = 'o'
        ax.plot(m[0], m[1], m[2],
                marker, color=cmap(angle[mi]), alpha=1, ms=5)
        ax.plot(m[0], m[1], z, marker, color=cmap(angle[mi]), alpha=.2, ms=5)

    ax.set_zlim(z, decoded_initial_state[..., 2].max())

    ax.set_xlabel('condition 1 vs. 2')
    ax.set_ylabel('target dim. 1')
    ax.set_zlabel('target dim. 2')

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    if not label:
        ax.set_xticks([],[])
        ax.set_yticks([],[])
        ax.set_zticks([],[])

    ax.view_init(elev=14, azim=-114)


def plot_bar_dots_per_condition(ax, value, label_model=None, label_condition=None, colors=None, dots=False):
    """
    Plots bar per condition
    int:
        values (ndarray): #model x #condition. model is the across, condition is the within.
    """

    number_models, number_conditions = len(value), len(value[0])

    cmap = matplotlib.cm.get_cmap('pink')

    colors = [cmap(j/number_conditions) for j in range(number_conditions)] if colors is None else colors

    x = 0

    for i in range(number_models):
        x += 1
        for j in range(number_conditions):
            ax.bar(x, value[i][j].mean(), yerr=value[i][j].std(axis=-1).mean()/np.sqrt(value[i][j].shape[-1]),
                       color=colors[j], ecolor=(1, 0.2, 0.2))
            if dots:
                ax.scatter(np.full(value[i][j].shape[:-1], x)+np.random.uniform(-0.2, 0.2, value[i][j].shape[:-1]), value[i][j].mean(axis=-1),
                           color=(0, 0, 0, 0.2), edgecolor=(0, 0, 0, 0), s=20)
                ax.scatter(x, value[i][j].mean(),
                           color=(1, 0.2, 0.2), marker='_', s=50)
            x += 1

    patches = [mpatches.Patch(color=colors[j], label=label_condition[j]) for j in range(number_conditions)]
    ax.legend(handles=patches)

    if label_model is not None:
        ax.set_xticks(np.arange(0, (number_models)*(number_conditions+1), number_conditions+1)+number_conditions/2+0.5, label_model)


def plot_correlation_matrix(ax, fig, correlation_matrix, vmin=None, vmax=None):
    """
    Plots a correlation matrix.
    in:
        fig (matplotlib fig): a figure to add the color bar to the axis
        correlation_matrix: a correlation matrix
    """

    if vmin is None:
        vmin = np.percentile(np.abs(correlation_matrix), 10)
    if vmax is None:
        vmax = np.percentile(np.abs(correlation_matrix), 90)

    im = ax.imshow(correlation_matrix, cmap='gnuplot2', vmin=vmin, vmax=vmax, origin='upper', aspect='equal')

    fig.colorbar(im, ax=ax, shrink=.6, label='Correlation', ticks=[vmin, 0, vmax])
