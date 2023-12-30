# standard packages
import numpy as np
import scipy.stats as sps

# data/path handling
from glob import glob
import pickle

# plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import seaborn as sns


################################################################################################################
# load stuff
################################################################################################################

# load sliceTCA components after hierarchical model optimization
with open(glob('../files/wagner-sliceTCA-3-3-0-processed-components.p')[0], 'rb') as f: 
    comp = pickle.load(f)

# load raw data and time stamps
cbl = np.load('../files/cbl_data.npy')
ctx = np.load('../files/ctx_data.npy')
ts = np.load('../files/timestamps.npy')

# load behavioral data frame
with open(glob('../files/behavior.p')[0], 'rb') as f: 
    df = pickle.load(f)


################################################################################################################
# statistical tests
################################################################################################################

# test activity diff between error and correct trials in trial-time slices
for c in comp[1][1]:
    print(sps.mannwhitneyu(np.mean(c[df.outcome.values==2][:,sum(ts):], -1), 
                           np.mean(c[df.outcome.values!=2][:,sum(ts):], -1)))


################################################################################################################
# plot components
################################################################################################################

# define trial colors
colors = ['darkred', 'royalblue', 'darkgrey']

# define time values
time = (np.arange(150) - ts[0]) * 0.033 # 150 samples, 0.033 sampling rate


# Figure specifications
fig = plt.figure(figsize=[8, 3*3.5])
gs = grid.GridSpec(3*2, 3, height_ratios=[1.5, 3]*3)

# Trial slicing components
part = 0

vecs = comp[part][0]
mats = comp[part][1]

facts = [np.abs(c).max() for c in vecs]

for vi,v in enumerate(vecs):
    v = v/facts[vi]
    if vi==2: v *= -1

    ax = fig.add_subplot(gs[vi*3*2])

    # plot scatter plot color-coded by trial outcome
    ax.scatter(np.arange(len(v)), v, lw=.3, 
               color=np.array([colors[k] for k in df.outcome.values]), s=3)

    # for legend
    plt.plot(-1,0, '.', color=colors[0], label='Left')
    plt.plot(-1,0, '.', color=colors[1], label='Right')
    plt.plot(-1,0, '.', color=colors[2], label='Error')

    ax.set_xlabel('Trials')
    ax.set_ylabel('Weight')
    plt.legend(frameon=False, fontsize=6)
    plt.xlim([0,len(v)])

for mi,m in enumerate(mats):
    if mi==2: m *= -1 # switch sign

    # for first slice, get indices for peak latency for cbl and ctx
    if mi==0: 
        cbl_idx = np.argsort(np.argmax(m[:cbl.shape[0]],1))
        ctx_idx = np.argsort(np.argmax(m[cbl.shape[0]:],1))

for mi,m in enumerate(mats):
    m = m*facts[mi]

    # separate slice into cbl and ctx and sort by peak latency in the 1st slice
    cbl_s = m[:cbl.shape[0]][cbl_idx]
    ctx_s = m[cbl.shape[0]:][ctx_idx]

    ax = fig.add_subplot(gs[mi*3*2+3])

    # plot sorted slices
    ax.imshow(np.concatenate([cbl_s, ctx_s]), 
        vmin=-1, vmax=1, 
        extent=[time[0], time[-1], 0, len(m)], 
        aspect='auto', cmap='twilight', origin='lower')

    # plot line between cbl and ctx
    ax.plot([time[0],time[-1]],[cbl.shape[0], cbl.shape[0]], 'w-', lw=1)

    # plot line for time stamps
    for t in range(len(ts)):
        ax.plot([time[np.sum(ts[:t+1])], time[np.sum(ts[:t+1])]], 
                [0, m.shape[0]], 
                'w--', lw=1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neurons')
    ax.set_yticks([cbl.shape[0]/2, cbl.shape[0]+ctx.shape[0]/2])
    ax.set_yticklabels(['Cbl', 'Ctx'])
    plt.xlim([time[0], time[-1]])
    plt.ylim([0,m.shape[0]])


# neuron slicing components 
part = 1
vecs = comp[part][0]
mats = comp[part][1]

facts = [np.abs(c).max() for c in vecs]

for vi,v in enumerate(vecs):
    v = v/facts[vi]
    if vi<2: v *= -1 # switch sign

    ax = fig.add_subplot(gs[vi*3*2+1])

    # bar plots for neuron weights
    ax.bar(np.arange(len(v)), v, color='k')

    # line to separate cbl and ctx
    ax.plot([cbl.shape[0], cbl.shape[0]], [-1,1], 'k-')

    ax.set_xticks([cbl.shape[0]/2, cbl.shape[0]+ctx.shape[0]/2])
    ax.set_xticklabels(['Cbl', 'Ctx'])
    ax.set_xlabel('Neurons')
    ax.set_ylabel('Weight')
    plt.xlim([0,len(v)])

for mi,m in enumerate(mats):
    m = m*facts[mi]
    if mi<2: m *= -1 # switch sign

    ax = fig.add_subplot(gs[mi*3*2+4])

    # sort slice by corr/err, left/right, trial number
    dfs = df.sort_values(['cor','dir','trial'], ascending=[False, True, True])
    
    # find indices to draw lines at
    lines = [np.where(dfs.dir=='r')[0][0], 
             np.where(dfs.cor==False)[0][0],
             np.where((dfs.dir=='r')&(dfs.cor==False))[0][0],
             np.where((dfs.dir=='r')&(dfs.cor==False))[0][-1]+2]

    # plot matrices
    ax.imshow(m[dfs.trial.values], vmin=-1, vmax=1, 
        extent=[time[0], time[-1], 0, len(m)], 
        aspect='auto', cmap='twilight', origin='lower')
    
    # draw lines for trial classes
    for l in lines:
        ax.plot([time[0],time[-1]],[l, l], 'w-', lw=1)

    # draw lines at time stamps
    for t in range(len(ts)):
        ax.plot([time[np.sum(ts[:t+1])], time[np.sum(ts[:t+1])]], 
                [0, m.shape[0]], 'w--', lw=1)

    ax.set_yticks([lines[0]/2, lines[1]-(lines[1]-lines[0])/2, 
                   lines[2]-(lines[2]-lines[1])/2, lines[3]-(lines[3]-lines[2])/2])
    ax.set_yticklabels(['Left, corr', 'Right, corr', 'Left, err', 'Right, err'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trials')
    plt.xlim([time[0], time[-1]])
    plt.ylim([0,m.shape[0]])

sns.despine()
plt.tight_layout()

plt.show()
