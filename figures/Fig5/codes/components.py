# data/path handling
from glob import glob
import pickle

# standard packages
import numpy as np
import scipy as sp

# plotting
import matplotlib.colors as mcols 
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import seaborn as sns


#############################################################################################################
# load stuff
#############################################################################################################

# load sliceTCA components after hierarchical model optimization
with open(glob('../files/IBL-sliceTCA-2-3-3-components.p')[0], 'rb') as f: 
    comp = pickle.load(f)

# load behavioral data frame
with open('../files/behavior.p', 'rb') as f: 
    df = pickle.load(f)

# load region labels
with open('../files/region_labels.p', 'rb') as f: 
    idcs, labels, regions = pickle.load(f)


#############################################################################################################
# plot the main panels
#############################################################################################################

# custom colormap
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcols.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# colors for regions
allencolors = ['teal', 'yellowgreen', 'yellowgreen',  'salmon', 'violet', 'violet', 'k']

# time: extracted time series are from -1 to 2.5 s after stimulus presentation
time = np.linspace(-1, 2.5-.01, 350)

# time stamps: stimulus presentation, 
ts = [0,.49]

vmin = 0
vmax = .25

# get number of neurons for each region and prepare tickmark positions for neuron axis
lengths     = [sum(i) for i in idcs]
neuronticks = [sum(lengths[:i])+lengths[i]/2 for i in range(len(lengths))]



# figure settings
fig = plt.figure(figsize=[9, 3*4], dpi=300)
gs = grid.GridSpec(3*2, 3, height_ratios=[1.5, 4]*3)

# trial slicing
vecs = comp[0][0]
mats = comp[0][1]

vecmax = []
for vi,v in enumerate(vecs):
    # color-code first vector by correct vs error
    if vi<1:
        conditions = df.feedbackType.values
        conditions[conditions==-1] = 0
        label = 'Correct'
        cmap1 = truncate_colormap(plt.get_cmap('magma'), 0, .8)

    # color code second vector by log RT
    else:
        conditions = np.log(df.response_times.values-df.stimOn_times.values)
        conditions[conditions>1] = 1
        label = 'RT'
        cmap1 = truncate_colormap(plt.get_cmap('magma'), 0, 1)
    vecmax.append(v.max())

    # plot vectors
    ax = fig.add_subplot(gs[vi*3*2])
    ax.scatter(np.arange(len(v)), v/vecmax[vi], lw=.3, c=conditions, s=4, cmap=cmap1, label=label)
    ax.set_xlabel('Trials')
    ax.set_ylabel('Weight')
    plt.legend(frameon=False, fontsize=10)
    plt.xlim([0,len(v)])

# plot slices
for mi,m in enumerate(mats):
    # min-max scale slice for plotting
    t = m.copy()
    t = np.array([ti-ti.min() for ti in t])
    t = np.array([ti/ti.max() for ti in t])

    # sort slice by region and then by latency of the maximum activation
    idx = np.concatenate([np.where(i)[0][np.argsort(np.argmax(t[i], 1))] 
                          for i in idcs])

    ax = fig.add_subplot(gs[mi*3*2+3])
    im = ax.imshow(t[idx]*vecmax[mi], 
                   vmin=0, vmax=.25,
                   extent=[time[0], time[-1], 0, len(m)],
                   aspect='auto', cmap=truncate_colormap(plt.get_cmap('twilight'), .5, 1), 
                   origin='lower')
    
    # plot time stamps
    for t in ts:
        ax.plot([time[np.where(time>t)[0][0]], time[np.where(time>t)[0][0]]], 
                [0, m.shape[0]], 
                'w--', lw=1)

    # plot white lines separating regions
    l = 0
    for le in lengths:
        l+=le
        ax.plot([time[0], time[-1]], [l, l], 'w-', lw=1)

    ax.set_yticks(neuronticks)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neurons')

# neuron slicing
vecs = comp[1][0]
mats = comp[1][1]

vecmax = []
for vi,v in enumerate(vecs):
    vecmax.append(v.max())

    ax = fig.add_subplot(gs[vi*3*2+1])

    l = 0
    for ii,i in enumerate(idcs):
        ax.plot([l,l], [-.05,1.05], color='k', alpha=.2, lw=1)

        # sort bars by sub-regions
        bars = v[i][np.argsort(regions[i])]
        ax.bar(np.arange(len(bars))+l, bars/vecmax[vi], color=allencolors[ii], width=1)
        l+=lengths[ii]

    ax.set_xticks(neuronticks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel('Neurons')
    ax.set_ylabel('Weight')
    plt.xlim([0,len(v)])

for mi,m in enumerate(mats):

    # create a sorted behavioral dataframe
    if mi>0:
        dfs = df.sort_values(by=['correct','right','trials'], ascending=[False,False,True])
        trialnames = ['Corr,', 'Corr,', 'Err,', 'Err,']

    else:
        dfs = df.sort_values(by=['right','contrast','trials'], ascending=[False,True,True])
        trialnames = ['', '', 'Right', '', '', '', '', 'Left', '', '']

    dfidx = dfs.index.values
    brk = np.concatenate([[0],np.where(dfidx[1:]-dfidx[:-1] <1)[0], [len(dfs)]])
    trialticks = [brk[i]+(brk[i+1]-brk[i])/2 for i in range(len(brk)-1)]

    ax = fig.add_subplot(gs[mi*3*2+4])
    ax.imshow(m[dfidx]*vecmax[mi], 
              vmin=0, vmax=.25, extent=[time[0], time[-1], 0, len(m)],
              aspect='auto', cmap=truncate_colormap(plt.get_cmap('twilight'), .5, 1), 
              origin='lower')

    # plot time stamps
    for t in ts:
        ax.plot([time[np.where(time>t)[0][0]], time[np.where(time>t)[0][0]]], 
                [0, m.shape[0]], 'w--', lw=1)

    # plot white lines separating trial types
    for b in brk:
        ax.plot([time[0], time[-1]], [b, b], 'w-', lw=1)

    ax.set_yticks(trialticks)
    ax.set_yticklabels(trialnames)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trials')


# time slicing
vecs = comp[2][0]
mats = comp[2][1]

vecmax = []
for vi,v in enumerate(vecs):
    vecmax.append(v.max())

    ax = fig.add_subplot(gs[vi*3*2+2])
    ax.plot(time, v/vecmax[vi], color='k')
    # plot time stamps
    for t in ts:
        ax.plot(time[np.where(time>t)[0][0]], 0, '^k')

    ax.plot(time, np.zeros(len(v)), 'k--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Weight')
    plt.xlim([time[0],time[-1]])

for mi,m in enumerate(mats[:1]):
    t = m.copy()
    # sort slice by region and then average activity in the first second
    t = sp.ndimage.gaussian_filter1d(t, sigma=10, axis=0)
    idx = np.concatenate([np.where(i)[0][np.argsort(np.mean(t[:100,i], 0), 0)] 
                          for i in idcs])

for mi,m in enumerate(mats):
    t = m.copy()[:,idx]

    ax = fig.add_subplot(gs[mi*3*2+5])
    ax.imshow(t*vecmax[mi], extent=[0, m.shape[1], 0, len(m)], 
              vmin=0, vmax=.25, aspect='auto', 
              cmap=truncate_colormap(plt.get_cmap('twilight'), .5, 1), 
              origin='lower')

    # plot lines for regions
    l = 0
    for le in lengths:
        l+=le
        ax.plot([l, l], [0, len(m)], 'w-', lw=1)

    ax.set_xlabel('Neurons')
    ax.set_ylabel('Trials')
    ax.set_xticks(neuronticks)
    ax.set_xticklabels(labels, rotation=45)
    plt.xlim([0,m.shape[1]])
    plt.ylim([0,m.shape[0]])

sns.despine()
plt.tight_layout()

plt.show()


