
# data/path handling
from glob import glob
import pickle

# standard
import numpy as np
import scipy as sp

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


########################################################################################################
# load stuff
########################################################################################################

# load reconstructed tensor and raw data
t_sliceTCA  = np.load('../files/sliceTCA_reconstructed.npy').transpose([1,0,2]) # neurons, trials, time

# load neural data
cbl = np.load('../files/cbl_data.npy')
ctx = np.load('../files/ctx_data.npy')

t_raw = np.concatenate([cbl, ctx])

# load behavioral info and timestamps
with open(glob('../files/behavior.p')[0], 'rb') as f: 
    df = pickle.load(f)

ts = np.load('../files/timestamps.npy') 

conditions = df.outcome.values


########################################################################################################
# LDA projections
########################################################################################################


def LDA_proj(tensor1, tensor2, labels, ts):
    # tensor 1 is the data on which LDA axes are found
    # tensor 2 the data that is projected

    # first axis: left vs right during movement

    # get temporal average during movement (movement onset to reward time)
    meandat1 = np.mean(tensor1[:,:, ts[0]:np.sum(ts)], -1)

    # get weights
    lda1 = LDA(n_components=1).fit(meandat1.T, labels).coef_
    
    # second axis: time of movment onset vs time of reward

    # get temporal average of 20 samples around movement onset vs time of reward
    meandat1 = np.concatenate([np.mean(tensor1[:,:, ts[0]-10:ts[0]+10], -1).T, 
                              np.mean(tensor1[:,:, np.sum(ts)-10:np.sum(ts)+10], -1).T])
    lda2 = LDA(n_components=1).fit(meandat1, 
                                   np.concatenate([np.zeros(len(labels)), np.ones(len(labels))])).coef_

    # third axis:  pre-movement vs mid-movement
    meandat1 = np.concatenate([np.mean(tensor1[:,:, -20:], -1).T, 
                              np.mean(tensor1[:,:, 65:85], -1).T])
    lda3 = LDA(n_components=1).fit(meandat1, 
                                   np.concatenate([np.zeros(len(labels)), np.ones(len(labels))])).coef_

    # orthogonalize axes
    all_weights = np.concatenate([lda1, lda2, lda3])
    weights_orth = sp.linalg.orth(all_weights.T).T

    #
    lowD_orth = (tensor2.reshape([tensor2.shape[0], -1]).T \
                @ weights_orth.T).T.reshape([3, tensor2.shape[1], tensor2.shape[2]])

    lowD = (tensor2.reshape([tensor2.shape[0], -1]).T \
            @ all_weights.T).T.reshape([3, tensor2.shape[1], tensor2.shape[2]])

    # return smoothed LDA projections
    return sp.ndimage.gaussian_filter1d(lowD_orth, sigma=5, axis=-1), \
        sp.ndimage.gaussian_filter1d(lowD, sigma=5, axis=-1)
            


### find axis that separates left vs right and two temporal axes with LDA, only correct trials

# cerebellum
print('raw -> raw')
cbl_raw_orth, cbl_raw = LDA_proj(t_raw[:cbl.shape[0], np.where(conditions<2)[0]], 
                                 t_raw[:cbl.shape[0], np.where(conditions<2)[0]],
                                 conditions[conditions<2], ts)

print('sliceTCA -> sliceTCA')
cbl_sliceTCA_orth, cbl_sliceTCA = LDA_proj(t_sliceTCA[:cbl.shape[0], np.where(conditions<2)[0]], 
                                           t_sliceTCA[:cbl.shape[0], np.where(conditions<2)[0]], 
                                           conditions[conditions<2], ts)

print('raw -> sliceTCA')
cbl_proj_orth, cbl_proj = LDA_proj(t_sliceTCA[:cbl.shape[0], np.where(conditions<2)[0]],
                                   t_raw[:cbl.shape[0], np.where(conditions<2)[0]],
                                   conditions[conditions<2], ts)


# cortex
print('raw -> raw')
ctx_raw_orth, ctx_raw = LDA_proj(t_raw[cbl.shape[0]:, np.where(conditions<2)[0]],
                                 t_raw[cbl.shape[0]:, np.where(conditions<2)[0]], 
                                 conditions[conditions<2], ts)

print('sliceTCA -> sliceTCA')
ctx_sliceTCA_orth, ctx_sliceTCA = LDA_proj(t_sliceTCA[cbl.shape[0]:, np.where(conditions<2)[0]],
                                           t_sliceTCA[cbl.shape[0]:, np.where(conditions<2)[0]], 
                                           conditions[conditions<2], ts)

print('raw -> sliceTCA')
ctx_proj_orth, ctx_proj = LDA_proj(t_sliceTCA[cbl.shape[0]:, np.where(conditions<2)[0]], 
                                   t_raw[cbl.shape[0]:, np.where(conditions<2)[0]], 
                                   conditions[conditions<2], ts)


########################################################################################################
# plot projections on first LDA axis
########################################################################################################


time = (np.arange(150)-ts[0])*.033

colors = [plt.get_cmap('twilight')(np.linspace(.5,1,745)[::-1]), 
          plt.get_cmap('twilight')(np.linspace(0,.5,745))]


names = ['Raw data', 'sliceTCA', 'Raw data, projected on LDA axis from sliceTCA']

plt.figure(figsize=(4.2,4), dpi=300)

# loop over different projections
for li,latents in enumerate([cbl_raw, cbl_sliceTCA, cbl_proj]):

    plt.subplot(3,2,1+2*li)

    # plot example trials, left vs right
    plt.plot(time, latents[0, np.where(conditions[conditions<2]==0)[0][-10:]].T, 
            'darkred', alpha=.5)
    plt.plot(time, latents[0, np.where(conditions[conditions<2]==1)[0][-10:]].T, 
            'royalblue', alpha=.5)

    plt.plot(time, np.zeros(len(time)), 'k--', alpha=.2)
    plt.xlim(time[0], time[-1])
    plt.xticks([])
    plt.ylabel('LDA projection')
    
    # first row (raw -> raw has different y scale)
    if li==0: 
        plt.ylim(-10000,10000)
        plt.yticks([-10000,10000], [r'$-10^4$',r'$10^4$'])
    else: 
        plt.ylim(-50,50)
        plt.yticks([-50,50], [-50,50])
    
    plt.title(names[li], fontsize=10)
    
    # plot time stamps
    for t in range(len(ts)):
        plt.plot(time[sum(ts[:t+1])], 0, 'k^')

for li,latents in enumerate([ctx_raw, ctx_sliceTCA, ctx_proj]):

    plt.subplot(3,2,2+2*li)

    # plot example trials, left vs right
    plt.plot(time, latents[0, np.where(conditions[conditions<2]==0)[0][-10:]].T, 
            'darkred', alpha=.5)
    plt.plot(time, latents[0, np.where(conditions[conditions<2]==1)[0][-10:]].T, 
            'royalblue', alpha=.5)

    plt.plot(time, np.zeros(len(time)), 'k--', alpha=.2)
    plt.xlim(time[0], time[-1])
    plt.xticks([])

    if li==0: 
        plt.ylim(-1000,1000)
        plt.yticks([-1000,1000], [r'$-10^3$',r'$10^3$'])
    else:
        plt.ylim(-50,50)
        plt.yticks([-50,50], [-50,50])
    plt.title(names[li], fontsize=10)
    for t in range(len(ts)):
        plt.plot(time[sum(ts[:t+1])], 0, 'k^')

    plt.tight_layout()
    sns.despine(bottom=True)


########################################################################################################
# plot orthogonalized LDA projections in 3D 
########################################################################################################

proj = [[cbl_raw_orth,      ctx_raw_orth],
        [cbl_sliceTCA_orth, ctx_sliceTCA_orth],
        [cbl_proj_orth,     ctx_proj_orth]]

colors = [plt.get_cmap('twilight')(np.linspace(.5,1,452)[::-1]), 
          plt.get_cmap('twilight')(np.linspace(0,.5,452))]
colors2 = ['darkred', 'royalblue']

# figure setup
plt.figure(figsize=(12,6), dpi=300)

# loop over different data projections
for pi,p in enumerate(proj):

    # loop over cerebellum vs cortex
    for li,latents in enumerate(p):

        ax = plt.subplot(2,4,li*4+pi+1, projection='3d')

        # loop over left and right trials
        for i in [0,1]:
            trials = np.where(conditions[conditions<2]==i)[0]

            # plot ten example trials for each condition
            for t in trials[-10:]: #20:-10]:
                x_interp = np.arange(0, 149, 0.33)

                # cerebellum-specific view
                if li==0: 
                    ax.set_xlim(-4,3)
                    ax.set_ylim(-3,3)
                    ax.set_zlim(-5,1)

                    # interpolate for smooth curves
                    f0 = sp.interpolate.interp1d(np.arange(0,150), latents[0,t], kind='quadratic')
                    f1 = sp.interpolate.interp1d(np.arange(0,150), latents[1,t], kind='quadratic')
                    f2 = sp.interpolate.interp1d(np.arange(0,150), latents[2,t], kind='quadratic')
                    ax.view_init(elev=36, azim=-143)
                    
                    # plot shadows
                    ax.plot(f0(x_interp), f1(x_interp), np.zeros(len(x_interp))-5, c=colors2[i], alpha=.05)

                # cortex-specific view
                if li==1: 
                    ax.set_xlim(-2,2)
                    ax.set_ylim(-2,3)
                    ax.set_zlim(-5,1)

                    # interpolate for smooth curves
                    f0 = sp.interpolate.interp1d(np.arange(0,150), latents[0,t], kind='quadratic')
                    f1 = sp.interpolate.interp1d(np.arange(0,150), latents[1,t], kind='quadratic')
                    f2 = sp.interpolate.interp1d(np.arange(0,150), latents[2,t], kind='quadratic')
                    ax.view_init(elev=28, azim=-61)

                    # plot shadows
                    ax.plot(f0(x_interp), f1(x_interp), np.zeros(len(x_interp))-5, c=colors2[i], alpha=.05)

                # plot trajectories
                ax.scatter(f0(x_interp), f1(x_interp), f2(x_interp), c=colors[i], s=1, alpha=.4)
                
                # plot time stamp 0: movement onset
                ax.scatter(f0(x_interp)[ts[0]*3], f1(x_interp)[ts[0]*3], f2(x_interp)[ts[0]*3], 
                            c=colors2[i], s=4)

                # plot time stamp -1: reward
                ax.scatter(f0(x_interp)[sum(ts)*3], f1(x_interp)[sum(ts)*3], f2(x_interp)[sum(ts)*3], 
                            c='k', s=4)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])


        ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0)
plt.tight_layout()
sns.despine()

plt.show()

