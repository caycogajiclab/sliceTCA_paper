
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


def PCA_smooth(tensor1, tensor2, ncomp):
    # tensor1 is for finding subspace, tensor2 is projected onto tensor1 subspace
    dat1_m = tensor1.reshape([tensor1.shape[0], -1])
    dat1_m = np.array([m-np.mean(m) for m in dat1_m])

    dat2_m = tensor2.reshape([tensor2.shape[0], -1])
    dat2_m = np.array([m-np.mean(m) for m in dat2_m])

    U, s, Vt = np.linalg.svd(dat1_m, full_matrices=False)
    dat_r = (dat2_m.T @ U[:,:ncomp]).T.reshape([ncomp,tensor2.shape[1],tensor2.shape[2]])
    dat_r = sp.ndimage.gaussian_filter1d(dat_r, sigma=5)

    return dat_r


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
# distance between left and right trials at movement onset
########################################################################################################

selfdist = []
otherdist = []

# loop over raw data and sliceTCA reconstruction
for di,dat in enumerate([t_raw, t_sliceTCA]):

    # cbl, then ctx
    for rdat in [dat[:cbl.shape[0]], dat[cbl.shape[0]:]]: 
        dists_self = []
        dists_other = []

        # self vs other
        for c in range(2):
            if c==0:
                cond1 = np.where(conditions==0)[0]
                cond2 = np.where(conditions==1)[0]
            else:
                cond1 = np.where(conditions==1)[0]
                cond2 = np.where(conditions==0)[0]
            
            # temporal average around time of movement onset
            means1 = np.mean(rdat[:,cond1,ts[0]-10:ts[0]+10], -1)
            means2 = np.mean(rdat[:,cond2,ts[0]-10:ts[0]+10], -1)

            # calculate distance
            dists_self.append([np.sqrt( sum( (m-np.mean(means1,-1))**2) ) 
                               for m in means1.T])
            dists_other.append([np.sqrt( sum( (m-np.mean(means2,-1))**2) )
                                for m in means1.T])
        
        selfdist.append(np.concatenate(dists_self))
        otherdist.append(np.concatenate(dists_other))

selfdist = np.array(selfdist)
otherdist = np.array(otherdist)

# ratio between / within: 
# 4 rows : cbl raw, ctx raw, cbl sliceTCA, ctx sliceTCA
ratio = otherdist/selfdist

# test raw vs sliceTCA distances
print('cbl', sp.stats.wilcoxon(ratio[0], ratio[2]))
print('ctx', sp.stats.wilcoxon(ratio[1], ratio[3]))

offset = 0


########################################################################################################
# plot stuff
########################################################################################################

plt.figure(figsize=(2,3), dpi=300)

# loop over different projections
for ri,r in enumerate(ratio):

    if ri%2==0: 
        color = 'k'
        offset +=1
    else:       
        color = 'chocolate'

    plt.plot(np.zeros(len(r))+ri+offset+np.random.randn(len(r))/5, r, '.', color=color, alpha=.1)
    plt.errorbar(ri+offset, np.mean(r), sp.stats.sem(r), color=color, capsize=5)

plt.plot(-1,-1, 'o', color='k', label='Cerebellum')
plt.plot(-1,-1, 'o', color='chocolate', label='Cortex')

plt.xticks([1.5,4.5], ['Raw data', 'sliceTCA'], rotation=45)
plt.xlim(-1,7)
plt.yticks([0,2,4,6])
plt.ylabel(r'$\Delta_{between} / \Delta_{within}$')
plt.legend(frameon=False)
plt.ylim(0,6)

sns.despine()
plt.tight_layout()

plt.show()

