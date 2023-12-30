# standard packages
import numpy as np

# data/path handling
from glob import glob
import pickle

# plotting
import matplotlib.pyplot as plt
import seaborn as sns


########################################################################################################
# load PCA results
########################################################################################################

with open(glob('../files/wagner-trialPCA-3.p')[0], 'rb') as f: 
    mat, vec = pickle.load(f)
mat = mat.transpose([1,0,2]) # number of components, number of neurons, time

# calculate eigenspectra for each component/slice
spectraPCA = []
for m in mat:
    m = np.array([me-np.mean(me) for me in m])
    U,S,VT = np.linalg.svd(m, full_matrices=False)
    spectraPCA.append(S**2)
spectraPCA = np.array(spectraPCA)


########################################################################################################
# load FA results
########################################################################################################

with open(glob('../files/wagner-trialFA-3.p')[0], 'rb') as f: 
    mat, vec = pickle.load(f)
mat = mat.transpose([1,0,2])

# calculate eigenspectra
spectraFA = []
for m in mat:
    m = np.array([me-np.mean(me) for me in m])
    U,S,VT = np.linalg.svd(m, full_matrices=False)
    spectraFA.append(S**2)
spectraFA = np.array(spectraFA)


########################################################################################################
# load sliceTCA results
########################################################################################################

with open(glob('../files/wagner-sliceTCA-3-3-0-processed-components.p')[0], 'rb') as f: 
    comp = pickle.load(f)

mat = comp[0][1] # get trial-slicing components

# calculate eigenspectra
spectraSliceTCA = []
for m in mat:
    m = np.array([me-np.mean(me) for me in m])
    U,S,VT = np.linalg.svd(m, full_matrices=False)
    spectraSliceTCA.append(S**2)
spectraSliceTCA = np.array(spectraSliceTCA)


########################################################################################################
# plot stuff
########################################################################################################

# eigenspectra
fig = plt.figure(figsize=(2.5,2.5), dpi=200)
ax = plt.subplot(111)

colors = ['steelblue', 'chocolate', 'k']
names = ['PCA', 'FA', 'sliceTCA']


# loop through different methods
for si,s in enumerate([spectraPCA, spectraFA, spectraSliceTCA]):

    # normalize eigenvalues
    s = np.array([se/np.sum(se) for se in s])
    
    # loop through the three components
    for i in range(3):
        ax.plot(np.arange(1,len(s[i])+1), s[i], color=colors[si], alpha=.2)
    
    # plot average eigenspectrum (over three components)
    ax.plot(np.arange(1,len(s[0])+1), np.mean(s, 0), 
            color=colors[si], label=names[si])
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Eigenvalue (normalized, log)')
    ax.set_xlabel('Eigenvalue number (log)')
    ax.set_xlim(1,150)
    ax.set_ylim(10**-5,1)
    ax.set_yticks([10**-5,10**-4, 10**-3, 10**-2, 10**-1, 1])
    plt.title('Neuron-time slices')
plt.legend(frameon=False, loc=3)
plt.tight_layout()
sns.despine()


# first eigenvalue (normalized)
fig = plt.figure(figsize=(1.5,2.5), dpi=200)
ax2 = plt.subplot(111)

# loop through 
for si,s in enumerate([spectraPCA, spectraFA, spectraSliceTCA]):

    # normalize eigenvalues
    s = np.array([se/np.sum(se) for se in s])

    # plot 1st eigenvalue
    ax2.plot(np.zeros(len(s))+si+np.random.randn(len(s))/10, s[:,0], 
        'o', ms=3, color=colors[si])
    ax2.set_xlim(-1,3)
    ax2.set_ylim(.2,.8)
    ax2.set_xticks([0,1,2])
    ax2.set_xticklabels(names, rotation=45)
    ax2.set_yticks([.2, .3, .4, .5, .6,.7,.8])
    ax2.set_ylabel('Eigenvalue (normalized)')
    plt.title(' ')
plt.tight_layout()
sns.despine()

plt.show()

