# data/path handling
from glob import glob
import pickle

# standard packages
import numpy as np
import scipy as sp
import torch 

# plotting
import matplotlib.colors as mcols 
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import seaborn as sns



##################################################################################################
# load stuff
#############################################################################################################

# load model
with open('../files/IBL-sliceTCA-2-3-3-model.p', 'rb') as f: 
    model = pickle.load(f)

# pass model to CPU
model.device = torch.device('cpu')
model.to(model.device)

# load raw data
data = np.concatenate([np.load('../files/IBL_raw_data%.0f.npy' %i) 
                       for i in range(6)])

# load region labels
with open('../files/region_labels.p', 'rb') as f: 
    idcs, labels, regions = pickle.load(f)

# exclude "other" neurons (from random regions)
idcs = idcs[:-1]


#############################################################################################################
# plot goodness of fit
#############################################################################################################

# reconstruct model
full = model.construct().detach().numpy()

# goodness of fit
plt.figure(figsize=(3,1.75), dpi=300)

# calculate goodness of fit measure
gof = 1 - np.sum(np.sum((data-full)**2, 0), 1) / np.sum(np.sum(data**2, 0), 1)
gofs = [gof[i] for i in idcs]

# plot
plt.bar(np.arange(len(gofs)), [g.mean() for g in gofs], color='silver')
print([g.mean() for g in gofs])

for gi,g in enumerate(gofs):
    plt.plot(np.zeros(len(g))+gi+np.random.randn(len(g))/10, g, 
             '.', color='k', alpha=.5, ms=2)

plt.ylabel('Goodness of fit')
plt.title('Full reconstruction', fontsize=11)

plt.xticks(np.arange(len(gofs)), labels[:-1], rotation=45)
plt.ylim(0,1)
plt.yticks([0,1])

plt.tight_layout()
sns.despine()


#############################################################################################################
# plot goodness of fit
#############################################################################################################

# colors for regions
allencolors = ['teal', 'yellowgreen', 'yellowgreen',  'salmon', 'violet', 'violet', 'k']

# contribution of single component to full reconstruction
plt.figure(figsize=(8,5), dpi=300)
names = ['Trial-slicing', 'Neuron-slicing', 'Time-slicing']

# loop over component types
for part in range(3):

    # loop over components in the specific slice type
    for c in range(len(model.vectors[part][0])):

        # reconstruct single component
        rec = model.construct_single_component(part,c).detach().numpy()

        plt.subplot(3,3,part+c*3+1)

        # contribution of single component to full reconstruction, sum over trials and time
        frac = np.sum(np.sum((rec), 0), 1) / np.sum(np.sum((full), 0), 1)
        
        #  fraction by region
        fracs = [frac[i] for i in idcs]

        # plot average and single-neuron fractions
        for fi,f in enumerate(fracs):
            plt.bar(fi, f.mean(), color=allencolors[fi])
        for fi,f in enumerate(fracs):
            plt.plot(np.zeros(len(f))+fi+np.random.randn(len(f))/10, f, 
                     '.', color='k', alpha=.5, ms=2)

        plt.ylabel('Comp. weight')
        plt.title(names[part]+' component ' + str(c+1), fontsize=11)
        plt.xticks(np.arange(6), labels[:-1], rotation=45)
        plt.ylim(0,1)
        plt.yticks([0,1])
plt.tight_layout()
sns.despine()

plt.show()
