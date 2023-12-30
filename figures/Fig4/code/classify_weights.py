
# standard packages
import numpy as np
import scipy as sp

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# data/path handling
from glob import glob
import pickle

# plotting
import matplotlib.pyplot as plt
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
# plot histograms
################################################################################################################

# conditions: left, right, error
conditions = df.outcome.values

colors = ['darkred', 'royalblue', 'darkgrey']

# ## normalized weights for trial-slicing components
c1 = comp[0][0][0]/np.max(np.abs(comp[0][0][0]))
c2 = comp[0][0][1]/np.max(np.abs(comp[0][0][1]))
c3 = comp[0][0][2]/np.max(np.abs(comp[0][0][2]))

# figure specifications
plt.figure(figsize=(4,3), dpi=300)


# trial slicing components
plt.subplot(3,2,1)
for i in range(3):
    plt.hist(c1[conditions==i], color=colors[i], alpha=.7, bins=50, range=(-1.5,1.5))

# fit LDA corr vs error
lda = LDA().fit(c1.reshape(-1, 1), np.array(conditions==2).astype('int'))

# plot decision boundary
plt.plot([-lda.intercept_/lda.coef_[0], -lda.intercept_/lda.coef_[0]], [0,22], 'k--')

# percent correct
print(lda.score(c1.reshape(-1, 1), np.array(conditions==2).astype('int')))

plt.ylim(0,22)
plt.xlim(-1.5, 1.5)
plt.yticks([])
plt.xlabel('Weight component 1')

plt.subplot(3,2,3)
for i in range(3):
    plt.hist(c2[conditions==i], color=colors[i], alpha=.7, bins=50, range=(-1.5,1.5))

# fit LDA left vs right (correct only)
lda = LDA().fit(c2[conditions<2].reshape(-1, 1), np.array(conditions[conditions<2]))
plt.plot([-lda.intercept_/lda.coef_[0], -lda.intercept_/lda.coef_[0]], [0,22], 'k--')
print(lda.score(c2[conditions<2].reshape(-1, 1), np.array(conditions[conditions<2])))

plt.ylim(0,22)
plt.xlim(-1.5, 1.5)
plt.yticks([])
plt.xlabel('Weight component 2')

plt.subplot(3,2,5)
for i in range(3):
    plt.hist(c3[conditions==i], color=colors[i], alpha=.7, bins=50, range=(-1.5,1.5))

plt.ylim(0,22)
plt.xlim(-1.5, 1.5)
plt.yticks([])
plt.xlabel('Weight component 3')

c1 = -comp[1][0][0]/np.max(np.abs(comp[1][0][0])) # switch sign as in component plots
c2 = -comp[1][0][1]/np.max(np.abs(comp[1][0][1]))
c3 = comp[1][0][2]/np.max(np.abs(comp[1][0][2]))


# neuron slicing components

plt.subplot(3,2,2)

# plot histograms cbl vs ctx
plt.hist(c2[:cbl.shape[0]], color='k', alpha=.7, bins=50, range=(-1.5,1.5))
plt.hist(c2[cbl.shape[0]:], color='darkgrey', alpha=.7, bins=50, range=(-1.5,1.5))

# lda cbl vs ctx
lda = LDA().fit(c2.reshape(-1, 1), np.array(np.arange(len(c2))<cbl.shape[0]).astype('int'))
plt.plot([-lda.intercept_/lda.coef_[0], -lda.intercept_/lda.coef_[0]], [0,22], 'k--')
print(lda.score(c2.reshape(-1, 1), np.array(np.arange(len(c2))<cbl.shape[0]).astype('int')))

plt.ylim(0,22)
plt.xlim(-1.5, 1.5)
plt.yticks([])
plt.xlabel('Weight component 1')

plt.subplot(3,2,4)
plt.hist(c1[:cbl.shape[0]], color='k', alpha=.7, bins=50, range=(-1.5,1.5))
plt.hist(c1[cbl.shape[0]:], color='darkgrey', alpha=.7, bins=50, range=(-1.5,1.5))

lda = LDA().fit(c1.reshape(-1, 1), np.array(np.arange(len(c1))<cbl.shape[0]).astype('int'))
plt.plot([-lda.intercept_/lda.coef_[0], -lda.intercept_/lda.coef_[0]], [0,22], 'k--')
print(lda.score(c1.reshape(-1, 1), np.array(np.arange(len(c1))<cbl.shape[0]).astype('int')))

plt.ylim(0,22)
plt.xlim(-1.5, 1.5)
plt.yticks([])
plt.xlabel('Weight component 2')

plt.subplot(3,2,6)
plt.hist(c3[:cbl.shape[0]], color='k', alpha=.7, bins=50, range=(-1.5,1.5))
plt.hist(c3[cbl.shape[0]:], color='darkgrey', alpha=.7, bins=50, range=(-1.5,1.5))

lda = LDA().fit(c3.reshape(-1, 1), np.array(np.arange(len(c3))<cbl.shape[0]).astype('int'))
plt.plot([-lda.intercept_/lda.coef_[0], -lda.intercept_/lda.coef_[0]], [0,22], 'k--')
print(lda.score(c3.reshape(-1, 1), np.array(np.arange(len(c3))<cbl.shape[0]).astype('int')))

plt.ylim(0,22)
plt.xlim(-1.5, 1.5)
plt.yticks([])
plt.xlabel('Weight component 3')


sns.despine(left=True)
plt.tight_layout()

plt.show()