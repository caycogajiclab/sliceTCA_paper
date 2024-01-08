import warnings
import torch

def get_filename(slice_TCA=None, positive=None, orthogonal_constraint=None, orthogonal_penalty=None, iterations=None,
                 learning_rate=None, decay_learning_rate=None, decay_mask=None, decay_iterations=None, mask=None,
                 batch_size=None, initialization=None, number_components=None, cross_validation=None,
                 loss_kernel_sigmas=None, seed=None, mask_cross_validation=None, **kwargs):

    filename = ''
    if slice_TCA is not None:
        filename += '-slice'+str(slice_TCA)
    if positive is not None:
        filename += '-positive'+str(positive)
    if orthogonal_constraint is not None:
        filename += '-orthC'+str(orthogonal_constraint)
    if orthogonal_penalty is not None:
        filename += '-orthP' +str(orthogonal_penalty)
    if iterations is not None:
        filename += '-iter'+str(iterations)
    if learning_rate is not None:
        filename += '-lr'+str(learning_rate)
    if mask is not None:
        filename += '-mask'+str(mask)
    if mask_cross_validation is not None:
        filename += '-maskCV' + str(mask_cross_validation)

    if decay_learning_rate is not None:
        filename += '-decayLr'+str(decay_learning_rate)
    if decay_mask is not None:
        filename += '-decayMask'+str(decay_mask)
    if decay_iterations is not None:
        filename += '-decayIter'+str(decay_iterations)

    if batch_size is not None:
        filename += '-batch'+str(batch_size)
    if initialization is not None:
        filename += '-init_'+initialization
    if number_components is not None:
        filename += '-subranks'+str(number_components)
    if cross_validation is not None:
        filename += '-cv'+str(cross_validation)
    if loss_kernel_sigmas is not None:
        filename += '-sigmas'+str(loss_kernel_sigmas)
    if seed is not None:
        filename += '-seed_'+str(seed)

    for i in kwargs.keys():
        filename += '-'+i+str(kwargs[i])

    filename = filename[1:]
    if len(filename)>254:
        warnings.warn('Max file name size reached (255), end of name will be trimmed.')
    filename = filename[:254]
    return filename

def get_block_mask(dimensions, block_dimensions, number_blocks, exact=True,
                   device=('cuda' if torch.cuda.is_available() else 'cpu'), seed=0): #if exact slow

    #recommended cpu for < 200x200x200
    torch.manual_seed(seed)
    
    valence = len(dimensions)
    tensor_block_dimensions = torch.tensor(block_dimensions)

    max_index = torch.tensor(dimensions)-tensor_block_dimensions+1
    flattened_max_dim = torch.prod(max_index)

    if exact == True:
        start = torch.zeros(flattened_max_dim, device=device)
        start[:number_blocks] = 1
        start = start[torch.randperm(flattened_max_dim, device=device)]
        start = start.reshape(tuple(max_index))
    else:
        if device == 'cpu':
            start = (torch.rand(tuple(max_index))<number_blocks/flattened_max_dim).long()
        elif device == 'cuda':
            start = (torch.cuda.FloatTensor(tuple(max_index)).uniform_() < number_blocks / flattened_max_dim).long()

    final_tensor = torch.ones(dimensions, device=device)
    start_index = start.nonzero()
    number_blocks = len(start_index)

    a = [[slice(start_index[j][i],start_index[j][i]+tensor_block_dimensions[i]) for i in range(valence)]
         for j in range(number_blocks)]

    for j in a:
        final_tensor[j] = 0

    return final_tensor


#Inspired from https://en.wikipedia.org/wiki/Talk:Varimax_rotation
def get_varimax_rotation(Phi, gamma = 1, q = 20, tol = 1e-6, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    p,k = Phi.shape
    R = torch.eye(k).to(device)
    d=0
    for i in range(q):
        d_old = d
        Lambda = Phi @ R
        u,s,vh = torch.svd(Phi.T @ (Lambda**3 - (gamma/p) * ( Lambda @ torch.diag(torch.diag(Lambda.T @ Lambda)))))
        R = u@vh
        d = torch.sum(s)
        if d_old != 0 and d/d_old < tol: break
    return R

def varimax(components, gamma = 1, q = 20, tol = 1e-6, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    for j in range(len(components)):
        if len(components[j]) != 0:
            if len(components[j][0].size())==2:
                vec_index = 0
                slice_index = 1
            elif len(components[j][1].size())==2:
                vec_index = 1
                slice_index = 0
            else:
                raise Exception('Varimax currently only works with slice TCA.')
            for q in range(len(components[j][0])):
                l = torch.sqrt(torch.sum(components[j][vec_index][q]**2))
                components[j][vec_index][q] /= l
                components[j][slice_index][q] *= l
            R = get_varimax_rotation(components[j][0].permute(1,0), gamma=gamma, q=q, tol=tol, device=device)
            R_inv = torch.inverse(R)
            components[j][vec_index] = R @ components[j][0]
            components[j][slice_index] = torch.einsum('ij,jkl->ikl', (R_inv, components[j][1]))

    return components

def get_gram_schmidt(X):

    U, S, V = torch.svd(X)

    return torch.linalg.inv((torch.diag(S) @ V)), (torch.diag(S) @ V)

def gram_schmidt(components):

    for j in range(len(components)):
        if len(components[j]) != 0:
            if len(components[j][0].size())==2:
                vec_index = 0
                slice_index = 1
            elif len(components[j][1].size())==2:
                vec_index = 1
                slice_index = 0
            else:
                raise Exception('Gram-Schmidt currently only works with slice TCA.')
            for q in range(len(components[j][0])):
                l = torch.sqrt(torch.sum(components[j][vec_index][q]**2))
                components[j][vec_index][q] /= l
                components[j][slice_index][q] *= l
            R, R_inv = get_gram_schmidt(components[j][0].permute(1,0))
            components[j][vec_index] = R @ components[j][0]
            components[j][slice_index] = torch.einsum('ij,jkl->ikl', (R_inv, components[j][1]))

    return

import numpy as np
def pca_basis(model):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    new_components = [[torch.zeros_like(model.vectors[i][0]), torch.zeros_like(model.vectors[i][1])] for i in range(len(model.subranks))]
    for i in range(len(model.subranks)):
        if model.subranks[i] != 0:
            reconstructed_subtensor = torch.zeros(model.dims, device=device)
            for j in range(model.subranks[i]):
                reconstructed_subtensor += model.construct_single_component(i,j)
            flattened_reconstructed_subtensor = reconstructed_subtensor.permute([i]+[q for q in range(len(model.subranks)) if q != i])
            flattened_reconstructed_subtensor = flattened_reconstructed_subtensor.reshape(model.dims[i],-1).transpose(0,1)
            """flattened_reconstructed_subtensor = flattened_reconstructed_subtensor.detach().cpu().numpy()
            U, S, V = np.linalg.svd(flattened_reconstructed_subtensor)"""
            U, S, V = torch.linalg.svd(flattened_reconstructed_subtensor.detach().cpu())
            U, S, V = torch.tensor(U, device=device), torch.tensor(S, device=device), torch.tensor(V, device=device)
            #U, S, V = torch.linalg.svd(flattened_reconstructed_subtensor)
            U, S, V = U[:,:model.subranks[i]], S[:model.subranks[i]], V[:model.subranks[i]]
            print(torch.dist(flattened_reconstructed_subtensor, U[:,:model.subranks[i]] @ torch.diag(S[:model.subranks[i]]) @ V[:model.subranks[i]]))
            US = (U @ torch.diag(S))
            slice = US.transpose(0,1).reshape([model.subranks[i]]+[model.dims[q] for q in range(len(model.subranks)) if q != i])
            print(V.shape, slice.shape)
            new_components[i][0] = V#.transpose(0,1)
            new_components[i][1] = slice

    return new_components


"""import time
t = time.time()
get_block_mask((200,100,200),(20,20,20),30, True, 'cpu')
print(time.time()-t)"""
