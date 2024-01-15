from core.classes import partition_tca, slice_tca, tca, orthognal_slice_tca, non_linear_tca
import torch
import random
from core.metrics import *
import warnings

def estimate_init_weights(data):

    return 0.0

def decompose(seed, data, number_components, iterations=500, decay_rate_mask=0.5, decay_rate_lr=0.5, decay_iterations=5,
              decay_type_mask='exponential', decay_type_lr='exponential', learning_rate=0.2, mask=0.5, cut_cv_mask=0, 
              mask_cross_validation=0.1, batch_size=20, sliceTCA=True, positive=False, orthogonal_penalty=False,
              metric=False, cross_validation=False, precision=torch.float32, orthogonal_constraint=False,
              loss_kernel_sigmas=None, positive_function=torch.abs, initialization='uniform',
              orthogonal_skip=(), test_freq=-1, verbose_train=False, verbose_test=True,
              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), animator=None):

    if type(data) is not torch.Tensor:
        data = torch.tensor(data)
        warnings.warn("""It is preferable to cast data to torch tensor with 
                        data = torch.tensor(data) before passing to decompose.""")

    dimensions = list(data.size())

    if seed is None:
        seed = random.randint(1,10**8)

    torch.manual_seed(seed)

    if sliceTCA == False and orthogonal_constraint == True:
        raise Exception("""Analytic orthogonal constraint is not implemented for non-slice TCA. 
            Either set sliceTCA=True or (orthogonal_constraint=False and orthogonal_penalty=True).""")

    if orthogonal_constraint == False:
        if sliceTCA == True:

            model = slice_tca(dimensions, subranks=number_components, positive=positive, orthogonal_penalty=orthogonal_penalty,
                                seed=seed, precision=precision, initialization=initialization,
                                init_weight=estimate_init_weights(data), loss_kernel_sigmas=loss_kernel_sigmas,
                                orthogonal_skip=orthogonal_skip, device=device, positive_function=positive_function)

            """model = non_linear_tca(dimensions, model.partitions, subranks=number_components, positive=positive, orthogonal_penalty=orthogonal_penalty,
                                seed=seed, precision=precision, initialization=initialization,
                                init_weight=estimate_init_weights(data),
                                orthogonal_skip=orthogonal_skip, device=device, non_linearity=torch.sigmoid)"""
        else:
            model = tca(dimensions, rank=number_components, positive=positive, orthogonal_penalty=orthogonal_penalty,
                                seed=seed, precision=precision, initialization=initialization,
                                init_weight=estimate_init_weights(data), loss_kernel_sigmas=loss_kernel_sigmas,
                                orthogonal_skip=orthogonal_skip, device=device, positive_function=positive_function)

            """model = non_linear_tca(dimensions, model.partitions, subranks=number_components, positive=positive,
                                   orthogonal_penalty=orthogonal_penalty,
                                   seed=seed, precision=precision, initialization=initialization,
                                   init_weight=estimate_init_weights(data),
                                   orthogonal_skip=orthogonal_skip, device=device, non_linearity=torch.sigmoid)"""
    else:
        model = orthognal_slice_tca(dimensions, subranks=number_components, seed=seed, device=device)
        warnings.warn("""Orthogonal constrained Slice TCA ignores parameters : 
            positive, orthogonal_penalty, precision, initialization, init_weight, orthogonal_skip, device.""")

    data = data.to(device)

    if cross_validation:
        cv_mask = mask_cross_validation.to(device)

    else:
        cv_mask = torch.tensor(1).to(device)

    if (decay_type_lr=='linear' and decay_rate_lr*(decay_iterations-1)>learning_rate) or \
            (decay_type_mask=='linear' and decay_rate_mask*(decay_iterations-1)>learning_rate):
        raise Exception('If decay is linear decay_rate x decay_iter must not be higher than initial learning rate.')

    iterations = int(iterations/decay_iterations)

    all_losses = [torch.mean((model.construct()*cv_mask-data*cv_mask)**2).item()]
    #all_losses = []

    ls = -1
    convergence = False
    for i in range(decay_iterations):
        ls, convergence = model.fit(data, max_iter=iterations, learning_rate=learning_rate, mask=mask,
                                    fixed_mask=cv_mask, batch_size=batch_size, test_freq=test_freq,
                                    verbose_train=verbose_train, verbose_test=verbose_test)

        all_losses.append(ls)

        if decay_type_mask == 'exponential':
            mask *= decay_rate_mask
        elif decay_type_mask == 'linear':
            mask -= decay_rate_mask
        else:
            raise Exception('Wrong decay type, select one of : exponential, linear.')
        if decay_type_lr == 'exponential':
            learning_rate *= decay_rate_lr
        elif decay_type_lr == 'linear':
            learning_rate -= decay_rate_lr
        else:
            raise Exception('Wrong decay type, select one of : exponential, linear.')

    if cross_validation:
        cv_mask_size = torch.mean(1-cv_mask)
        print('cv mask size: ', cv_mask_size)
        if cut_cv_mask==0:
            l_cv = torch.sqrt(torch.mean(((1-cv_mask)*model.construct() - (1-cv_mask)*data)**2)/cv_mask_size)
        else:
            cut_mask = cv_mask.clone()
            for ci,c in enumerate(cut_mask):
                for vi,v in enumerate(c):
                    if len(torch.where(v==0)[0])!=0: 
                        start, stop = torch.where(v==0)[0][0], torch.where(v==0)[0][-1]
                        cut_mask[ci,vi,int(start):int(start+cut_cv_mask)] = 1
                        cut_mask[ci,vi,int(stop-cut_cv_mask+1):int(stop+1)] = 1

            cut_mask_size = torch.mean(1-cut_mask)
            print('cut mask size', cut_mask_size)

            l_cv = torch.sqrt(torch.mean(((1-cut_mask)*model.construct() - (1-cut_mask)*data)**2)/cut_mask_size)
            print(ls, l_cv)


            
    else:
        l_cv = ()

    components = model.get_components()

    if metric == True:
        metric = multilinear_variance_explained(model, data)
    else:
        metric = ()

    return components, metric, all_losses, ls, l_cv, convergence, model


