import random
import pickle
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor as Pool
from core.metrics import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from run.run_tca import decompose


def decompose_mp_sample(number_components, data, sample_size=1, threads_sample=1, iterations=100, decay_rate_mask = 0.5,
            decay_rate_lr=0.5, decay_iterations=5, decay_type_mask='exponential', decay_type_lr='exponential',
            learning_rate=0.2, mask=0.2, mask_cross_validation=0.1, cut_cv_mask=0, batch_size=20, sliceTCA=True, positive=False,
            orthogonal_penalty=False, orthogonal_constraint=False, metric=False, cross_validation=True,
            precision=torch.float32, initialization='uniform', orthogonal_skip=(), verbose_train=False, 
            verbose_test=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            number_solutions=False, folder=None):

    dec = partial(decompose_mp, data=data, iterations=iterations, number_components=number_components,
                  decay_rate_lr=decay_rate_lr, decay_rate_mask=decay_rate_mask, decay_type_mask=decay_type_mask,
                  decay_type_lr=decay_type_lr, mask_cross_validation=mask_cross_validation, cut_cv_mask=cut_cv_mask,
                  decay_iterations=decay_iterations, learning_rate=learning_rate, mask=mask, batch_size=batch_size,
                  sliceTCA=sliceTCA, orthogonal_constraint=orthogonal_constraint, 
                  positive=positive, orthogonal_penalty=orthogonal_penalty, metric=metric,
                  cross_validation=cross_validation, precision=precision, 
                  initialization=initialization, orthogonal_skip=orthogonal_skip, verbose_train=verbose_train,
                  verbose_test=verbose_test, device=device, return_model=number_solutions)

    sample = [i for i in range(sample_size)] # where i is seed

    # cv_seed_model_grid = dec(sample[0])
    with Pool(max_workers=threads_sample) as pool:
        cv_seed_model_grid = list(pool.map(dec, sample))

    cv_seed_model_grid = np.array(cv_seed_model_grid)
    cv = cv_seed_model_grid[:,0]
    ls = cv_seed_model_grid[:,1]

    pickle.dump(cv, open(folder + '/cv_grid_'+str(number_components)+'.p', 'wb'))
    pickle.dump(ls, open(folder + '/ls_grid_'+str(number_components)+'.p', 'wb'))

    seeds = cv_seed_model_grid[:,2]
    if number_solutions:
        best_model = torch.argmin(torch.tensor(cv.astype('float')))
        distances = torch.zeros((sample_size, sample_size))
        dist_to_best = torch.zeros(sample_size)

        if None not in cv_seed_model_grid[:,3]:
            models = cv_seed_model_grid[:,3]
            dist_to_best = torch.tensor([similarity_component_wise(models[best_model], m) for m in models])

            for a in range(sample_size):
                for b in range(sample_size):
                    if a<b:
                        distances[a, b] = similarity_component_wise(models[a], models[b], device=device)
            distances += torch.clone(distances).transpose(0,1)

            threshold = 5*10**-2 #-3
            dis = (distances < threshold).float().numpy()
            graph = csr_matrix(dis)
            nb_solutions = connected_components(graph, return_labels=False)

        else:
            nb_solutions = 1

        pickle.dump(distances, open(folder + '/dist_grid_'+str(number_components)+'.p', 'wb'))
        pickle.dump(dist_to_best, open(folder + '/dist2best_grid_'+str(number_components)+'.p', 'wb'))

        return cv, seeds, np.array([nb_solutions for i in range(len(seeds))]) #, distances
    else:
        return cv, seeds

def decompose_mp(seed, data, iterations=100, decay_rate_mask=0.5, decay_rate_lr=0.5, number_components=[],
            decay_iterations=5, decay_type_mask='exponential', decay_type_lr='exponential', cut_cv_mask=0,
            learning_rate=0.2, mask=0.2, mask_cross_validation=0.1, batch_size=20, sliceTCA=False, positive=False,
            orthogonal_penalty=False, orthogonal_constraint=False, metric=False, cross_validation=True, 
            precision=torch.float32, initialization='uniform', orthogonal_skip=(), verbose_train=False,
            verbose_test=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), return_model=False):

    print('Starting fitting components:', number_components, '- seed:', seed)
    if number_components == 0:#[0 for i in range(len(number_components))]:
        model = None

        if cross_validation:
            loss = torch.mean((mask_cross_validation[seed]*data)**2).item()

            cv_mask_size = torch.mean(1-mask_cross_validation[seed])
            print('cv mask size: ', cv_mask_size)

            if cut_cv_mask==0:
                loss_cv = torch.sqrt(torch.mean(((1-mask_cross_validation[seed])*data)**2)/cv_mask_size)
            else:    
                cut_mask = mask_cross_validation[seed].clone()
                for ci,c in enumerate(cut_mask):
                    for vi,v in enumerate(c):
                        if len(torch.where(v==0)[0])!=0: 
                            start, stop = torch.where(v==0)[0][0], torch.where(v==0)[0][-1]
                            cut_mask[ci,vi,int(start):int(start+cut_cv_mask)] = 1
                            cut_mask[ci,vi,int(stop-cut_cv_mask+1):int(stop+1)] = 1

                cut_mask_size = torch.mean(1-cut_mask)
                print('cut mask size', cut_mask_size)

                loss_cv = torch.sqrt(torch.mean(((1-cut_mask)*data)**2)/cut_mask_size).item()

        else: 
            loss_cv = None
            loss = torch.mean(data**2).item()

    else:
        _,_, _,ls, l_cv, _, model = decompose(seed=seed, data=data, number_components=number_components, 
            iterations=iterations, decay_rate_mask=decay_rate_mask,
            decay_rate_lr=decay_rate_lr, decay_iterations=decay_iterations, decay_type_mask=decay_type_mask,
            decay_type_lr=decay_type_lr, learning_rate=learning_rate, mask=mask,
            mask_cross_validation=mask_cross_validation[seed], batch_size=batch_size, sliceTCA=sliceTCA,
            positive=positive, orthogonal_penalty=orthogonal_penalty, metric=metric, cut_cv_mask=cut_cv_mask,
            cross_validation=cross_validation, precision=precision, orthogonal_constraint=orthogonal_constraint,
            initialization=initialization, orthogonal_skip=orthogonal_skip, test_freq=10**9,
            verbose_train=verbose_train, verbose_test=verbose_test, device=device)
        if cross_validation==True:
            loss_cv = l_cv.item()
        else:
            loss_cv = None
        if return_model==True:
            model.cpu()
        loss = ls

    if return_model == False:
        return loss_cv, loss, seed
    else:
        return loss_cv, loss, seed, model

def get_grid_sample(min_dims,max_dims):

    grid = torch.stack(
        torch.meshgrid([torch.tensor([i for i in range(min_dims[j],max_dims[j])]) for j in range(len(max_dims))]))

    return grid.flatten(start_dim=1).permute(1, 0).tolist()

def cv_search_sample(data, max_rank, min_rank=(), sample_size=1, threads_sample=1, threads_grid=1, iterations=100,
            decay_rate_mask=0.5, decay_rate_lr=0.5, decay_iterations=5, decay_type_mask='exponential', 
            decay_type_lr='exponential', seed=random.randint(1,10**4), learning_rate=0.2, mask=0.2, folder=None,
            mask_cross_validation=0.1, batch_size=20, sliceTCA=False, positive=False, orthogonal_penalty=False,
            orthogonal_constraint=False, metric=False, cross_validation=True, precision=torch.float32, cut_cv_mask=0,
            initialization='uniform', orthogonal_skip=(), verbose_train=False, verbose_test=False,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), number_solutions=False):

    mp.set_start_method('spawn')
    max_rank += 1
    rank_spam = max_rank-min_rank
    # valence = len(rank_spam)

    grid = [i for i in range(min_rank, max_rank)]
    print('Grid size:', str(rank_spam), '- sample:', sample_size,
          '- total_fit:', torch.tensor(grid).size()[0]*sample_size)

    if threads_grid != 1 and (verbose_train != False or verbose_test != False):
        #raise Exception('Verbose does not work with more than 1 thread.')
        # !!! change to automatic selection of verbose = False without raising error
        warnings.warn("""'Verbose does not work with more than 1 thread. Setting verbose to False.""")
        verbose_train = False
        verbose_test = False

    dec = partial(decompose_mp_sample, data=data, iterations=iterations, 
                  decay_rate_lr=decay_rate_lr, decay_rate_mask=decay_rate_mask, decay_type_mask=decay_type_mask,
                  decay_type_lr=decay_type_lr, mask_cross_validation=mask_cross_validation, cut_cv_mask=cut_cv_mask,
                  decay_iterations=decay_iterations, learning_rate=learning_rate, mask=mask, batch_size=batch_size,
                  sliceTCA=sliceTCA, orthogonal_constraint=orthogonal_constraint, threads_sample=threads_sample,
                  positive=positive, orthogonal_penalty=orthogonal_penalty, metric=metric, 
                  cross_validation=cross_validation, precision=precision, sample_size=sample_size, 
                  initialization=initialization, orthogonal_skip=orthogonal_skip, verbose_train=verbose_train,
                  verbose_test=verbose_test, folder=folder, device=device, number_solutions=number_solutions)


    with Pool(max_workers=threads_grid) as pool:
        cv_seed_grid = np.array(list(pool.map(dec, grid)), dtype=np.float32)

    cv_grid = torch.tensor(cv_seed_grid[:,0])
    ls_grid = torch.tensor(cv_seed_grid[:,1])
    seed_grid = torch.tensor(cv_seed_grid[:,2]).int()
    if number_solutions==True:
        nb_solutions_grid = torch.tensor(cv_seed_grid[:, 3]).int()

    cv_grid = cv_grid.reshape(rank_spam+[sample_size]).mean(dim=-1)
    seeds = seed_grid.reshape(rank_spam+[sample_size])

    if number_solutions == True:
        return cv_grid, seeds, solutions
    else:
        return cv_grid, seeds