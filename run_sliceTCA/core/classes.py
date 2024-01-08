import torch
import torch.nn as nn
from core.metrics import gaussian_tensor_distance
import warnings

class partition_tca(nn.Module):

    def __init__(self, dimensions, partitions, subranks, positive=False, positive_function=torch.abs,
                 orthogonal_penalty=False, orthogonal_skip=(), initialization='uniform', init_weight=0.0,
                 loss_kernel_sigmas=None, seed=7, precision=torch.float32,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        super(partition_tca, self).__init__()
        torch.manual_seed(seed)

        components = [[[dimensions[k] for k in j] for j in i] for i in partitions]

        if positive == True:
            self.positive_function = positive_function
        else:
            self.positive_function = self.identity

        self.vectors = nn.ModuleList([])

        for i in range(len(subranks)):
            rank = subranks[i]
            dim = components[i]

            # k-tensors of the outer product
            if initialization=='normal':
                self.v = nn.ParameterList([nn.Parameter(self.positive_function(torch.randn([rank] + j, dtype=precision, device=device))) for j in dim])
            elif initialization=='uniform':
                self.v = nn.ParameterList([nn.Parameter(self.positive_function(torch.rand([rank] + j, dtype=precision, device=device)*2-1)) for j in dim])
            elif initialization=='uniform-positive':
                self.v = nn.ParameterList([nn.Parameter(self.positive_function(torch.rand([rank] + j, dtype=precision, device=device)+init_weight)) for j in dim])
            else:
                raise Exception('Wrong initialization, select one of : normal, uniform, uniform-positive')

            self.vectors.append(self.v)

        self.partitions = partitions
        self.subranks = subranks
        self.components = components
        self.dims = dimensions
        self.valence = len(dimensions)
        self.positive = positive
        self.initialization = initialization
        self.init_weight = init_weight # HEIKE added init_weight
        self.orthogonal = orthogonal_penalty
        self.orthogonal_loss = nn.MSELoss()
        self.initial_orth_loss = -1
        self.skipped_indices = [[] for i in range(len(components))]
        self.seed = seed
        self.precision = precision
        self.orthogonal_skip = orthogonal_skip
        self.device = device
        self.loss_kernel_sigmas = loss_kernel_sigmas

        #GPU
        self.vectors.to(device)

        self.s = torch.prod(torch.tensor(self.dims))

        self.inverse_permutations = []
        self.flattened_permutations = []
        for i in self.partitions:
            temp = []
            for j in i:
                for k in j:
                    temp.append(k)
            self.flattened_permutations.append(temp)
            self.inverse_permutations.append(torch.argsort(torch.tensor(temp)).tolist())

        self.set_einsums()

        if loss_kernel_sigmas is not None:
            self.metric = gaussian_tensor_distance(dimensions,loss_kernel_sigmas).to(device)

        
    def identity(self, x):
        return x

    def set_einsums(self):

        self.einsums = []
        for i in self.partitions:
            lhs = ''
            rhs = ''
            for j in range(len(i)):
                for k in i[j]:
                    lhs += chr(105+k)
                    rhs += chr(105+k)
                if j != len(i)-1:
                    lhs += ','
            self.einsums.append(lhs+'->'+rhs)

    def construct(self):

        temp = torch.zeros(self.dims).to(self.device)

        for i in range(len(self.components)):
            for j in range(self.subranks[i]):
                if j not in self.skipped_indices[i]:
                    temp += self.construct_single_component(i,j)
                    """temp2 = [self.vectors[i][k][j] for k in range(len(self.components[i]))]
                    outer = self.positive_function(torch.einsum(self.einsums[i],temp2))
                    temp += outer.permute(self.inverse_permutations[i])"""
        return temp

    def construct_single_component(self, type, k):

        temp2 = [self.positive_function(self.vectors[type][q][k]) for q in range(len(self.components[type]))]
        outer = torch.einsum(self.einsums[type],temp2)
        outer = outer.permute(self.inverse_permutations[type])
        if k in self.skipped_indices[type]:
            warnings.warn('Constructing a tensor that has been ignore and is no longer optimized.')
        return outer


    def get_losses_skipped(self, data, slice_index):

        losses = [0 for i in range(self.subranks[slice_index])]
        with torch.no_grad():
            for i in range(self.subranks[slice_index]):
                if i not in self.skipped_indices[slice_index]:
                    self.skipped_indices[slice_index].append(i)
                    losses[i] = torch.sqrt(self.loss_mse(self.construct(), data)).item()
                    self.skipped_indices[slice_index].remove(i)
                else:
                    losses[i] = 10**6

        return losses

    def add_skip_index(self, data, slice_index):

        losses = self.get_losses_skipped(data, slice_index)

        index_smallest = losses.index(min(losses))
        self.skipped_indices[slice_index].append(index_smallest)
        losses[index_smallest] = 2 * 10 ** 6

    def add_worse_to_skip_indices(self, data, batch_removal_size=1):

        losses_list = []
        for i in range(len(self.vectors)):
            losses_list.append(self.get_losses_skipped(data, i))

        for j in range(batch_removal_size):
            smallest = 10**6
            smallest_index_slice = 10**5
            smallest_index_comp = 10**5
            for i in range(len(losses_list)):
                for q in range(len(losses_list[i])):
                    if losses_list[i][q]<smallest:
                        smallest_index_comp = q
                        smallest_index_slice = i
                        smallest = losses_list[i][q]

            self.skipped_indices[smallest_index_slice].append(smallest_index_comp)
            losses_list[smallest_index_slice][smallest_index_comp] = 2 * 10 ** 5

    def loss(self, a, b):
        if self.orthogonal == False:
            if self.loss_kernel_sigmas is not None:
                return self.metric(a,b)
            else:
                return self.loss_mse(a,b)
        else:
            return self.loss_orth(a,b)

    def loss_mse(self,a,b):
        if self.loss_kernel_sigmas is not None:
            return self.metric(a, b)
        else:
            return torch.sum((a - b) **2 ) / self.s

    def loss_orth(self):
        l2 = 0

        n=0

        for i in range(len(self.vectors)):

            ind = torch.tensor(self.skipped_indices[i]).long()

            for j in range(len(self.vectors[i])):

                if len(self.vectors[i][j].size())==2 and j not in self.orthogonal_skip:

                    temporary_loss = torch.matmul(self.positive_function(self.vectors[i][j]), self.positive_function(self.vectors[i][j]).permute(1,0))*(1-torch.eye(self.vectors[i][j].size()[0])).to(self.device) #right of * is non-orthonormal
                    temporary_loss[ind] *= 0
                    temporary_loss[:,ind] *= 0

                    #l2+= self.orthogonal_loss(temporary_loss, torch.zeros(temporary_loss.size()).to(device))
                    #print(l2.dtype)

                    if (torch.prod(torch.tensor(temporary_loss.size()))-temporary_loss.size()[0]) != 0: #if not nan, to be changed to if not 1 component (skipped taken into account)
                        l2 += torch.sqrt(torch.sum(torch.square(temporary_loss))/
                                         (torch.prod(torch.tensor(temporary_loss.size()))-temporary_loss.size()[0])) #prod???
                    n+=1

        #KL constraint?
        if self.initial_orth_loss==-1:
            self.initial_orth_loss = l2.detach()

        return l2/self.initial_orth_loss/n #need adaptive way of doing this

    def get_inverted_index(self, len_max, i):
        inverted_index = []
        for q in range(len_max):
            if q not in self.skipped_indices[i]:
                inverted_index.append(q)

        return inverted_index

    def get_components(self):

        temp = [[] for i in range(len(self.vectors))]

        for i in range(len(self.vectors)):
            for j in range(len(self.vectors[i])):
                temp_index = torch.tensor(self.get_inverted_index(len(self.vectors[i][j]),i))
                if len(temp_index) != 0:
                    temp[i].append(self.positive_function(self.vectors[i][j][temp_index]).data)

        return temp

    def set_components(self, components): #bug if positive_function != abs?

        for i in range(len(self.vectors)):
            for j in range(len(self.vectors[i])):
                temp_index = torch.tensor(self.get_inverted_index(len(self.vectors[i][j]),i))
                if len(temp_index) != 0:
                    #self.vectors[i][j][temp_index].data = components[i][j][temp_index]
                    with torch.no_grad():
                        #self.vectors[i][j][temp_index].copy_(components[i][j][temp_index])
                        self.vectors[i][j][temp_index] *= 0
                        self.vectors[i][j][temp_index] += components[i][j][temp_index].to(self.device) #critical change
        self.zero_grad()

    def init_train(self, max_iter=1000, min_delta=0.0001, steps_delta=10):
        self.max_iter = max_iter
        self.min_delta = min_delta
        self.steps_delta = steps_delta

    def fit(self, tensor, max_iter=1000, learning_rate=0.02, noise=0.0, mask=0,
            fixed_mask=torch.tensor(1), batch_size=10, test_freq=-1, verbose_train=True, verbose_test=True):

        if test_freq == -1:
            test_freq = batch_size

        self.init_train()
        self.max_iter = max_iter
        masked_entries = 1
        t = tensor * fixed_mask

        torch.manual_seed(self.seed)

        # Initialize the optimizer
        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)

        #t = tensor*fixed_mask

        #replace *= by torch.masked_fill ? perf difference?

        previous = -10**7

        queue = []

        # Optimize
        n = 0
        continue_while = True
        while n<self.max_iter and continue_while==True:

            if n%batch_size == 0 and mask != 0:
                masked_entries = (torch.rand(self.dims) >= mask).type(self.precision).to(self.device)
                t = tensor*masked_entries*fixed_mask

            # if n>self.steps_delta:
            #     if abs(previous-torch.sqrt(l).item()) < self.min_delta:
            #         #continue_while = False
            #         pass
            #     else:
            #         previous = torch.sqrt(l).item()

            that = self.construct()
            if noise != 0:
                that = that+noise*torch.randn(self.dims).to(self.device)
            if mask != 0:
                that = that*masked_entries
            that = that*fixed_mask

            total_mask = (fixed_mask*masked_entries == 0).type(self.precision).mean()

            l = self.loss_mse(that, t)/(1-total_mask)

            if self.orthogonal == True:
                l2 = self.loss_orth()
                l += l2

            if verbose_train == True:
                if self.orthogonal==True:
                    print('Iteration:', n, '\tmse_loss: %.10f' %(torch.sqrt(self.loss_mse(that,t)).item()),
                          '\torth_loss: %.10f' %torch.sqrt((l2*self.initial_orth_loss)).item())
                else:
                    print('Iteration:',n, '\tmse_loss: %.10f' %(torch.sqrt(l).item()))

            opt.zero_grad()
            l.backward()

            self.float()
            opt.step()
            self.type(self.precision)

            queue.append(l.item())
            if n%test_freq==0 and mask != 0 and verbose_test==True:
                with torch.no_grad():
                    test_loss = self.loss_mse(self.construct()*fixed_mask, tensor*fixed_mask)
                    print('Test -- Iteration:', n, '\tmse_loss: %.10f' %(torch.sqrt(test_loss).item()))
                    
            n+=1

        if verbose_test==True or verbose_train==True:
            if self.orthogonal == True:
                print('Final loss:','\tmse_loss: %.10f' %(torch.sqrt(self.loss_mse(that, t)).item()), '\torth_loss: %.10f'
                      %(torch.sqrt((l2 * self.initial_orth_loss)).item()))
            else:
                print('Final loss', '\tmse_loss: %.10f' %(torch.sqrt(l).item()))

        return self.loss_mse(tensor*fixed_mask, self.construct()*fixed_mask).item()/torch.mean((fixed_mask==1).float()).item(), not continue_while #torch.sqrt(test_loss).item() #self.vectors

class orthognal_slice_tca:

    def __init__(self, dimensions, subranks, seed=7):

        torch.manual_seed(seed)

        self.entries = torch.prod(torch.tensor(dimensions)).item()
        self.valence = len(dimensions)
        self.number_components = self.valence

        self.dims = dimensions
        self.subranks = subranks
        self.seed = seed
        self.components = [[torch.tensor([[0]]), torch.tensor([[0]])] for i in range(self.number_components)]
        self.that = torch.zeros(dimensions)

    def loss_mse(self,a,b):
        return torch.sum((a - b) **2 ) / self.entries

    def get_components(self):

        return [[self.components[i][0],
            self.components[i][1].reshape([len(self.components[i][0])]+[self.dims[q]
                    for q in range(len(self.dims)) if q != i])] if self.subranks[i] != 0 else [] for i in range(len(self.components))]

    def construct(self):
        return self.that

    def flat(self, t, leg):
        perm = [(leg + i) % self.valence for i in range(self.valence)]
        t = t.permute(perm)
        t = t.reshape(self.dims[leg], -1)
        return t

    def unflat(self, t, leg):

        perm = [(leg + i) % self.valence for i in range(self.valence)]
        t = t.reshape([self.dims[i] for i in perm])
        inverse_perm = [((self.valence - leg + i)) % self.valence for i in range(self.valence)]
        t = t.permute(inverse_perm)

        return t

    def fit(self, tensor, max_iter=50, noise=0.0, mask=0.0, batch_size=20, fixed_mask=None, test_freq=-1,
            verbose_train=True, verbose_test=True, learning_rate=None):

            leg = torch.randint(0, self.valence - 1, (1,)).item()

            if test_freq == -1:
                test_freq = batch_size

            if fixed_mask == None:
                fixed_mask = [torch.tensor(1) for i in range(self.valence)]

            t = tensor#*fixed_mask

            n = 0
            continue_while = True
            print('Initial loss:', str(torch.sqrt(torch.sum((t-self.that)**2)/self.entries).item())[:10])
            while n<max_iter and continue_while == True:

                if n%batch_size == 0 and mask != 0:
                    masked_entries = []
                    for i in range(self.valence):
                        masked_entries.append((torch.rand([self.dims[j] if j !=i else 1 for j in range(self.valence)])>=mask).float())

                self.that = self.flat(self.that, leg)

                # fit
                a, b = self.components[leg][1].transpose(0,1), self.components[leg][0]
                self.that -= (a @ b).transpose(0,1)
                self.that = self.unflat(self.that, leg)

                t = tensor + noise * torch.randn(self.dims)
                if mask != 0:
                    t = t * masked_entries[leg]*fixed_mask[leg]
                    that_masked = self.that * masked_entries[leg] * fixed_mask[leg]
                else:
                    that_masked = self.that
                    t = t*fixed_mask[leg]

                self.that = self.flat(self.that, leg)
                t = self.flat(t, leg)

                that_masked = self.flat(that_masked, leg)
                U, S, V = torch.pca_lowrank((t-that_masked).transpose(0,1))
                k = self.subranks[leg]

                us = U[:, :k] @ torch.diag(S[:k])
                b = us @ V[:, :k].transpose(0, 1)

                self.components[leg] = [V[:, :k].transpose(0, 1), us.transpose(0,1)]
                self.that += b.transpose(0,1)

                t = self.unflat(t, leg)
                self.that = self.unflat(self.that, leg)

                leg = (leg + torch.randint(1, self.valence - 1, (1,)).item()) % self.valence

                if verbose_test == True and n%test_freq==0:
                    print('Test -- Iteration:', n, '\tmse_loss:', str(torch.sqrt(self.loss_mse(tensor, self.that)).item())[:10])
                if verbose_train == True:
                    print('Iteration:',n, '\tmse_loss:', str(torch.sqrt(self.loss_mse(t,self.that)).item())[:10])

                n+=1

            if verbose_train == True or verbose_test == True:
                print('Final loss:', '\tmse_loss:', str(torch.sqrt(self.loss_mse(t,self.that)).item())[:10])
            return None, None


class slice_tca(partition_tca):
    def __init__(self, dimensions, subranks, positive=False, positive_function=torch.abs,
                 orthogonal_penalty=False, orthogonal_skip=(), initialization='uniform', init_weight=0.0,
                 loss_kernel_sigmas=None, seed=7, precision=torch.float32,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        valence = len(dimensions)
        partitions = [[[i],[j for j in range(valence) if j != i]] for i in range(valence)]

        super().__init__(dimensions, partitions=partitions, subranks=subranks, positive=positive,
                         positive_function=positive_function, orthogonal_penalty=orthogonal_penalty,
                         orthogonal_skip=orthogonal_skip, initialization=initialization, init_weight=init_weight,
                         loss_kernel_sigmas=loss_kernel_sigmas, seed=seed, precision=precision,
                         device=device)

class tca(partition_tca):
    def __init__(self, dimensions, rank, positive=False, positive_function=torch.abs,
                 orthogonal_penalty=False, orthogonal_skip=(), initialization='uniform', init_weight=0.0,
                 loss_kernel_sigmas=None, seed=7, precision=torch.float32,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        if type(rank) is not tuple:
            rank = (rank,)
        valence = len(dimensions)
        partitions = [[[j] for j in range(valence)]]

        super().__init__(dimensions, partitions=partitions, subranks=rank, positive=positive,
                         positive_function=positive_function, orthogonal_penalty=orthogonal_penalty,
                         orthogonal_skip=orthogonal_skip, initialization=initialization, init_weight=init_weight,
                         loss_kernel_sigmas=loss_kernel_sigmas, seed=seed, precision=precision,
                         device=device)

        """self.get_components_2 = self.get_components

    def get_components(self):

        return self.get_components_2()[0]"""


class non_linear_tca(partition_tca):
    def __init__(self, dimensions, partitions, subranks, positive=False, positive_function=torch.abs,
                 orthogonal_penalty=False, orthogonal_skip=(), initialization='uniform', init_weight=0.0,
                 loss_kernel_sigmas=None, seed=7, precision=torch.float32,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), non_linearity=torch.sigmoid):

        super().__init__(dimensions, partitions=partitions, subranks=subranks, positive=positive,
                         positive_function=positive_function, orthogonal_penalty=orthogonal_penalty,
                         orthogonal_skip=orthogonal_skip, initialization=initialization, init_weight=init_weight,
                         loss_kernel_sigmas=loss_kernel_sigmas, seed=seed, precision=precision,
                         device=device)

        self.non_linearity = torch.sigmoid
        self.coefficients = nn.ParameterList([nn.Parameter(torch.rand(i)) for i in subranks])

    def construct_single_component(self, type, k):

        temp2 = [self.vectors[type][q][k] for q in range(len(self.components[type]))]
        outer = torch.einsum(self.einsums[type],temp2)
        outer = self.non_linearity(outer.permute(self.inverse_permutations[type]))
        if k in self.skipped_indices[type]:
            warnings.warn('Constructing a tensor that has been ignore and is no longer optimized.')
        return outer

    def construct(self):

        temp = torch.zeros(self.dims).to(self.device)

        for i in range(len(self.components)):
            for j in range(self.subranks[i]):
                if j not in self.skipped_indices[i]:
                    if self.positive == True:
                        temp += self.positive_function(self.coefficients[i][j])*self.construct_single_component(i,j)
                    else:
                        temp += self.coefficients[i][j] * self.construct_single_component(i, j)
        return temp
