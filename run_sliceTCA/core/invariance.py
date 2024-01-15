from itertools import combinations
import torch
import torch.nn as nn
import copy

class uniqueness(nn.Module):

    def __init__(self, model, positive=False, criterion=2): 
        # criterion 2, between
        # criterion 3, within
        super(uniqueness, self).__init__()

        #model.subranks: # comp. per slice type
        self.criterion = criterion
        self.number_components = len(model.subranks)
        if not positive:
            self.free_gl = nn.ParameterList([nn.Parameter(torch.randn([i, i])) for i in model.subranks])
        else:
            self.free_gl = nn.ParameterList([nn.Parameter(torch.eye(i)) for i in model.subranks]) 
            # this will lead to no rotation/no learning??
            # self.free_gl = nn.ParameterList([nn.Parameter(torch.randn([i, i])) for i in model.subranks]) 
        self.free_vectors_combinations = nn.ModuleList(
            [nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for j in range(self.number_components)]) for i in
             range(self.number_components)])

        self.components = model.get_components()
        self.model = copy.deepcopy(model)
        for param in self.model.parameters():
            param.requires_grad = False

        self.remaining_index_combinations = [[None for j in range(self.number_components)] for i in range(self.number_components)]
        self.remaining_dims_combinations = [[None for j in range(self.number_components)] for i in range(self.number_components)]

        for combination in combinations(list(range(self.number_components)), 2):
            if model.subranks[combination[0]] != 0 and model.subranks[combination[1]] != 0:
                temp = set(model.partitions[combination[0]][1])
                temp = temp.intersection(set(model.partitions[combination[1]][1]))
                self.remaining_index_combinations[combination[0]][combination[1]] = list(temp)

                self.remaining_dims_combinations[combination[0]][combination[1]] = [model.dims[i] for i in temp]
                remaining_dims = self.remaining_dims_combinations[combination[0]][combination[1]]

                free_vectors_dim = [model.subranks[combination[0]], model.subranks[combination[1]]] + remaining_dims
                free_vectors = nn.Parameter(torch.randn(free_vectors_dim)*(1-int(positive)))
                self.free_vectors_combinations[combination[0]][combination[1]] = free_vectors

    def within(self, components):

        for i in range(self.number_components):
            if self.model.subranks[i] != 0:
                det = torch.det(self.free_gl[i])
                components[i][0] = mm(self.free_gl[i] / det, components[i][0])
                components[i][1] = mm(torch.inverse(self.free_gl[i]) * det, components[i][1])

        return components

    def between(self, components):

        for combination in combinations(list(range(self.number_components)), 2):
            if self.model.subranks[combination[0]] != 0 and self.model.subranks[combination[1]] != 0:
                a_index = self.model.partitions[combination[0]][0][0]
                b_index = self.model.partitions[combination[1]][0][0]

                A_indexes = [b_index] + self.remaining_index_combinations[combination[0]][combination[1]]
                B_indexes = [a_index] + self.remaining_index_combinations[combination[0]][combination[1]]
                perm_B = [A_indexes.index(i) for i in self.model.partitions[combination[0]][1]]
                perm_A = [B_indexes.index(i) for i in self.model.partitions[combination[1]][1]]

                free_vectors = self.free_vectors_combinations[combination[0]][combination[1]]
                A = batch_outer(self.model.vectors[combination[0]][0], free_vectors)
                B = batch_outer(self.model.vectors[combination[1]][0], free_vectors.transpose(0, 1))
                A = A.sum(dim=0)
                B = B.sum(dim=0)
                A = A.transpose(0, 1)
                B = B.transpose(0, 1)
                A = A.permute([0] + [1 + i for i in perm_A])
                B = B.permute([0] + [1 + i for i in perm_B])
                components[combination[0]][1] += B
                components[combination[1]][1] -= A

        return components

    def forward(self, components):
        if self.criterion==2:
            return self.between(components)
        if self.criterion==3:
            return self.within(components)
        else:
            return self.within(self.between(components))

def outer(a,b):
    temp1 = [chr(105+i) for i in range(len(a.size()))]
    temp2 = [chr(105+len(a.size())+i) for i in range(len(b.size()))]
    indexes1 = ''.join(temp1)
    indexes2 = ''.join(temp2)
    formula = indexes1+','+indexes2+'->'+indexes1+indexes2
    return torch.einsum(formula, [a,b])

def batch_outer(a,b):
    temp1 = [chr(105+i+1) for i in range(len(a.size())-1)]
    temp2 = [chr(105+len(a.size())+i+1) for i in range(len(b.size())-1)]
    indexes1 = ''.join(temp1)
    indexes2 = ''.join(temp2)
    formula = chr(105)+indexes1+','+chr(105)+indexes2+'->'+chr(105)+indexes1+indexes2
    return torch.einsum(formula, [a,b])

def mm(a, b):
    temp1 = [chr(105+i) for i in range(len(a.size()))]
    temp2 = [chr(105+len(a.size())-1+i) for i in range(len(b.size()))]
    indexes1 = ''.join(temp1)
    indexes2 = ''.join(temp2)
    rhs = ''.join(temp1[:-1])+''.join(temp2[1:])
    formula = indexes1+','+indexes2+'->'+rhs
    return torch.einsum(formula,[a,b])

#Example of criteria

# def varimax(components):
#     l = 0
#     for i in range(len(components)):
#         if len(components[i])>0:
#             Phi = components[i][0].permute(1,0)
#             p,k = Phi.size()
#             I_C = torch.eye(p)-torch.ones([p,p])/p
#             N = 1 - torch.eye(k)
#             l += (torch.trace(Phi.T**2 @ I_C @ Phi**2 @ N)/4)
#     return l


def varimax(components):
    # https://arxiv.org/pdf/2004.05387.pdf
    l = 0
    for i in range(len(components)):
        if len(components[i])>0:
            x = components[i][0].permute(1,0) # should be n. elements x n. components
            l += sum(torch.mean(x**4 - (torch.mean(x**2, 0)**2).expand(x.shape), 0))
    return l


def L2_component_type_wise(components):
    l = 0
    for component_type in components:
        l += torch.mean(torch.square(component_type))
    return l/len(components)


def l2(reconstructed_tensors_of_each_type):
    l = 0
    for t in reconstructed_tensors_of_each_type:
        l += (t**2).mean()
    return l


def var(reconstructed_tensors_of_each_type):
    l = 0
    for t in reconstructed_tensors_of_each_type:
        l += ((t - t.mean())**2).mean()
    return l


#Junk solution
class construct:

    def __init__(self, model):
        super(construct, self).__init__()

        self.model = model

    def construct(self, components):

        temp = [torch.zeros(self.model.dims).to(self.model.device) for i in range(len(components))]

        for i in range(len(components)):
            for j in range(self.model.subranks[i]):
                temp[i] += self.construct_single_component(components,i,j)
        return temp
    
    def construct_single_component(self, components, type, k):

        temp2 = [components[type][q][k] for q in range(len(components[type]))]
        outer = self.model.positive_function(torch.einsum(self.model.einsums[type],temp2))
        outer = outer.permute(self.model.inverse_permutations[type])

        return outer

def clone_list(comp):
    newcomp = []
    for c in comp:
        subcomp = []
        for s in c:
            subcomp.append(torch.clone(s.detach()))
        newcomp.append(subcomp)
    return newcomp

def unique(model, objective_function, criterion=2, iterations=500, learning_rate=10**-3, batch_size=20, mask=.1, seed=0, verbose=False):

    torch.manual_seed(seed)

    positive = model.positive
    if positive:
        model.set_components(clone_list(model.get_components()))

    transformation = uniqueness(model, positive, criterion)
    if positive:
        optim = torch.optim.SGD(transformation.parameters(), lr=learning_rate) #SGD
    else:
        optim = torch.optim.Adam(transformation.parameters(), lr=learning_rate)
    construction_object = construct(model)

    components = model.get_components()
    if positive:
        new_transformation = uniqueness(model, positive, criterion)
        optim_new = torch.optim.SGD(new_transformation.parameters(), lr=learning_rate)  # SGD

    for iteration in range(iterations):

        components_transformed = transformation(clone_list(components))
        components_transformed_constructed = construction_object.construct(components_transformed)

        # mask entries for batch training
        if iteration%batch_size == 0 and mask!=0:
            masked_entries = []
            for i in range(len(components)):
                masked_entries.append((torch.rand(tuple(components_transformed_constructed[i].shape))>=mask).float())

        if criterion==2:
            components_masked = []
            if mask != 0:
                for i in range(len(components)):
                    components_masked.append(components_transformed_constructed[i] * masked_entries[i])
                l = objective_function(components_masked)
            else:
                l = objective_function(components_transformed_constructed)

        elif criterion==3:
                l = objective_function(components_transformed)

        if verbose:
            print('Iteration:', iteration, '\tloss:', l.item())

        optim.zero_grad()
        l.backward(retain_graph=positive)
        optim.step()

        if positive:

            components_transformed = transformation(clone_list(components))
            components_transformed_new = new_transformation(clone_list(components))
            softplus = nn.Softplus()
            l_temp = 0
            for i in range(len(components_transformed)):
                for j in range(len(components_transformed[i])):
                        l_temp += softplus(components_transformed_new[i][j][components_transformed[i][j]<10**-8]).mean() #<0?
            l_temp.backward()

            parameter_masks = [((i.grad==0).float() if i.grad is not None else None) for i in new_transformation.parameters()]

            optim_new.zero_grad()
            components_transformed_new = new_transformation(clone_list(components))
            components_transformed_constructed_new = construction_object.construct(components_transformed_new)

            if criterion==2:
                l = objective_function(components_transformed_constructed_new)
            elif criterion==3:
                l = objective_function(components_transformed)

            optim_new.zero_grad()
            l.backward()

            for p, p_mask in zip(new_transformation.parameters(), parameter_masks):
                if p.grad is not None:
                    p.grad *= p_mask

            optim_new.step()
            with torch.no_grad():
                for p, p_new in zip(transformation.parameters(), new_transformation.parameters()):
                    p.copy_(p_new)

    return transformation(clone_list(components)), l


# def unique(model, objective_function, iterations=500, learning_rate=10**-3, verbose=False):

#     positive = model.positive
#     if positive:
#         model.set_components(clone_list(model.get_components()))

#     transformation = uniqueness(model, positive)
#     if positive:
#         optim = torch.optim.SGD(transformation.parameters(),lr=learning_rate) #SGD
#     else:
#         optim = torch.optim.Adam(transformation.parameters(), lr=learning_rate)
#     construction_object = construct(model)

#     components = model.get_components()
#     if positive:
#         new_transformation = uniqueness(model, positive)
#         optim_new = torch.optim.SGD(new_transformation.parameters(), lr=learning_rate)  # SGD

#     for iteration in range(iterations):

#         components_transformed = transformation(clone_list(components))

#         components_transformed_constructed = construction_object.construct(components_transformed)
#         l = objective_function(components_transformed_constructed)

#         if verbose:
#             print('Iteration:', iteration, '\tloss:', l.item())

#         optim.zero_grad()
#         l.backward(retain_graph=positive)
#         optim.step()

#         if positive:

#             components_transformed = transformation(clone_list(components))
#             components_transformed_new = new_transformation(clone_list(components))
#             softplus = nn.Softplus()
#             l_temp = 0
#             for i in range(len(components_transformed)):
#                 for j in range(len(components_transformed[i])):
#                         l_temp += softplus(components_transformed_new[i][j][components_transformed[i][j]<10**-8]).mean() #<0?
#             l_temp.backward()

#             parameter_masks = [((i.grad==0).float() if i.grad is not None else None) for i in new_transformation.parameters()]

#             optim_new.zero_grad()
#             components_transformed_new = new_transformation(clone_list(components))
#             components_transformed_constructed_new = construction_object.construct(components_transformed_new)
#             l = objective_function(components_transformed_constructed_new)

#             optim_new.zero_grad()
#             l.backward()

#             for p, p_mask in zip(new_transformation.parameters(), parameter_masks):
#                 if p.grad is not None:
#                     p.grad *= p_mask

#             optim_new.step()
#             with torch.no_grad():
#                 for p, p_new in zip(transformation.parameters(), new_transformation.parameters()):
#                     p.copy_(p_new)

#     return transformation(clone_list(components))
