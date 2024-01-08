import torch
import torch.nn as nn
#from classes import slice_tca
import copy
from core.convNd import convNd
import warnings
from scipy.stats import multivariate_normal
import torch.nn as nn
import torch
import numpy as np

def multilinear_variance_explained(slice_tca_object, data):

    dims = list(data.size())

    previous_error = torch.sqrt(slice_tca_object.loss_mse(slice_tca_object.construct(),data))

    previous_skipped_indices = copy.deepcopy(slice_tca_object.skipped_indices) #deep copy?

    metric = [[0 for j in range(slice_tca_object.subranks[i])] for i in range(len(slice_tca_object.components))]

    for i in range(len(slice_tca_object.components)):
        for j in range(slice_tca_object.subranks[i]):
            if j not in slice_tca_object.skipped_indices[i]:
                rank1 = slice_tca(dims,[[[i] for i in range(len(dims))]],[1], positive=slice_tca_object.positive,
                      positive_function=slice_tca_object.positive_function, initialization=slice_tca_object.initialization)

                slice_tca_object.skipped_indices[i].append(j)

                with torch.no_grad():
                    reconstructed_tensor = slice_tca_object.construct()

                    data_new = data - reconstructed_tensor

                rank1.fit(data_new, verbose=False, max_iter=100, learning_rate=0.05, noise=0, mask=0.2, batch_size=10, test_freq=-1)
                new_error = rank1.fit(data_new, verbose=False, max_iter=50, learning_rate=0.01, noise=0, mask=0, batch_size=10, test_freq=-1)

                metric[i][j] = ((new_error - previous_error)/new_error).item()

                slice_tca_object.skipped_indices = copy.deepcopy(previous_skipped_indices)

    return metric


class gaussian_tensor_distance(nn.Module):

    def __init__(self, dims, sigmas, approximation_threshold=0.99, noconv_threshold=0.01, trained=False, verbose_init=True):

        super(gaussian_tensor_distance, self).__init__()

        self.valence = len(sigmas)
        self.sigmas = sigmas
        self.dims = dims
        self.approximation_threshold = approximation_threshold
        self.noconv_threshold = noconv_threshold
        self.verbose_init = verbose_init

        self.kernel, self.kernel_shape = self.init_kernel()
        self.conv = self.init_conv()

        if trained==False:
            for param in self.parameters():
                param.requires_grad = False

    def init_kernel(self):

        covariance_matrix = torch.square(torch.diag(torch.tensor(self.sigmas).float()))
        mean = torch.zeros(len(self.sigmas))
        kernel_shape = torch.ones(self.valence)
        #threshold_Nd gives a lower bound for the error in valence n by integrating in valence 1
        threshold_Nd = torch.tensor(1 - (1 - self.approximation_threshold) / self.valence / 2)
        for i in range(self.valence):
            pdf = torch.distributions.normal.Normal(mean[i], self.sigmas[i])
            kernel_shape[i] = pdf.icdf(threshold_Nd)

        kernel_shape = kernel_shape*(kernel_shape>self.noconv_threshold).float()
        kernel_shape = torch.ceil(kernel_shape)
        kernel_shape = kernel_shape*2+1
        kernel_shape = torch.min(torch.stack((kernel_shape, torch.tensor(self.dims)*2-1)), dim=0)[0]
        kernel_shape_list = kernel_shape.int().tolist()

        indices = torch.from_numpy(np.indices(kernel_shape_list))
        indices = indices - (kernel_shape-1).view([self.valence]+[1 for i in range(self.valence)])/2
        indices = indices.permute([(i+self.valence+2)%(self.valence+1) for i in range(self.valence+1)])

        distrib = multivariate_normal(mean,covariance_matrix)
        kernel = torch.tensor(distrib.pdf(indices.view(-1,self.valence))).view(kernel_shape_list)

        if self.verbose_init == True:
            print('Kernel shape:', kernel_shape_list, '| approximation:', str(kernel.sum().item())[:10])
        kernel /= kernel.sum()  # ?

        return kernel, kernel_shape

    def init_conv(self):

        kernel_shape_tuple = tuple(self.kernel_shape.int().tolist())
        padding = tuple(((self.kernel_shape-1)/2).int().tolist())
        stride = tuple([1 for i in range(self.valence)])

        if self.valence >3:
            return convNd(
                            in_channels=1,
                            out_channels=1,
                            num_dims=self.valence,
                            kernel_size=kernel_shape_tuple,
                            stride=stride,
                            padding=padding,
                            use_bias=False,
                            padding_mode='zeros',
                            groups=1,
                            kernel_initializer=lambda x: self.kernel.unsqueeze(0).unsqueeze(0)) #device

        else:
            if self.valence == 3:
                conv_nn = nn.Conv3d
            elif self.valence == 2:
                conv_nn = nn.Conv2d
            elif self.valence == 1:
                conv_nn = nn.Conv1d

            conv = conv_nn(
                            in_channels=1,
                            out_channels=1,
                            kernel_size=kernel_shape_tuple,
                            stride=stride,
                            padding=padding,
                            bias=False,
                            padding_mode='zeros',
                            groups=1)
            with torch.no_grad():
                conv.weight.copy_( self.kernel.unsqueeze(0).unsqueeze(0))
            return conv

    def forward(self, x, y):

        dif = x-y
        convolved = self.conv(dif.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        z = torch.mean(convolved*dif)

        if z<0:
            warnings.warn('Negative non-convex loss. Possible fix : choose another kernel shape.')

        return z


def similarity_factor_wise(a,b): #not used

    total_distance = 0
    for i in range(len(a)):
        component_type_distance = 0
        for j in range(len(a[i])):  #
            if len(a[i][j]) != 0:
                pairwise_distance = torch.zeros((len(a[i][j]),len(a[i][j])))

                sigmas = list(torch.ones(len(a[i][j][0].size()))) #tba

                metric = gaussian_tensor_distance(list(a[i][j][0].size()), sigmas,
                                approximation_threshold=0.99, noconv_threshold=0.01, trained=False, verbose_init=False)

                for x in range(len(a[i][j])):
                    for y in range(len(a[i][j])):
                        pairwise_distance[x][y] = metric(a[i][j][x], b[i][j][y]).item()

                perm = torch.randperm(len(pairwise_distance))
                pairwise_distance = pairwise_distance[perm]
                #remaining_indices = [q for q in range(len(pairwise_distance))]
                min_distance = []

                remaining_indices = torch.zeros(len(pairwise_distance))

                for q in range(len(pairwise_distance)):
                    min_value, min_index = torch.min(pairwise_distance[q]+remaining_indices, dim=0)
                    min_distance.append(min_value.item())
                    remaining_indices[min_index.item()] += 10**8

                component_type_distance += torch.tensor(min_distance).mean().item()

            else:
                component_type_distance += 0

        total_distance += component_type_distance

    return total_distance

def similarity_component_wise(a,b, sigmas=(0.001,0.001,0.001), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    a.to(device)
    b.to(device)

    metric = gaussian_tensor_distance(a.dims, sigmas,
                                      approximation_threshold=0.99, noconv_threshold=0.01, trained=False,
                                      verbose_init=False)

    metric.to(device)

    subranks = a.subranks
    total_distance = 0

    for i in range(len(subranks)):
        if subranks[i] != 0:
            pairwise_distance = torch.zeros((subranks[i],subranks[i]))

            for x in range(subranks[i]):
                for y in range(subranks[i]):
                    pairwise_distance[x][y] = metric(a.construct_single_component(i,x),
                                                         b.construct_single_component(i,y)).item()

            perm = torch.randperm(len(pairwise_distance))
            pairwise_distance = pairwise_distance[perm]
            min_distance = []

            remaining_indices = torch.zeros(len(pairwise_distance))

            for q in range(len(pairwise_distance)):
                min_value, min_index = torch.min(pairwise_distance[q]+remaining_indices, dim=0)
                min_distance.append(min_value.item())
                remaining_indices[min_index.item()] += 10**8

            total_distance += torch.tensor(min_distance).mean().item()

    if sum(subranks) != 0:
        total_distance /= sum(subranks)

    a.cpu()
    b.cpu()

    return total_distance