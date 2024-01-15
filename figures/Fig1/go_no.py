import torch
from torchsde import sdeint
import torch.nn as nn

torch.manual_seed(7)


def get_data(diffusion=3, number_neurons=100, number_trials=110):
    class net(nn.Module):

        def __init__(self):
            super(net, self).__init__()

            self.sde_type = 'ito'
            self.noise_type = 'diagonal'

        def f(self, t, x):

            return a*(2-x)

        def g(self, t, x):

            return torch.ones_like(x)*diffusion

    a = 0.5+torch.rand(number_neurons)*0.3

    model = net()
    ws = sdeint(model, torch.ones(1,number_neurons),torch.linspace(0,10,number_trials), dt=0.01).squeeze()

    trials = (torch.rand(number_trials)<0.5)

    final_w = torch.clamp(ws,0,2)

    slice = final_w.permute(1,0)*trials.float().unsqueeze(0)+(2-final_w.permute(1,0))*(1-trials.float().unsqueeze(0))
    slice = slice[torch.argsort(slice[:,5])]

    U, S, V = torch.linalg.svd(slice)

    slice += torch.randn_like(slice)*0.2
    slice = torch.relu(slice)

    return slice, S, (ws, trials)

def get_data_slice_2(number_neurons=100, number_trials=110, diffusion=3):
    slice, S, additional_variables = get_data(diffusion,number_neurons, number_trials)
    return slice, additional_variables

def get_data_vector_2(number_time=90):

    return torch.exp(-torch.linspace(-4,4,number_time)**2)

def get_tensor_2(diffusion=3,number_trials=110, number_neurons=100, number_time=90):
    slice, additional_variables = get_data_slice_2(diffusion=diffusion, number_trials=number_trials, number_neurons=number_neurons)
    v = get_data_vector_2(number_time)
    t = torch.einsum('ij,k->ijk', [slice, v])
    return (t/t.mean()).permute(1,0,2),  slice, v, additional_variables
