
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

import pyro.distributions as dist


class DGP:
    def __init__(self):
        pass
    
    def generate_data(self, n):
        raise NotImplementedError
    
    def get_log_likelihood(self, x):
        raise NotImplementedError
    

class ModelPawelczyk(DGP):
    def __init__(self):
        mix = dist.Categorical(torch.ones(3,))
        comp = dist.Independent(dist.Normal(torch.tensor([[-10.0,5], [0,5], [10,0]]), torch.ones(3,2)), 1)
        self.gmm = dist.MixtureSameFamily(mix, comp)
        self.columns = ['x1', 'x2']
        self.f = lambda sample: sample[:,1] > 6 

    def generate_data(self, n):  
        data = self.gmm.sample((n,)).numpy()
        data = pd.DataFrame(data, columns=self.columns)
        data['y'] = self.f(data.to_numpy())
        return data

    def get_log_likelihood(self, data):
        data = torch.tensor(data[self.columns].to_numpy())
        log_prob = self.gmm.log_prob(data)
        return log_prob


class TwoSines(DGP):
    def __init__(self):
        super().__init__()
    
    def generate_data(self, n):
        mix = dist.Categorical(torch.ones(2,)).sample((n,))
        x1 = dist.Normal(mix*1.0, 3.0).sample()
        x2 = dist.Normal(torch.sin(x1) - 2.0 * mix + 1, 0.3).sample() 
        data = torch.stack((x1, x2, mix)).numpy()
        names = ['x1', 'x2', 'y']
        return pd.DataFrame(data.T, columns=names)
    
    def get_log_likelihood(self, data):
        # gets log likelihood ignoring y
        log_probss = []
        for y in [0, 1]:
            x1 = torch.tensor(data['x1'])
            x2 = torch.tensor(data['x2'])
            y_ = torch.tensor(y)

            d_x1 = dist.Normal(y_*1.0, 3.0)
            d_x2 = dist.Normal(torch.sin(x1) - 2.0 * y_ + 1, 0.3)
            log_probs = d_x1.log_prob(x1) + d_x2.log_prob(x2)
            log_probss.append(log_probs)
        log_probss = torch.stack(log_probss)
        return torch.logsumexp(log_probss, 0).numpy()


class Cassini(DGP):
    def __init__(self):
        super().__init__()

    def generate_data(self, n):
        y1 = dist.Categorical(torch.tensor([1,2])).sample((n,))
        y2 = dist.Categorical(torch.ones(2,)).sample((n,))
        y = y1 + y1 * y2

        x1_center = dist.Normal(0.0, 0.2)
        x1_outer = dist.Normal(0.0, 0.5)

        x1_mix = dist.MaskedMixture(y1.to(torch.bool), x1_center, x1_outer).sample()

        x2_center = dist.Normal(0.0*x1_mix, 0.2)
        x2_outer = dist.Normal(torch.cos(x1_mix)*torch.pow(-1, y2), 0.2)

        x2_mix = dist.MaskedMixture(y1.to(torch.bool), x2_center, x2_outer).sample()

        data = [x1_mix.numpy(), x2_mix.numpy(), y.numpy()]
        names = ['x1', 'x2', 'y']
        return pd.DataFrame(np.stack(data).T, columns=names)


    def get_log_likelihood(self, data):
        x1_obs = torch.tensor(data['x1'])
        x2_obs = torch.tensor(data['x2'])
        
        combs = [(0,0), (1, 0), (1, 1)]
        logs_probss = []
        for (y1, y2) in combs:
            y1, y2 = torch.tensor(y1).repeat(x1_obs.shape[0]), torch.tensor(y2).repeat(x1_obs.shape[0])
            x1_center = dist.Normal(0.0, 0.2)
            x1_outer = dist.Normal(0.0, 0.5)

            x1_lls = dist.MaskedMixture(y1.to(torch.bool), x1_center, x1_outer).log_prob(x1_obs)

            x2_center = dist.Normal(0.0*x1_obs, 0.2)
            x2_outer = dist.Normal(torch.cos(x1_obs)*torch.pow(-1, y2), 0.2)

            x2_lls = dist.MaskedMixture(y1.to(torch.bool), x2_center, x2_outer).log_prob(x2_obs)

            log_probs = x1_lls + x2_lls
            logs_probss.append(log_probs)
        logs_probss = torch.stack(logs_probss)
        return torch.logsumexp(logs_probss, 0).numpy()


# ## still need to implement the log likelihood function


# class TwoMoons(DGP):
#     def __init__(self, noise=0.1):
#         super().__init__()
#         self.noise = 0.1

#     def generate_data(self, n):
#         y = dist.Categorical(torch.ones(2,)).sample((n,))

#         pos = dist.Uniform(0, torch.pi).sample((n,))
#         x1 = torch.cos(pos) * torch.pow(-1, y) + y
#         # outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
#         # inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
#         x2 = torch.sin(pos) * torch.pow(-1, y) + y * 0.5
#         # outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
#         # inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5
#         noise = dist.Normal(0, self.noise).sample((n, 2))
#         x1 = x1 + noise[:, 0]
#         x2 = x2 + noise[:, 1]
#         data = [x1, x2, y]
#         names = ['x1', 'x2', 'y']
#         return pd.DataFrame(torch.stack(data).T.numpy(), columns=names)


# class Shapes(DGP):
#     def __init__(self):
#         super().__init__()

#     def generate_data(self, n):
#         yh = dist.Categorical(torch.ones(2)).sample((n,))
#         yv = dist.Categorical(torch.ones(2)).sample((n,))
#         y = yh + 2.0*yv

#         # blob top left
#         blob_x1 = dist.Normal(torch.tensor([-1.0]), torch.tensor([0.2]))

#         # rectangle bottom left
#         rect_x1 = dist.Uniform(torch.tensor([-1.5]), torch.tensor([-0.5]))

#         # triangle top right
#         triangle_x1 = dist.Uniform(torch.tensor(0.0), torch.tensor(2.0))

#         # sine wave bottom right
#         sine_x1 = dist.Uniform(torch.tensor(0.0), torch.tensor(2.0))

#         # separate mixing distributions for x1 and x2
#         # three mixtures for x1, three for x2 (one for yh and two for yv each)

#         dist_x1 = dist.MaskedMixture(yh.to(torch.bool),
#                                      dist.MaskedMixture(yv.to(torch.bool), rect_x1, blob_x1),
#                                      dist.MaskedMixture(yv.to(torch.bool), sine_x1, triangle_x1))
        
#         x1 = dist_x1.sample()
        
#         rect_x2 = dist.Uniform(torch.tensor([-2.0]).repeat(n), torch.tensor([0]).repeat(n))
#         blob_x2 = dist.Normal(torch.tensor([1.0]).repeat(n), torch.tensor([0.2]).repeat(n))
#         spread = torch.abs((0.5 - torch.abs(1 - x1) / 2.0)) * yv * yh + torch.finfo(torch.float32).eps
#         triangle_x2 = dist.Uniform(torch.tensor(1.0).repeat(n), torch.tensor(1.0 + spread))
#         sine_x2 = dist.Normal(10*torch.sin(x1*torch.pi), torch.tensor(0.5))

#         dist_x2 = dist.MaskedMixture(yh.to(torch.bool),
#                                      dist.MaskedMixture(yv.to(torch.bool), rect_x2, blob_x2),
#                                      dist.MaskedMixture(yv.to(torch.bool), sine_x2, triangle_x2))

#         x1, x2 = dist_x1.sample(), dist_x2.sample()

#         data = [x1.numpy(), x2.numpy(), y.numpy()]
#         names = ['x1', 'x2', 'y']
#         return pd.DataFrame(np.stack(data).T, columns=names)

# data = Shapes().generate_data(10000)

# data_sine = data.loc[data['y'] == 3, :]

# plt.scatter(data_sine['x1'], data_sine['x2'])
# plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='gen_2d_data', description='generates 2d dgps')
    parser.add_argument('savepath', help='in most cases repo/python/ is best')
    parser.add_argument('-p', '--plot', action='store_true')
    args = parser.parse_args()

    pw = ModelPawelczyk()
    data = pw.generate_data(10000)
    data.to_csv(args.savepath + 'pawelczyk.csv', index=False)
    if args.plot:
        plt.scatter(data['x1'], data['x2'], c=data['y'])
        plt.colorbar()  # Add color scale/legend
        plt.show()


    tm = TwoSines()
    data = tm.generate_data(10000)
    data.to_csv(args.savepath + 'two_sines.csv', index=False)
    likelihood = tm.get_log_likelihood(data)

    if args.plot:
        plt.scatter(data['x1'], data['x2'], c=data['y'])
        plt.colorbar()  # Add color scale/legend
        plt.show()
    

    data = Cassini().generate_data(10000)
    data.to_csv(args.savepath + 'cassini.csv', index=False)

    if args.plot:
        plt.scatter(data['x1'], data['x2'], c=data['y'])
        plt.show()

    steps = np.linspace(-1, 1, 100)
    x1, x2 = np.meshgrid(steps, steps)
    grid = pd.DataFrame({'x1': x1.ravel(), 'x2': x2.ravel()})

    likelihood = Cassini().get_log_likelihood(grid)
    if args.plot:
        plt.scatter(x1, x2, c=likelihood)
        plt.colorbar()  # Add color scale/legend
        plt.show()


