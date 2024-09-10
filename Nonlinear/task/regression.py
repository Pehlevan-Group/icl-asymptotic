"""
In-context regression tasks

author: William Tong (wtong@g.harvard.edu)
modified by Mary Letey for 2024 In-Context Learning Project
"""

# <codecell>

# Connection To Theory Paper
# n_points:     N + 1 = Length of a given context + 1 (includes the final x_{N+1} query vector)
# n_dims:       d = dimension of a token vectors
# eta_scale:    standard deviation of noise scalars eta
# w_scale:      standard deviation of beta vectors in isotropic case beta~N(0, sigma_beta I)
# batch_size:   P = number of contexts in prompt

import numpy as np
import random

# Implements DIVERSE ISOTROPIC case with C = I
class LinearRegressionCorrect:
    def __init__(self, n_points=6, n_dims=2, eta_scale=1, w_scale=1, data_cov=1, batch_size=128, seed=None) -> None:
        self.n_points = n_points # n_points = N+1 where N = context length, as n_points includes the (N+1)st query vector
        self.n_dims = n_dims # d = dimension of tokens
        self.w_scale = w_scale # sigma_beta
        self.eta_scale = eta_scale # noise sigma
        self.data_cov = data_cov # C = 1 usually but want to customise like 1/sqrt(alpha)
        self.batch_size = batch_size # P = number of contexts
        self.rng = np.random.default_rng(seed)
    
    def __next__(self):
        xs = (self.data_cov)*self.rng.normal(loc=0, scale = 1/np.sqrt(self.n_dims), size=(self.batch_size, self.n_points, self.n_dims))
        ws = self.rng.normal(loc=0, scale = self.w_scale, size=(self.batch_size, self.n_dims, 1))
        ys = xs @ ws + self.rng.normal(loc=0, scale = self.eta_scale, size=(self.batch_size, self.n_points, 1))
        Z = np.zeros((self.batch_size, self.n_points, self.n_dims + 1))
        Z[:,:,0:self.n_dims] = xs
        Z[:,:,-1] = ys.squeeze()
        Z[:,-1, self.n_dims] = 0 # padding for final context
	    
	    # returns the Z [x,y,x,y]... configuration and the true N+1 value for testing 
        return Z, ys[:,-1].squeeze()

    def __iter__(self):
        return self

# We introduce a new parameter here as well
# diversity: K = number of distinct beta_k to be sampled UNIFORMLY for each context 
class FiniteSampler:
    def __init__(self, n_points=6, variable_context=False, n_dims=2, eta_scale=1, w_scale=1, diversity=6, batch_size=128, seed=None) -> None:
        self.n_points = n_points # n_points = N+1 where N = context length, as n_points includes the (N+1)st query vector
        self.variable_context = variable_context
        self.n_dims = n_dims # d = dimension of tokens
        self.w_scale = w_scale # sigma_beta
        self.eta_scale = eta_scale # noise sigma
        self.batch_size = batch_size # P = number of contexts
        self.diversity = diversity
        self.rng = np.random.default_rng(seed)
        # Now we fix a set of betas which will be sampled from during all other calls to iter or next once this object is instantiated. 
        # Once we get to the actual sampling, we will use
        self.E = self.rng.normal(loc=0, scale = self.w_scale, size=(self.n_dims, self.diversity)) 
    
    def __next__(self):
        if self.variable_context:
            context_length = int(random.uniform(self.n_points-int(self.n_points/2), self.n_points+int(self.n_points/2)))
        else:
            context_length = self.n_points
        uniform_ps = np.array([random.randrange(self.diversity) for _ in range(self.batch_size)])
        ws = np.array([self.E[:,uniform_ps[i]] for i in range(len(uniform_ps))]) 
        ws = ws[:,:,np.newaxis] # batch_size x n_dims x 1 as before
        print("ws are ", ws.shape)
        xs = self.rng.normal(loc=0, scale = 1/np.sqrt(self.n_dims), size=(self.batch_size, context_length, self.n_dims))
        print("xs are ", xs.shape)
        ys = xs @ ws + self.rng.normal(loc=0, scale = self.eta_scale, size=(self.batch_size, context_length, 1))
        Z = np.zeros((self.batch_size, context_length, self.n_dims + 1))
        Z[:,:,0:self.n_dims] = xs
        Z[:,:,-1] = ys.squeeze()
        Z[:,-1, self.n_dims] = 0 # padding for final context
    # returns the Z [x,y,x,y]... configuration and the true N+1 value for testing 
        return Z, ys[:,-1].squeeze()

    def __iter__(self):
        return self


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt

#     # task = LinearRegression(batch_size=5, n_dims=1)
#     # xs, ys = next(task)

#     # plt.scatter(xs[0][0:-1:2], xs[0][1::2])
#     # plt.scatter([xs[0][-1]], ys[0])

#     # task = LinearRegression(batch_size=5, n_dims=2, n_points=500, seed=1)
#     # xs, ys = next(task)

#     #task = FiniteSampler(n_points=30, n_dims=1, eta_scale=1, w_scale=1, diversity=4, batch_size=128)
#     task = LinearRegressionCorrect(n_points=500, n_dims=2, eta_scale=1, w_scale=1, batch_size=128)
#     zs, ys = next(task)

#     print(zs[0:10,:,:])
#     xs = zs[0,:,0:-1]
#     print(xs.shape)

#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     for p in range(1):
#         ax.scatter3D(zs[p,:,0],zs[p,:,1],zs[p,:,-1],label=f'line {p}')
#     plt.legend()
#     plt.savefig('lines.png')

#     # ax.scatter(xs[0][0:-1:2, 0], xs[0][0:-1:2, 1], xs[0][1::2, 0], alpha=0.3)
#     # ax.scatter(xs[0][-1,0], xs[0][-1, 1], ys[0])

