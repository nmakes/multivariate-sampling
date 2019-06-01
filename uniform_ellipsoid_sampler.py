'''
  Perform Uniform Sampling within an N-dimensional ellipsoid.
  Written for pytorch (cuda compatible).
  
  Written by Naveen Venkat.
  nav.naveenvenkat@gmail.com
  naveenvenkat.com
'''

import torch

class EllipsoidSampler:

    # Ellipsoid sampler (uniform sampling within an ellipse)

    def __init__(self, mu, axes, use_cuda=True):

        '''
            mu:   mean of the ellipse (centre of the ellipse in the N-dimensional space)
                  shape = (N,)
                  
            axes: axes length across each dimension
                  shape = (N,)
                  
            Points will be uniformly sampled within the ellipse constructed using centre = mu, and
            axes lengths = axes.
        '''

        assert(len(mu.shape) == len(axes.shape) == 1)
        assert(mu.shape == axes.shape)

        self.use_cuda = use_cuda

        self.mu = mu
        self.axes = axes
        self.dims = mu.shape[0]

        if self.use_cuda:
            self.mu = self.mu.cuda()
            self.axes = self.axes.cuda()

    def f(self, x):

        '''
            Get the value of the ellipsoid function for a given batch of x
        '''

        assert (len(x.shape) == 2), 'Not implemented for shape != 2D'
        if self.use_cuda:
            x = x.cuda()

        Xsq = torch.pow(x, 2)

        Asq = torch.pow(self.axes, 2)

        div = Xsq / Asq

        eq = torch.sum(div, dim=-1) - 1

        return eq

    def sample(self, num_points):

        '''
            Sample num_points from within the ellipsoid
        '''

        # 1. Sample n points from the surface of a unit sphere
        # 2. Scale each dimension using torch.rand() (a random number between 0-1) so that it lies within the sphere
        # 3. Multiply with self.axes to make it ellipsoidic
        # 4. Shift the mean to the mean of the ellipse

        # 1. Sample points on a unit sphere
        z = torch.randn((num_points, self.dims))
        if self.use_cuda:
            z = z.cuda()
        z_on_unit_sphere = z / torch.sum(z**2, dim=-1).view((num_points, 1))

        # 2. Scale each dimension by multiplying with a number between (0-1)
        scale = torch.rand((1,))
        if self.use_cuda:
            scale = scale.cuda()
        z_scaled = z_on_unit_sphere * scale

        # 3. Multiply with self.axes
        z_ellipsoidal = z_scaled * self.axes

        # 4. Shift the mean
        z_shifted = z_ellipsoidal + self.mu

        idx = (self.f(z_shifted) <= 0) # Cross-check

        assert(len(idx) == num_points), "Couldn't sample enough points from within the ellipsoid. Check the algorithm"

        return z_shifted
