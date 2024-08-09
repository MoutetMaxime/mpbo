"""
Created on Sun Jul 28 15:24:00 2024

@author: MoutetMaxime


Class for objective synthetic functions.

Initialization parameters:
    dim: the dimension of input data
    benchmark: the name of the benchmark function to use. Currently only
        'Ackley', 'Hartmann', and 'Michalewicz' are supported
    negate: whether or not to negate the output of the benchmark function
    lower_bound: the lower bound of the input space. If None, the default
        lower bound is used
    upper_bound: the upper bound of the input space. If None, the default
        upper bound is used

Methods:
    evaluate(x, noise_std): Evaluates the benchmark function at point x with
        noise standard deviation noise_std.
    evaluate_true(x): Evaluates the benchmark function at point x without noise.

    get_bounds(benchmark, dim, lower_bound, upper_bound): static methode that
        returns the bounds of the input space for the benchmark function.
    
"""

import torch
import numpy as np

from botorch.test_functions import Ackley, Hartmann, Michalewicz


class ObjectiveFunction:
    def __init__(self, benchmark, dim=2, negate=True, device=torch.device("cpu")):
        self.dim = dim
        self.negate = negate
        self.device = device

        if benchmark == "Ackley":
            self.f = Ackley(dim=self.dim, negate=self.negate)

            self.size = 64
            self.upper_bound = 32
            self.lower_bound = -32

        elif benchmark == "Hartmann":
            try:
                assert dim == 6
            except AssertionError:
                raise ValueError("Hartmann function is only defined for dim=6.")

            self.f = Hartmann(dim=self.dim, negate=self.negate)

            self.size = 5
            self.upper_bound = 1
            self.lower_bound = 0

        elif benchmark == "Michalewicz":
            self.f = Michalewicz(dim=self.dim, negate=self.negate)

            self.upper_bound = np.pi
            self.lower_bound = 0
            if self.dim == 2:
                self.size = 64

            elif self.dim == 4:
                self.size = 10

            else:
                raise ValueError("Choose dim=2 or dim=4 for Michalewicz function.")
        else:
            raise ValueError(
                "Choose a valid benchmark function in ['Ackley', 'Hartmann', 'Michalewicz']."
            )

    def create_input_space(self):
        X = np.linspace(self.lower_bound, self.upper_bound, self.size)
        X = np.array([X for _ in range(self.dim)])
        X = np.meshgrid(*X)
        ch2xy = np.array([x.flatten() for x in X]).T
        ch2xy = torch.from_numpy(ch2xy).float().to(self.device)

        return ch2xy

    def generate_true_response(self, input_space):
        response = np.array([self.f(torch.tensor([*x])) for x in input_space])
        response = (response - response.min()) / (response.max() - response.min())
        response = torch.from_numpy(response).float().to(self.device)
        return response


if __name__ == "__main__":
    obj = ObjectiveFunction("Ackley", dim=2, negate=True)

    ch2xy = obj.create_input_space()
    response = obj.generate_true_response(ch2xy)

    print(len(list(response.shape)))
