import torch
import numpy as np
import matplotlib.pyplot as plt


def UCB(mu, sigma, kappa=1):
    acquisition = mu + kappa * torch.sqrt(sigma)
    next_query = torch.where(
        torch.isclose(acquisition, torch.max(acquisition), rtol=1e-2)
    )
    # plt.imshow(acquisition.view(64, 64))
    # plt.show()
    if len(next_query[0]) > 1:
        next_query = next_query[0][np.random.randint(len(next_query[0]))]

    else:
        next_query = next_query[0][0]

    return next_query
