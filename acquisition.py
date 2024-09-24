import torch
import numpy as np
import matplotlib.pyplot as plt


def UCB(mu, sigma, kappa=1):
    """
    Upper Confidence Bound acquisition function
    """
    acquisition = mu + kappa * torch.nan_to_num(torch.sqrt(sigma))
    next_query = torch.where(
        torch.isclose(
            acquisition.reshape(len(acquisition)),
            torch.max(acquisition.reshape(len(acquisition))),
            rtol=1e-2,
        )
    )
    # plt.imshow(acquisition.view(64, 64))
    # plt.show()
    if len(next_query[0]) > 1:
        next_query = next_query[0][np.random.randint(len(next_query[0]))]

    else:
        next_query = next_query[0][0]

    return next_query
