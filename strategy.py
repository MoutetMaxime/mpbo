import numpy as np


def mpbo(optim_steps, keeped_queries, epoch):
    """
    Random strategy
    """

    # Sort the queries by response
    sorted_queries = np.argsort(optim_steps[epoch, keeped_queries, 1].cpu().numpy())
    possible_del = keeped_queries[sorted_queries[:-1]]
    random_index = np.random.randint(possible_del.shape[0])

    return possible_del[random_index]
