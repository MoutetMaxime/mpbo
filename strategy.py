import numpy as np


def mpbo(optim_steps, keeped_queries, epoch, up=None, low=None):
    """
    Random strategy
    """

    # Sort the queries by response
    sorted_queries = np.argsort(optim_steps[epoch, keeped_queries, 1].cpu().numpy())

    if (low is None and up is None) or (low == 0 and up == 1):
        possible_del = keeped_queries[sorted_queries[:-1]]

    else:
        if low is None:
            low = 0
        if up is None:
            up = 1

        assert low < up

        possible_del = keeped_queries[
            sorted_queries[
                int(low * len(sorted_queries)) : int(up * len(sorted_queries))
            ]
        ]

    random_index = np.random.randint(possible_del.shape[0])

    return possible_del[random_index]
