import numpy as np

from scipy.stats import gmean


def mpbo(optim_steps, keeped_queries, epoch, strategy):
    """
    Random strategy
    """

    if strategy == "MP-BO":
        index_to_del = random_removing(optim_steps, keeped_queries, epoch)
    elif strategy == "FiFo":
        index_to_del = fifo_removing(keeped_queries)
    elif strategy == "Mean":
        index_to_del = mean_removing(optim_steps, keeped_queries, epoch)
    elif strategy == "GeoMean":
        index_to_del = geometric_mean_removing(optim_steps, keeped_queries, epoch)
    elif strategy == "Worst":
        index_to_del = worst_removing(optim_steps, keeped_queries, epoch)

    return index_to_del


def random_removing(optim_steps, keeped_queries, epoch):
    # Sort the queries by response
    sorted_queries = np.argsort(optim_steps[epoch, keeped_queries, 1].cpu().numpy())
    possible_del = keeped_queries[sorted_queries[:-1]]
    random_index = np.random.randint(possible_del.shape[0])

    return possible_del[random_index]


def fifo_removing(keeped_queries):
    return keeped_queries[0]


def mean_removing(optim_steps, keeped_queries, epoch):
    mean_response = optim_steps[epoch, keeped_queries, 1].mean(axis=0)

    # Find the query with the closest response to the mean
    closest_query = np.argmin(
        np.abs(optim_steps[epoch, keeped_queries, 1] - mean_response)
    )

    return keeped_queries[closest_query.item()]


def geometric_mean_removing(optim_steps, keeped_queries, epoch):
    geo_mean_response = gmean(optim_steps[epoch, keeped_queries, 1])

    # Find the query with the closest response to the geometric mean
    closest_query = np.argmin(
        np.abs(optim_steps[epoch, keeped_queries, 1] - geo_mean_response)
    )

    return keeped_queries[closest_query.item()]


def worst_removing(optim_steps, keeped_queries, epoch):
    # Find the query with the worst response
    worst_query = np.argmin(optim_steps[epoch, keeped_queries, 1])

    return keeped_queries[worst_query.item()]
