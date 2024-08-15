"""
Created on Sun Jul 28 15:24:00 2024

@author: MoutetMaxime


Class for Bayesian Optimization.

Initialization parameters:
    search_space: the input space of the optimization problem
    response: the response of the optimization problem
    device: the device to use for the optimization
    noise_std: the noise standard deviation to add in response

Methods:
    initialize(initial_points, repetitions): Generates initial points for the optimization
    train(): Trains the Bayesian Optimization model and returns the exploration score
        and the optimization steps
    get_best_point(mu, optim_steps): Returns the best point in the optimization steps
"""

import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time

from model import GP, optimize
from acquisition import UCB
from strategy import mpbo


class BayesianOptimizer:
    def __init__(
        self,
        search_space,
        response,
        device=torch.device("cpu"),
        noise_std=0.025,
        kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5)),
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        isvalid=None,
        respMean_valid=None,
    ):
        self.search_space = search_space
        self.obj = response
        self.noise_std = noise_std
        self.device = device
        self.kernel = kernel
        self.likelihood = likelihood

        self.isvalid = isvalid
        self.respMean_valid = respMean_valid

    def initialize(self, initial_points=1, repetitions=30):
        inits = np.random.randint(
            0, self.search_space.shape[0], size=(repetitions, initial_points)
        )
        return inits

    def train(
        self,
        kappa,
        initial_points=1,
        repetitions=30,
        iterations=150,
        training_iter=10,
        strategy="Vanilla BO",
        begin_strat=20,
        follow_baseline=None,
        return_exploit=False,
        save_file=False,
        file_name=None,
    ):
        optim_steps = torch.zeros((repetitions, iterations, 2), device=self.device)
        inits = self.initialize(initial_points, repetitions)

        exploration_score = torch.zeros((repetitions, iterations))
        exploitation_score = torch.zeros((repetitions, iterations))
        distance_from_best = torch.zeros((repetitions, iterations))
        time_per_query = torch.zeros((repetitions, iterations), device=self.device)
        MSEloss = torch.zeros(repetitions, device=self.device)

        try:
            assert strategy in ["Vanilla BO", "MP-BO"]
            if strategy != "Vanilla BO":
                assert 0 < begin_strat < iterations
        except:
            raise AssertionError("Invalid strategy")

        # We separate cases where we have access to several measurements
        if len(list(self.obj.shape)) == 2:
            # Multiple measurement, we take the mean as ground truth and rescale it
            objective_func = self.respMean_valid
            objective_func = (objective_func - objective_func.min()) / (
                objective_func.max() - objective_func.min()
            )

            max_obj = torch.max(objective_func)
            bests = torch.where(objective_func == max_obj)

        else:
            objective_func = self.obj
            max_obj = torch.max(objective_func)
            bests = torch.where(objective_func == max_obj)

        if len(bests) > 1:
            best = bests[0]
        else:
            best = bests

        for rep in tqdm(range(repetitions)):
            deleted_queries = []
            init_queries = inits[rep]
            best_queries_list = []
            query = 0
            max_seen = 0

            while query < iterations:  # MaxQueries:
                if follow_baseline is not None and query < initial_points:
                    optim_steps[rep, query, 0] = follow_baseline[rep, query, 0]
                    optim_steps[rep, query, 1] = follow_baseline[rep, query, 1]
                else:
                    if query >= initial_points:
                        next_query = UCB(mu, sigma, kappa=kappa)
                        optim_steps[rep, query, 0] = next_query
                    else:
                        optim_steps[rep, query, 0] = int(init_queries[query])

                    sampled_point = optim_steps[rep, query, 0]

                    # if several measurements in response
                    if len(list(self.obj.shape)) == 2:
                        query_in_search_space = int(sampled_point.item())

                        valid_idx = list(
                            np.where(self.isvalid[:, query_in_search_space] == 1)[0]
                        ) + list(
                            np.where(self.isvalid[:, query_in_search_space] == -1)[0]
                        )
                        test_response = self.obj[
                            np.random.choice(valid_idx),
                            query_in_search_space,
                        ]
                    else:

                        test_response = (
                            self.obj[int(sampled_point.item())]
                            + torch.randn(1) * self.noise_std
                        )

                    # done reading response
                    optim_steps[rep, query, 1] = test_response

                # MP-BO
                if query >= begin_strat and strategy == "MP-BO":
                    deleted_query = mpbo(optim_steps, keeped_queries, rep)
                    deleted_queries.append(deleted_query)

                keeped_queries = np.delete(np.arange(0, query + 1, 1), deleted_queries)

                # Sanity check
                if strategy == "MP-BO":
                    assert len(keeped_queries) == min(begin_strat, query + 1)
                else:
                    assert keeped_queries.all() == np.arange(0, query + 1, 1).all()

                train_y = optim_steps[rep, keeped_queries, 1].float()
                train_x = self.search_space[
                    optim_steps[rep, keeped_queries, 0].long(), :
                ].float()

                if (torch.max(torch.abs(train_y)) > max_seen) or max_seen == 0:
                    max_seen = torch.max(torch.abs(train_y))

                if query == 0:
                    likelihood = self.likelihood
                    model = GP(train_x, train_y, self.likelihood, self.kernel).to(
                        self.device
                    )
                else:
                    model.set_train_data(
                        train_x,
                        train_y,
                        strict=False,
                    )

                start = time()
                model.train()
                likelihood.train()

                model, likelihood = optimize(
                    model,
                    likelihood,
                    training_iter,
                    train_x,
                    train_y,
                    verbose=False,
                )

                model.eval()
                likelihood.eval()

                with torch.no_grad():
                    test_x = self.search_space
                    observed_pred = likelihood(model(test_x))

                sigma = observed_pred.variance
                mu = observed_pred.mean

                duration = time() - start

                best_query = self.get_best_point(mu, optim_steps[rep, : query + 1, 0])
                best_queries_list.append(best_query)

                # Update metrics
                time_per_query[rep, query - 1] = duration
                distance_from_best[rep, query - 1] = torch.norm(
                    self.search_space[best_query] - self.search_space[best], p=2
                )

                exploration_score[rep, query] = 1 - objective_func[best_query] / max_obj
                exploitation_score[rep, query] = (
                    1 - objective_func[optim_steps[rep, query, 0].long()] / max_obj
                )

                query += 1

            MSEloss[rep] = torch.mean((mu - objective_func) ** 2)

        if save_file:
            np.savez(
                f"{file_name}.npz",
                kappa=kappa,
                repetitions=repetitions,
                iterations=iterations,
                strategy=strategy,
                begin_strat=begin_strat,
                regret=exploration_score,
                explt_regret=exploitation_score,
                MSE=MSEloss,
                distance_from_best=distance_from_best,
                time=time_per_query,
            )
        if return_exploit:
            return exploration_score, optim_steps, exploitation_score
        else:
            return exploration_score, optim_steps

    @staticmethod
    def get_best_point(mu, optim_steps):
        # Only test on already sampled points
        tested = torch.unique(optim_steps).long()
        mu_tested = mu[tested]

        if len(tested) == 1:
            best_query = tested
        else:
            best_query = tested[
                (mu_tested == torch.max(mu_tested)).reshape(len(mu_tested))
            ]
            if len(best_query) > 1:
                best_query = np.array(
                    [best_query[np.random.randint(len(best_query))].cpu()]
                )
            else:
                best_query = np.array([best_query[0].cpu()])

        return best_query.item()


class ExtensiveSearch(BayesianOptimizer):
    def __init__(
        self,
        search_space,
        response,
        device=torch.device("cpu"),
        noise_std=0.025,
    ):
        super().__init__(search_space, response, device=device, noise_std=noise_std)

    def train(
        self,
        kappa,
        initial_points=1,
        repetitions=30,
        iterations=150,
        training_iter=10,
        begin_strat=20,
        follow_baseline=None,
        save_file=False,
        file_name=None,
    ):
        optim_steps = torch.zeros((repetitions, iterations, 2), device=self.device)
        inits = self.initialize(initial_points, repetitions)

        exploration_score = torch.zeros((repetitions, iterations))
        exploitation_score = torch.zeros((repetitions, iterations))
        distance_from_best = torch.zeros((repetitions, iterations))
        time_per_query = torch.zeros((repetitions, iterations), device=self.device)
        MSEloss = torch.zeros(repetitions, device=self.device)

        # We separate cases where we have access to several measurements
        if len(list(self.obj.shape)) == 2:
            # Multiple measurement, we take the mean as ground truth and rescale it
            objective_func = self.obj.mean(axis=0)
            objective_func = (objective_func - objective_func.min()) / (
                objective_func.max() - objective_func.min()
            )

            max_obj = torch.max(objective_func)
            bests = torch.where(objective_func == max_obj)

        else:
            objective_func = self.obj
            max_obj = torch.max(objective_func)
            bests = torch.where(objective_func == max_obj)

        if len(bests) > 1:
            best = bests[0]
        else:
            best = bests

        for rep in tqdm(range(repetitions)):
            init_queries = inits[rep]
            best_queries_list = []
            query = 0

            while query < iterations:  # MaxQueries:
                if follow_baseline is not None and query < initial_points:
                    optim_steps[rep, query, 0] = follow_baseline[rep, query, 0]
                    optim_steps[rep, query, 1] = follow_baseline[rep, query, 1]
                else:
                    if initial_points <= query < begin_strat:
                        next_query = UCB(mu, sigma, kappa=kappa)
                        optim_steps[rep, query, 0] = next_query

                    elif query >= begin_strat:
                        # Extensive search
                        optim_steps[rep, query, 0] = np.random.randint(
                            len(self.search_space)
                        )
                    else:
                        optim_steps[rep, query, 0] = int(init_queries[query])

                    sampled_point = optim_steps[rep, query, 0]

                    # if several measurements in response
                    if len(list(self.obj.shape)) == 2:
                        test_response = self.obj[
                            np.random.randint(self.obj.shape[0]),
                            int(sampled_point.item()),
                        ]
                    else:
                        test_response = (
                            self.obj[int(sampled_point.item())]
                            + torch.randn(1) * self.noise_std
                        )

                    if test_response <= 0:
                        test_response = torch.tensor([0.0001], device=self.device)

                    # done reading response
                    optim_steps[rep, query, 1] = test_response

                train_y = optim_steps[rep, :, 1].float()
                train_x = self.search_space[optim_steps[rep, :, 0].long(), :].float()

                if query == 0:
                    likelihood = self.likelihood
                    model = GP(train_x, train_y, self.likelihood).to(self.device)
                elif query < begin_strat:
                    model.set_train_data(
                        train_x,
                        train_y,
                        strict=False,
                    )

                start = time()

                if query < begin_strat:
                    model.train()
                    likelihood.train()

                    model, likelihood = optimize(
                        model,
                        likelihood,
                        training_iter,
                        train_x,
                        train_y,
                        verbose=False,
                    )

                    model.eval()
                    likelihood.eval()

                    with torch.no_grad():
                        test_x = self.search_space
                        observed_pred = likelihood(model(test_x))

                    sigma = observed_pred.variance
                    mu = observed_pred.mean

                    duration = time() - start

                    best_query = self.get_best_point(
                        mu, optim_steps[rep, : query + 1, 0]
                    )
                    best_queries_list.append(best_query)
                    distance_from_best[rep, query - 1] = torch.norm(
                        self.search_space[best_query] - self.search_space[best], p=2
                    )
                    exploration_score[rep, query] = (
                        1 - objective_func[best_query] / max_obj
                    )
                else:
                    if (
                        objective_func[optim_steps[rep, query, 0].long()]
                        >= objective_func[best_query]
                    ):
                        exploration_score[rep, query] = (
                            1
                            - objective_func[optim_steps[rep, query, 0].long()]
                            / max_obj
                        )
                        best_query = optim_steps[rep, query, 0].long()
                        best_queries_list.append(best_query)

                    else:
                        exploration_score[rep, query] = (
                            1 - objective_func[best_query] / max_obj
                        )

                time_per_query[rep, query - 1] = duration
                exploitation_score[rep, query] = (
                    1 - objective_func[optim_steps[rep, query, 0].long()] / max_obj
                )

                query += 1

            MSEloss[rep] = torch.mean((mu - objective_func) ** 2)

        if save_file:
            np.savez(
                f"{file_name}.npz",
                kappa=kappa,
                repetitions=repetitions,
                iterations=iterations,
                begin_strat=begin_strat,
                regret=exploration_score,
                explt_regret=exploitation_score,
                MSE=MSEloss,
                distance_from_best=distance_from_best,
                time=time_per_query,
            )
        return exploration_score


if __name__ == "__main__":
    from ObjectiveFunction import ObjectiveFunction

    # Define the search space
    obj = ObjectiveFunction("Ackley", dim=2)
    search_space = obj.create_input_space()
    response = obj.generate_true_response(search_space)

    # Define the Bayesian Optimization model
    bo = BayesianOptimizer(search_space, response)

    regret_bo, baseline = bo.train(
        kappa=4,
        initial_points=1,
        repetitions=30,
        iterations=150,
        training_iter=10,
        strategy="Vanilla BO",
    )

    # Define the Extensive Search model
    es = ExtensiveSearch(search_space, response)

    regret_es = es.train(
        kappa=4,
        initial_points=1,
        repetitions=30,
        iterations=150,
        training_iter=10,
        begin_strat=20,
        follow_baseline=baseline,
    )

    plt.plot(regret_bo.mean(dim=0), label="Vanilla BO")
    plt.plot(regret_es.mean(dim=0), label="Extensive Search")
    plt.legend()
    plt.show()
