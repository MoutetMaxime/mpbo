# Memory-Pruning Bayesian Optimization Algorithm

Bayesian optimization (BO) is a powerful tool for optimizing noisy and costly-to-evaluate black-box functions, widely used in fields such as machine learning and engineering. However, Bayesian optimization faces significant challenges when applied to large datasets. Due to the computational and memory requirements associated with updating Gaussian Process (GP) models, computation times can quickly become unmanageable. To address these limitations, we propose a new Bayesian optimization algorithm with memory traversal (MP-BO) that iteratively eliminates data points from the training set, thus maintaining a constant algorithmic complexity of $\bigO(m^3)$ for some $m\ll n$ where $n$ is the size of the training set. This pruning strategy effectively reduces memory usage and computation time without significantly sacrificing performance. We evaluate the MP-BO algorithm on synthetic benchmarks and real datasets, demonstrating its robustness and efficiency in scenarios where the computation times of traditional BO would be too large. Our results suggest that MP-BO is a promising approach for applications requiring efficient optimization with limited computing resources.

# Main results:

