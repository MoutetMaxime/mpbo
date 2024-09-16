import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib


def create_time_comparison_figure():
    gpbo_ackley = np.load("Ackley_bo_3_10_300.npz")
    mpbo_ackley = np.load("Ackley_mpbo_3_10_800.npz")

    time_gpbo_ackley = np.cumsum(gpbo_ackley["time"].mean(axis=0))
    time_mpbo_ackley = np.cumsum(mpbo_ackley["time"].mean(axis=0))

    time_mpbo_ackley = time_mpbo_ackley[time_mpbo_ackley <= 8]
    time_gpbo_ackley = time_gpbo_ackley[time_gpbo_ackley <= 8]

    regret_gpbo_ackley = gpbo_ackley["regret"].mean(axis=0)[: len(time_gpbo_ackley)]
    regret_mpbo_ackley = mpbo_ackley["regret"].mean(axis=0)[: len(time_mpbo_ackley)]

    # plt.plot(time_gpbo_ackley, regret_gpbo, "r", linestyle="--", label="Vanilla BO")
    # plt.plot(time_mpbo_ackley, regret_mpbo, "tab:blue", label="MP-BO")

    print(len(time_gpbo_ackley), len(time_mpbo_ackley))

    gpbo_mich2d = np.load("Mich2D_bo_4_10_300.npz")
    mpbo_mich2d = np.load("Mich2D_mpbo_4_10_800.npz")

    time_gpbo_mich2d = np.cumsum(gpbo_mich2d["time"].mean(axis=0))
    time_mpbo_mich2d = np.cumsum(mpbo_mich2d["time"].mean(axis=0))

    time_mpbo_mich2d = time_mpbo_mich2d[time_mpbo_mich2d <= 8]
    time_gpbo_mich2d = time_gpbo_mich2d[time_gpbo_mich2d <= 8]

    regret_gpbo_mich2d = gpbo_mich2d["regret"].mean(axis=0)[: len(time_gpbo_mich2d)]
    regret_mpbo_mich2d = mpbo_mich2d["regret"].mean(axis=0)[: len(time_mpbo_mich2d)]

    print(len(time_gpbo_mich2d), len(time_mpbo_mich2d))

    gpbo_mich4d = np.load("Mich4D_bo_4_10_300.npz")
    mpbo_mich4d = np.load("Mich4D_mpbo_4_10_800.npz")

    time_gpbo_mich4d = np.cumsum(gpbo_mich4d["time"].mean(axis=0))
    time_mpbo_mich4d = np.cumsum(mpbo_mich4d["time"].mean(axis=0))

    time_mpbo_mich4d = time_mpbo_mich4d[time_mpbo_mich4d <= 10.5]
    time_gpbo_mich4d = time_gpbo_mich4d[time_gpbo_mich4d <= 10.5]

    regret_gpbo_mich4d = gpbo_mich4d["regret"].mean(axis=0)[: len(time_gpbo_mich4d)]
    regret_mpbo_mich4d = mpbo_mich4d["regret"].mean(axis=0)[: len(time_mpbo_mich4d)]

    print(len(time_gpbo_mich4d), len(time_mpbo_mich4d))

    gpbo_hart = np.load("Hart_bo_4_10_300.npz")
    mpbo_hart = np.load("Hart_mpbo_4_10_800.npz")

    time_gpbo_hart = np.cumsum(gpbo_hart["time"].mean(axis=0))
    time_mpbo_hart = np.cumsum(mpbo_hart["time"].mean(axis=0))

    time_mpbo_hart = time_mpbo_hart[time_mpbo_hart <= 12]
    time_gpbo_hart = time_gpbo_hart[time_gpbo_hart <= 12]

    regret_gpbo_hart = gpbo_hart["regret"].mean(axis=0)[: len(time_gpbo_hart)]
    regret_mpbo_hart = mpbo_hart["regret"].mean(axis=0)[: len(time_mpbo_hart)]

    print(len(time_gpbo_hart), len(time_mpbo_hart))

    font = {"weight": "normal", "size": 12}
    matplotlib.rc("font", **{"family": "serif", "serif": ["Times New Roman"]})
    matplotlib.rc("text", usetex=True)
    fig, axs = plt.subplots(1, 4, sharey=True, figsize=(12, 3))

    axs[0].plot(
        time_mpbo_ackley,
        regret_mpbo_ackley,
        "tab:blue",
        label="MP-BO",
    )
    axs[0].fill_between(
        time_mpbo_ackley,
        regret_mpbo_ackley
        - mpbo_ackley["regret"].std(axis=0)[: len(time_mpbo_ackley)] / np.sqrt(10),
        regret_mpbo_ackley
        + mpbo_ackley["regret"].std(axis=0)[: len(time_mpbo_ackley)] / np.sqrt(10),
        color="tab:blue",
        alpha=0.2,
    )
    axs[0].plot(
        time_gpbo_ackley, regret_gpbo_ackley, "r", linestyle="--", label="Vanilla BO"
    )
    axs[0].fill_between(
        time_gpbo_ackley,
        regret_gpbo_ackley
        - gpbo_ackley["regret"].std(axis=0)[: len(time_gpbo_ackley)] / np.sqrt(10),
        regret_gpbo_ackley
        + gpbo_ackley["regret"].std(axis=0)[: len(time_gpbo_ackley)] / np.sqrt(10),
        color="r",
        alpha=0.2,
    )

    axs[1].plot(
        time_mpbo_mich2d,
        regret_mpbo_mich2d,
        "tab:blue",
        label="MP-BO",
    )
    axs[1].fill_between(
        time_mpbo_mich2d,
        regret_mpbo_mich2d
        - mpbo_mich2d["regret"].std(axis=0)[: len(time_mpbo_mich2d)] / np.sqrt(10),
        regret_mpbo_mich2d
        + mpbo_mich2d["regret"].std(axis=0)[: len(time_mpbo_mich2d)] / np.sqrt(10),
        color="tab:blue",
        alpha=0.2,
    )

    axs[1].plot(
        time_gpbo_mich2d, regret_gpbo_mich2d, "r", linestyle="--", label="Vanilla BO"
    )
    axs[1].fill_between(
        time_gpbo_mich2d,
        regret_gpbo_mich2d
        - gpbo_mich2d["regret"].std(axis=0)[: len(time_gpbo_mich2d)] / np.sqrt(10),
        regret_gpbo_mich2d
        + gpbo_mich2d["regret"].std(axis=0)[: len(time_gpbo_mich2d)] / np.sqrt(10),
        color="r",
        alpha=0.2,
    )

    axs[2].plot(
        time_mpbo_mich4d,
        regret_mpbo_mich4d,
        "tab:blue",
        label="MP-BO",
    )

    axs[2].fill_between(
        time_mpbo_mich4d,
        regret_mpbo_mich4d
        - mpbo_mich4d["regret"].std(axis=0)[: len(time_mpbo_mich4d)] / np.sqrt(10),
        regret_mpbo_mich4d
        + mpbo_mich4d["regret"].std(axis=0)[: len(time_mpbo_mich4d)] / np.sqrt(10),
        color="tab:blue",
        alpha=0.2,
    )

    axs[2].plot(
        time_gpbo_mich4d, regret_gpbo_mich4d, "r", linestyle="--", label="Vanilla BO"
    )

    axs[2].fill_between(
        time_gpbo_mich4d,
        regret_gpbo_mich4d
        - gpbo_mich4d["regret"].std(axis=0)[: len(time_gpbo_mich4d)] / np.sqrt(10),
        regret_gpbo_mich4d
        + gpbo_mich4d["regret"].std(axis=0)[: len(time_gpbo_mich4d)] / np.sqrt(10),
        color="r",
        alpha=0.2,
    )

    axs[3].plot(
        time_mpbo_hart,
        regret_mpbo_hart,
        "tab:blue",
        label="MP-BO",
    )

    axs[3].fill_between(
        time_mpbo_hart,
        regret_mpbo_hart
        - mpbo_hart["regret"].std(axis=0)[: len(time_mpbo_hart)] / np.sqrt(10),
        regret_mpbo_hart
        + mpbo_hart["regret"].std(axis=0)[: len(time_mpbo_hart)] / np.sqrt(10),
        color="tab:blue",
        alpha=0.2,
    )

    axs[3].plot(
        time_gpbo_hart, regret_gpbo_hart, "r", linestyle="--", label="Vanilla BO"
    )

    axs[3].fill_between(
        time_gpbo_hart,
        regret_gpbo_hart
        - gpbo_hart["regret"].std(axis=0)[: len(time_gpbo_hart)] / np.sqrt(10),
        regret_gpbo_hart
        + gpbo_hart["regret"].std(axis=0)[: len(time_gpbo_hart)] / np.sqrt(10),
        color="r",
        alpha=0.2,
    )

    axs[0].set_title("\\textbf{Ackley}")
    axs[1].set_title("\\textbf{Michalewicz 2D}")
    axs[2].set_title("\\textbf{Michalewicz 4D}")
    axs[3].set_title("\\textbf{Hartmann}")

    axs[0].set_ylabel("\\textbf{Regret}", labelpad=-5)
    # axs[1, 0].set_ylabel("\\textbf{Regret}", labelpad=-5)
    axs[0].set_yticks([0, 0.5, 1], labels=["0", "", "1"])
    # axs[1].set_yticks([0, 0.5, 1], labels=["0", "", "1"])

    axs[0].set_xlabel("\\textbf{Time (s)}")
    axs[1].set_xlabel("\\textbf{Time (s)}")
    axs[2].set_xlabel("\\textbf{Time (s)}")
    axs[3].set_xlabel("\\textbf{Time (s)}")

    legend, _ = axs[0].get_legend_handles_labels()
    plt.legend(frameon=False, fontsize=14, ncols=2, bbox_to_anchor=(-0.65, 1.35))

    plt.tight_layout()

    plt.savefig("time_comp_lal.pdf", bbox_inches="tight")
    plt.show()


files = {
    "Ackley": {
        "gpbo": "Ackley_BO.npz",
        "mpbo": "Ackley_MPBO.npz",
        # "fifo": "Ackley_FIFO.npz",
        # "mean": "Ackley_Mean.npz",
        # "geomean": "Ackley_Geometric_Mean.npz",
        # "worst": "Ackley_Worst.npz",
    },
    "Michalewicz 2D": {
        "gpbo": "Michalewicz2D_BO.npz",
        "mpbo": "Michalewicz2D_MPBO.npz",
        # "fifo": "Michalewicz2D_FIFO.npz",
        # "mean": "Michalewicz2D_Mean.npz",
        # "geomean": "Michalewicz2D_Geometric_Mean.npz",
        # "worst": "Michalewicz2D_Worst.npz",
    },
    "Michalewicz 4D": {
        "gpbo": "Michalewicz4D_BO.npz",
        "mpbo": "Michalewicz4D_MPBO.npz",
        # "fifo": "Michalewicz4D_FIFO.npz",
        # "mean": "Michalewicz4D_Mean.npz",
        # "geomean": "Michalewicz4D_Geometric_Mean.npz",
        # "worst": "Michalewicz4D_Worst.npz",
    },
    "Hartmann": {
        "gpbo": "Hartmann6D_BO.npz",
        "mpbo": "Hartmann6D_MPBO.npz",
        # "fifo": "Hartmann6D_FIFO.npz",
        # "mean": "Hartmann6D_Mean.npz",
        # "geomean": "Hartmann6D_Geometric_Mean.npz",
        # "worst": "Hartmann6D_Worst.npz",
    },
}


def create_regret_comparison_figure(files):
    font = {"weight": "normal", "size": 12}
    matplotlib.rc("font", **{"family": "serif", "serif": ["Times New Roman"]})
    matplotlib.rc("text", usetex=True)

    colors = {
        "gpbo": "r",
        "mpbo": "tab:blue",
        "fifo": "g",
        "mean": "orange",
        "geomean": "purple",
        "worst": "brown",
    }
    linestyles = {
        "gpbo": "--",
        "mpbo": "-",
        "fifo": "-.",
        "mean": (5, (10, 3)),
        "geomean": (0, (5, 1)),
        "worst": (0, (3, 5, 1, 5)),
    }
    labels = {
        "gpbo": "Vanilla BO",
        "mpbo": "MP-BO",
        "fifo": "FiFo",
        "mean": "Mean",
        "geomean": "GeoMean",
        "worst": "Worst",
    }

    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    benchmarks = list(files.keys())

    q_star = 20
    for i in range(len(axs.flatten())):
        ax = axs.flatten()[i]
        ax.vlines(q_star, -0.1, 1.1, linestyle="dotted", color="k", label="q* = 20")

        strats = list(files[benchmarks[i]].keys())
        for strat in strats:
            print(files[benchmarks[i]][strat])
            bo = np.load(files[benchmarks[i]][strat])
            regret_bo = bo["regret"].mean(axis=0)

            epochs = bo["regret"].shape[0]
            queries = bo["regret"].shape[1]

            ax.plot(
                regret_bo,
                colors[strat],
                linestyle=linestyles[strat],
                label=labels[strat],
            )
            ax.fill_between(
                range(queries),
                regret_bo - bo["regret"].std(axis=0) / np.sqrt(epochs),
                regret_bo + bo["regret"].std(axis=0) / np.sqrt(epochs),
                color=colors[strat],
                alpha=0.2,
            )

            ax.set_title("\\textbf{" + benchmarks[i] + "}")
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0, 0.5, 1], labels=["0", "", "1"])
            ax.set_xticks([0, 75, 150], labels=["0", "", "150"])

    axs[0].set_ylabel("\\textbf{Regret}", labelpad=-5)

    axs[0].set_xlabel("\\textbf{Iteration}", labelpad=-5)
    axs[1].set_xlabel("\\textbf{Iteration}", labelpad=-5)
    axs[2].set_xlabel("\\textbf{Iteration}", labelpad=-5)
    axs[3].set_xlabel("\\textbf{Iteration}", labelpad=-5)
    legend, _ = axs[1].get_legend_handles_labels()
    plt.legend(
        frameon=False,
        fontsize=14,
        ncols=5,
        bbox_to_anchor=(0, 1.4),
    )
    # plt.savefig("bo_vs_mpbo_allstrats.svg", bbox_inches="tight")
    plt.savefig("bo_vs_mpbo.pdf", bbox_inches="tight")
    plt.show()


def create_neurostim_figure():
    path = "outpus/results.csv"
    df = pd.read_csv(path, sep=",", header=None)
    df = np.array(df)

    regret_gpbo = 1 - df[:22, :]
    regret_mpbo = 1 - df[22:44, :]
    regret_es = 1 - df[44:, :]

    regret_gpbo = np.mean(regret_gpbo, axis=0)
    regret_mpbo = np.mean(regret_mpbo, axis=0)
    regret_es = np.mean(regret_es, axis=0)

    regret_gpbo_std = np.std(regret_gpbo, axis=0)
    regret_mpbo_std = np.std(regret_mpbo, axis=0)
    regret_es_std = np.std(regret_es, axis=0)

    font = {"weight": "normal", "size": 12}
    matplotlib.rc("font", **{"family": "serif", "serif": ["Times New Roman"]})
    matplotlib.rc("text", usetex=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    epochs = 30
    queries = 96

    ax.plot(
        regret_mpbo,
        "tab:blue",
        label="MP-BO",
    )

    ax.fill_between(
        np.arange(queries),
        regret_mpbo - regret_mpbo_std / np.sqrt(epochs),
        regret_mpbo + regret_mpbo_std / np.sqrt(epochs),
        color="tab:blue",
        alpha=0.2,
    )

    ax.plot(regret_gpbo, "r", linestyle="--", label="Vanilla BO")
    ax.fill_between(
        np.arange(queries),
        regret_gpbo - regret_gpbo_std / np.sqrt(epochs),
        regret_gpbo + regret_gpbo_std / np.sqrt(epochs),
        color="r",
        alpha=0.2,
    )

    ax.plot(regret_es, "g", linestyle="-.", label="Extensive Search")
    ax.fill_between(
        np.arange(queries),
        regret_es - regret_es_std / np.sqrt(epochs),
        regret_es + regret_es_std / np.sqrt(epochs),
        color="g",
        alpha=0.2,
    )

    ax.set_title("\\textbf{Neurostimulation Dataset}")
    ax.set_ylabel("\\textbf{Regret}", labelpad=-5)
    ax.set_yticks([0, 0.5, 1], labels=["0", "", "1"])
    ax.set_xticks([0, 48, 96], labels=["0", "", "96"])
    ax.set_xlabel("\\textbf{Iteration}", labelpad=-5)
    ax.set_ylim(-0.1, 1.1)

    ax.vlines(32, -0.1, 1.1, linestyle="dotted", color="k", label="q* = 32")

    legend, _ = ax.get_legend_handles_labels()
    plt.legend(
        frameon=False,
        fontsize=14,
        ncols=2,
        bbox_to_anchor=(0.9, 1.2),
    )
    # plt.show()
    plt.savefig("neurostim.pdf", bbox_inches="tight")


create_regret_comparison_figure(files)


def create_qvary_figure():
    ackley = np.load("outpus/Ackley_varyq.npz")
    mich2d = np.load("outpus/Michalewicz2D_varyq.npz")
    mich4d = np.load("outpus/Michalewicz4D_varyq.npz")
    hart = np.load("outpus/Hartmann6D_varyq.npz")

    q_star = ackley["q_stars"]
    regret_ackley_bo = ackley["regret_gpbo"]
    regret_ackley_mpbo = ackley["regret_mpbo"]
    regret_ackley_fifo = ackley["regret_fifo"]

    gpbo = np.zeros(len(q_star))
    mpbo = np.zeros(len(q_star))
    fifo = np.zeros(len(q_star))

    gpbo_std = np.zeros(len(q_star))
    mpbo_std = np.zeros(len(q_star))
    fifo_std = np.zeros(len(q_star))

    for i, q in enumerate(q_star):
        gpbo[i] = regret_ackley_bo.mean(axis=1)[i, q - 1]
        mpbo[i] = regret_ackley_mpbo.mean(axis=1)[i, -1]
        fifo[i] = regret_ackley_fifo.mean(axis=1)[i, -1]

        gpbo_std[i] = regret_ackley_bo.std(axis=1)[i, q - 1]
        mpbo_std[i] = regret_ackley_mpbo.std(axis=1)[i, -1]
        fifo_std[i] = regret_ackley_fifo.std(axis=1)[i, -1]

    font = {"weight": "normal", "size": 12}
    matplotlib.rc("font", **{"family": "serif", "serif": ["Times New Roman"]})
    matplotlib.rc("text", usetex=True)

    fig, ax = plt.subplots(2, 2, figsize=(7, 6))

    ax[0, 0].plot(q_star, gpbo, "r", linestyle="--", label="Vanilla BO")
    ax[0, 0].plot(q_star, mpbo, "tab:blue", label="MP-BO")
    # ax[0, 0].plot(q_star, fifo, "g", linestyle="-.", label="FiFo")

    ax[0, 0].fill_between(
        q_star,
        gpbo - gpbo_std / np.sqrt(30),
        gpbo + gpbo_std / np.sqrt(30),
        color="r",
        alpha=0.2,
    )
    ax[0, 0].fill_between(
        q_star,
        mpbo - mpbo_std / np.sqrt(30),
        mpbo + mpbo_std / np.sqrt(30),
        color="tab:blue",
        alpha=0.2,
    )
    # ax[0, 0].fill_between(
    #     q_star,
    #     fifo - fifo_std / np.sqrt(30),
    #     fifo + fifo_std / np.sqrt(30),
    #     color="g",
    #     alpha=0.2,
    # )

    regret_mich2d_bo = mich2d["regret_gpbo"]
    regret_mich2d_mpbo = mich2d["regret_mpbo"]
    regret_mich2d_fifo = mich2d["regret_fifo"]

    gpbo = np.zeros(len(q_star))
    mpbo = np.zeros(len(q_star))
    fifo = np.zeros(len(q_star))

    gpbo_std = np.zeros(len(q_star))
    mpbo_std = np.zeros(len(q_star))
    fifo_std = np.zeros(len(q_star))

    for i, q in enumerate(q_star):
        gpbo[i] = regret_mich2d_bo.mean(axis=0)[q - 1]
        mpbo[i] = regret_mich2d_mpbo.mean(axis=1)[i, -1]
        fifo[i] = regret_mich2d_fifo.mean(axis=1)[i, -1]

        gpbo_std[i] = regret_mich2d_bo.std(axis=0)[q - 1]
        mpbo_std[i] = regret_mich2d_mpbo.std(axis=1)[i, -1]
        fifo_std[i] = regret_mich2d_fifo.std(axis=1)[i, -1]

    ax[0, 1].plot(q_star, gpbo, "r", linestyle="--", label="Vanilla BO")
    ax[0, 1].plot(q_star, mpbo, "tab:blue", label="MP-BO")
    # ax[0, 1].plot(q_star, fifo, "g", linestyle="-.", label="FiFo")

    ax[0, 1].fill_between(
        q_star,
        gpbo - gpbo_std / np.sqrt(30),
        gpbo + gpbo_std / np.sqrt(30),
        color="r",
        alpha=0.2,
    )
    ax[0, 1].fill_between(
        q_star,
        mpbo - mpbo_std / np.sqrt(30),
        mpbo + mpbo_std / np.sqrt(30),
        color="tab:blue",
        alpha=0.2,
    )
    # ax[0, 1].fill_between(
    #     q_star,
    #     fifo - fifo_std / np.sqrt(30),
    #     fifo + fifo_std / np.sqrt(30),
    #     color="g",
    #     alpha=0.2,
    # )

    regret_mich4d_bo = mich4d["regret_gpbo"]
    regret_mich4d_mpbo = mich4d["regret_mpbo"]
    regret_mich4d_fifo = mich4d["regret_fifo"]

    gpbo = np.zeros(len(q_star))
    mpbo = np.zeros(len(q_star))
    fifo = np.zeros(len(q_star))

    gpbo_std = np.zeros(len(q_star))
    mpbo_std = np.zeros(len(q_star))
    fifo_std = np.zeros(len(q_star))

    for i, q in enumerate(q_star):
        gpbo[i] = regret_mich4d_bo.mean(axis=0)[q - 1]
        mpbo[i] = regret_mich4d_mpbo.mean(axis=1)[i, -1]
        fifo[i] = regret_mich4d_fifo.mean(axis=1)[i, -1]

        gpbo_std[i] = regret_mich4d_bo.std(axis=0)[q - 1]
        mpbo_std[i] = regret_mich4d_mpbo.std(axis=1)[i, -1]
        fifo_std[i] = regret_mich4d_fifo.std(axis=1)[i, -1]

    ax[1, 0].plot(q_star, gpbo, "r", linestyle="--", label="Vanilla BO")
    ax[1, 0].plot(q_star, mpbo, "tab:blue", label="MP-BO")
    # ax[1, 0].plot(q_star, fifo, "g", linestyle="-.", label="FiFo")

    ax[1, 0].fill_between(
        q_star,
        gpbo - gpbo_std / np.sqrt(30),
        gpbo + gpbo_std / np.sqrt(30),
        color="r",
        alpha=0.2,
    )

    ax[1, 0].fill_between(
        q_star,
        mpbo - mpbo_std / np.sqrt(30),
        mpbo + mpbo_std / np.sqrt(30),
        color="tab:blue",
        alpha=0.2,
    )

    # ax[1, 0].fill_between(
    #     q_star,
    #     fifo - fifo_std / np.sqrt(30),
    #     fifo + fifo_std / np.sqrt(30),
    #     color="g",
    #     alpha=0.2,
    # )

    regret_hart_bo = hart["regret_gpbo"]
    regret_hart_mpbo = hart["regret_mpbo"]
    regret_hart_fifo = hart["regret_fifo"]

    gpbo = np.zeros(len(q_star))
    mpbo = np.zeros(len(q_star))
    fifo = np.zeros(len(q_star))

    gpbo_std = np.zeros(len(q_star))
    mpbo_std = np.zeros(len(q_star))
    fifo_std = np.zeros(len(q_star))

    for i, q in enumerate(q_star):
        gpbo[i] = regret_hart_bo.mean(axis=0)[q - 1]
        mpbo[i] = regret_hart_mpbo.mean(axis=1)[i, -1]
        fifo[i] = regret_hart_fifo.mean(axis=1)[i, -1]

        gpbo_std[i] = regret_hart_bo.std(axis=0)[q - 1]
        mpbo_std[i] = regret_hart_mpbo.std(axis=1)[i, -1]
        fifo_std[i] = regret_hart_fifo.std(axis=1)[i, -1]

    ax[1, 1].plot(q_star, gpbo, "r", linestyle="--", label="Vanilla BO")
    ax[1, 1].plot(q_star, mpbo, "tab:blue", label="MP-BO")
    # ax[1, 1].plot(q_star, fifo, "g", linestyle="-.", label="FiFo")

    ax[1, 1].fill_between(
        q_star,
        gpbo - gpbo_std / np.sqrt(30),
        gpbo + gpbo_std / np.sqrt(30),
        color="r",
        alpha=0.2,
    )

    ax[1, 1].fill_between(
        q_star,
        mpbo - mpbo_std / np.sqrt(30),
        mpbo + mpbo_std / np.sqrt(30),
        color="tab:blue",
        alpha=0.2,
    )

    # ax[1, 1].fill_between(
    #     q_star,
    #     fifo - fifo_std / np.sqrt(30),
    #     fifo + fifo_std / np.sqrt(30),
    #     color="g",
    #     alpha=0.2,
    # )

    ax[0, 0].set_title("\\textbf{Ackley}")
    ax[0, 1].set_title("\\textbf{Michalewicz 2D}")
    ax[1, 0].set_title("\\textbf{Michalewicz 4D}")
    ax[1, 1].set_title("\\textbf{Hartmann}")

    ax[0, 0].set_ylabel("\\textbf{Regret}", labelpad=-5)
    ax[1, 0].set_ylabel("\\textbf{Regret}", labelpad=-5)

    ax[1, 0].set_xlabel("\\textbf{q*}", labelpad=-5)
    ax[1, 1].set_xlabel("\\textbf{q*}", labelpad=-5)

    ax[0, 0].set_ylim(-0.1, 1.1)
    ax[0, 1].set_ylim(-0.1, 1.1)

    ax[1, 0].set_ylim(-0.1, 1.1)
    ax[1, 1].set_ylim(-0.1, 1.1)

    ax[0, 0].set_xticks([0, 30, 60], labels=["0", "", "60"])
    ax[0, 1].set_xticks([0, 30, 60], labels=["0", "", "60"])
    ax[1, 0].set_xticks([0, 30, 60], labels=["0", "", "60"])
    ax[1, 1].set_xticks([0, 30, 60], labels=["0", "", "60"])

    ax[0, 0].set_yticks([0, 0.5, 1], labels=["0", "", "1"])
    ax[1, 0].set_yticks([0, 0.5, 1], labels=["0", "", "1"])

    ax[0, 1].set_yticks([0, 0.5, 1], labels=["0", "", "1"])
    ax[1, 1].set_yticks([0, 0.5, 1], labels=["0", "", "1"])

    legend, _ = ax[1, 1].get_legend_handles_labels()
    plt.legend(
        frameon=False,
        fontsize=14,
        ncols=3,
        bbox_to_anchor=(0.8, 2.55),
    )

    # plt.show()
    plt.savefig("q_vary.pdf", bbox_inches="tight")
