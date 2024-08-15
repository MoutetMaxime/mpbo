import numpy as np
import matplotlib.pyplot as plt

import matplotlib


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
mpbo_mich4d = np.load("Mich4D_mpbo_4_1O_800.npz")

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

time_mpbo_hart = time_mpbo_hart[time_mpbo_hart <= 14.5]
time_gpbo_hart = time_gpbo_hart[time_gpbo_hart <= 14.5]

regret_gpbo_hart = gpbo_hart["regret"].mean(axis=0)[: len(time_gpbo_hart)]
regret_mpbo_hart = mpbo_hart["regret"].mean(axis=0)[: len(time_mpbo_hart)]

print(len(time_gpbo_hart), len(time_mpbo_hart))


font = {"weight": "normal", "size": 12}
matplotlib.rc("font", **{"family": "serif", "serif": ["Times New Roman"]})
matplotlib.rc("text", usetex=True)
fig, axs = plt.subplots(2, 2, sharey=True, figsize=(7, 6))

axs[0, 0].plot(
    time_mpbo_ackley,
    regret_mpbo_ackley,
    "tab:blue",
    label="MP-BO",
)
axs[0, 0].fill_between(
    time_mpbo_ackley,
    regret_mpbo_ackley
    - mpbo_ackley["regret"].std(axis=0)[: len(time_mpbo_ackley)] / np.sqrt(10),
    regret_mpbo_ackley
    + mpbo_ackley["regret"].std(axis=0)[: len(time_mpbo_ackley)] / np.sqrt(10),
    color="tab:blue",
    alpha=0.2,
)
axs[0, 0].plot(
    time_gpbo_ackley, regret_gpbo_ackley, "r", linestyle="--", label="Vanilla BO"
)
axs[0, 0].fill_between(
    time_gpbo_ackley,
    regret_gpbo_ackley
    - gpbo_ackley["regret"].std(axis=0)[: len(time_gpbo_ackley)] / np.sqrt(10),
    regret_gpbo_ackley
    + gpbo_ackley["regret"].std(axis=0)[: len(time_gpbo_ackley)] / np.sqrt(10),
    color="r",
    alpha=0.2,
)

axs[0, 1].plot(
    time_mpbo_mich2d,
    regret_mpbo_mich2d,
    "tab:blue",
    label="MP-BO",
)
axs[0, 1].fill_between(
    time_mpbo_mich2d,
    regret_mpbo_mich2d
    - mpbo_mich2d["regret"].std(axis=0)[: len(time_mpbo_mich2d)] / np.sqrt(10),
    regret_mpbo_mich2d
    + mpbo_mich2d["regret"].std(axis=0)[: len(time_mpbo_mich2d)] / np.sqrt(10),
    color="tab:blue",
    alpha=0.2,
)

axs[0, 1].plot(
    time_gpbo_mich2d, regret_gpbo_mich2d, "r", linestyle="--", label="Vanilla BO"
)
axs[0, 1].fill_between(
    time_gpbo_mich2d,
    regret_gpbo_mich2d
    - gpbo_mich2d["regret"].std(axis=0)[: len(time_gpbo_mich2d)] / np.sqrt(10),
    regret_gpbo_mich2d
    + gpbo_mich2d["regret"].std(axis=0)[: len(time_gpbo_mich2d)] / np.sqrt(10),
    color="r",
    alpha=0.2,
)

axs[1, 0].plot(
    time_mpbo_mich4d,
    regret_mpbo_mich4d,
    "tab:blue",
    label="MP-BO",
)

axs[1, 0].fill_between(
    time_mpbo_mich4d,
    regret_mpbo_mich4d
    - mpbo_mich4d["regret"].std(axis=0)[: len(time_mpbo_mich4d)] / np.sqrt(10),
    regret_mpbo_mich4d
    + mpbo_mich4d["regret"].std(axis=0)[: len(time_mpbo_mich4d)] / np.sqrt(10),
    color="tab:blue",
    alpha=0.2,
)

axs[1, 0].plot(
    time_gpbo_mich4d, regret_gpbo_mich4d, "r", linestyle="--", label="Vanilla BO"
)

axs[1, 0].fill_between(
    time_gpbo_mich4d,
    regret_gpbo_mich4d
    - gpbo_mich4d["regret"].std(axis=0)[: len(time_gpbo_mich4d)] / np.sqrt(10),
    regret_gpbo_mich4d
    + gpbo_mich4d["regret"].std(axis=0)[: len(time_gpbo_mich4d)] / np.sqrt(10),
    color="r",
    alpha=0.2,
)

axs[1, 1].plot(
    time_mpbo_hart,
    regret_mpbo_hart,
    "tab:blue",
    label="MP-BO",
)

axs[1, 1].fill_between(
    time_mpbo_hart,
    regret_mpbo_hart
    - mpbo_hart["regret"].std(axis=0)[: len(time_mpbo_hart)] / np.sqrt(10),
    regret_mpbo_hart
    + mpbo_hart["regret"].std(axis=0)[: len(time_mpbo_hart)] / np.sqrt(10),
    color="tab:blue",
    alpha=0.2,
)

axs[1, 1].plot(
    time_gpbo_hart, regret_gpbo_hart, "r", linestyle="--", label="Vanilla BO"
)

axs[1, 1].fill_between(
    time_gpbo_hart,
    regret_gpbo_hart
    - gpbo_hart["regret"].std(axis=0)[: len(time_gpbo_hart)] / np.sqrt(10),
    regret_gpbo_hart
    + gpbo_hart["regret"].std(axis=0)[: len(time_gpbo_hart)] / np.sqrt(10),
    color="r",
    alpha=0.2,
)

axs[0, 0].set_title("\\textbf{Ackley}")
axs[0, 1].set_title("\\textbf{Michalewicz 2D}")
axs[1, 0].set_title("\\textbf{Michalewicz 4D}")
axs[1, 1].set_title("\\textbf{Hartmann}")

axs[0, 0].set_ylabel("\\textbf{Regret}", labelpad=-5)
axs[1, 0].set_ylabel("\\textbf{Regret}", labelpad=-5)
axs[0, 0].set_yticks([0, 0.5, 1], labels=["0", "", "1"])
axs[1, 0].set_yticks([0, 0.5, 1], labels=["0", "", "1"])

axs[1, 0].set_xlabel("\\textbf{Time (s)}")
axs[1, 1].set_xlabel("\\textbf{Time (s)}")

legend, _ = axs[0, 0].get_legend_handles_labels()
plt.legend(frameon=False, fontsize=14, ncols=2, loc="upper center")

plt.tight_layout()

plt.savefig("bo_vs_mpbo.pdf", bbox_inches="tight")
plt.show()
