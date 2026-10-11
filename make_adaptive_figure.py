import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = json.load(open("adaptive_results.json"))
titles = {"bg1_testjitter0.6": "background 1.0, test cells mismatched (±60% write)",
          "bg1_testjitter0": "background 1.0, no mismatch (control)",
          "bg4_testjitter0.6": "background 4.0, mismatched"}
cols = {"uniform": "#888", "random": "#bbb", "fixed_best": "#2a6", "schedule": "#36c", "adaptive": "#c44"}
labels = {"uniform": "uniform (neutral)", "random": "random bank ping", "fixed_best": "best single ping, repeated",
          "schedule": "trained fixed sequence", "adaptive": "adaptive (posterior-driven)"}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
k = np.arange(1, 7)
for a, (cn, t) in zip(ax, titles.items()):
    for p, v in R["conditions"][cn]["policies"].items():
        a.fill_between(k, v["acc_min"], v["acc_max"], color=cols[p], alpha=0.15, lw=0)
        a.plot(k, v["acc_mean"], "o-", color=cols[p], label=labels[p], lw=2 if p in ("adaptive", "fixed_best") else 1.3)
    a.axhline(0.25, ls=":", c="k", lw=1)
    a.set_title(t, fontsize=10)
    a.set_xlabel("pings delivered (same energy each)")
    a.set_ylim(0.2, 1.0)
    a.grid(alpha=0.3)
ax[0].set_ylabel("held-out accuracy, 4 items (chance 0.25)")
ax[0].legend(fontsize=8, loc="lower right")
fig.suptitle("AdaptivePing on the S2 cable: does choosing the ping help, and does adapting help beyond one good ping?", fontsize=11)
fig.tight_layout()
fig.savefig("adaptive_ping.png", dpi=130)
print("wrote adaptive_ping.png")
