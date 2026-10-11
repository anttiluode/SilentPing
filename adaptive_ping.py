"""
AdaptivePing — does choosing the ping beat the neutral one, and is the gain from
finding one good ping or from adapting to earlier answers?

Substrate: S2 CABLE from silent_ping.py, unchanged (24 compartments, soma read,
silent gain tag at one of 4 sites). Every ping is a spatial input profile p >= 0
applied at one time step, with the SAME energy as the README's uniform ping:
||p||_2 = 0.5 * sqrt(24). Pings are delivered one after another to the same trial
(the state carries over, and a strong ping can WRITE the tag; that is part of the
test), each followed by a 160-step readout window and a 300-step quiet gap.

Protocol (written before the first run, frozen):
  * Training trials (n=400, write-strength jitter 0.0) build per-ping templates:
    mean binned soma response per item and a pooled shrinkage covariance.
  * Test trials (n=400, fresh seed, jitter 0.6 = model mismatch: the templates
    assume a fixed write strength, the test cells do not).
  * Every policy uses the SAME decoder: Gaussian log-likelihoods from the
    templates, summed over pings, MAP item. Only the choice of ping differs.
  * Policies, same ping bank (uniform, ramps, Gaussian bumps of two widths):
      uniform    the README's neutral ping, repeated
      random     a random bank ping each time
      fixed_best the single bank ping with the best one-ping accuracy on
                 TRAINING trials, repeated
      schedule   a fixed sequence chosen greedily on TRAINING trials, run
                 sequentially (the strongest non-adaptive baseline)
      adaptive   after each ping, choose the bank ping maximizing the
                 posterior-weighted discrimination sum_{i<j} pi_i pi_j D_ij(p),
                 D_ij(p) = (mu_pi - mu_pj)^T S_p^-1 (mu_pi - mu_pj), from templates
  * Metric: test accuracy after k = 1..6 pings; pings needed for 90%.
  * Conditions: background 1.0 (README headline) and 4.0 (drive high, ping
    advantage shrinking). Three test seeds each.

Added after the first (--quick) run, before the full run: a matched control
(background 1.0, test jitter 0.0, no mismatch) and a tag-integrity readout (how
much of the silent tag is still at the true site after each ping), because the
quick run showed accuracy FALLING with more pings in some policies.

python adaptive_ping.py  [--quick]
"""
import argparse
import json
import time

import numpy as np

from silent_ping import Cable, K

E_TARGET = 0.5 * np.sqrt(24)
WIN, GAP, NB, KMAX = 160, 300, 32, 6


def make_bank(M=24):
    bank, names = [np.ones(M)], ["uniform"]
    x = np.arange(M)
    bank += [x + 1.0, M - x]
    names += ["ramp_distal", "ramp_proximal"]
    for w in (1.5, 4.0):
        for c in range(M):
            bank.append(np.exp(-0.5 * ((x - c) / w) ** 2))
            names.append(f"bump_c{c}_w{w:g}")
    bank = np.array([b / np.linalg.norm(b) * E_TARGET for b in bank])
    return bank, names


def ping(c, st, p, bg, rng):
    """Deliver profile p (per trial: (n, M) or (M,)), read WIN steps, then GAP quiet
    steps. Returns binned soma features and the carried-over state."""
    v, s = st["v"], st["s"]
    n = v.shape[0]
    trace = []
    for t in range(WIN + GAP):
        inp = c._bg(bg, n, rng)
        if t == 0:
            inp = inp + p
        v = v @ c.A.T + (1 + c.g * s) * inp + c.noise * rng.standard_normal(v.shape)
        s = s - s / c.tau_s + c._write(inp, v, s)
        if t < WIN:
            trace.append(v[:, 0] + c.meas * rng.standard_normal(n))
    tr = np.array(trace).T
    feat = tr.reshape(n, NB, WIN // NB).mean(2)
    return feat, {"v": v, "s": s}


def history(c, n, bg, jitter, rng):
    y = np.repeat(np.arange(K), n // K)
    rng.shuffle(y)
    st, _ = c.run_history(y, bg, 1500, rng, jitter=jitter)
    return y, st


class Templates:
    def __init__(self, mu, icov, logdet):
        self.mu, self.icov, self.logdet = mu, icov, logdet   # (P,K,NB), (P,NB,NB), (P,)

    def loglik(self, j, X):
        d = X[:, None, :] - self.mu[j][None]                   # (n,K,NB)
        return -0.5 * np.einsum("nkb,bc,nkc->nk", d, self.icov[j], d)

    def disc(self):
        """D[j, a, b] = whitened squared distance between item templates under ping j."""
        mu = self.mu
        diff = mu[:, :, None, :] - mu[:, None, :, :]
        return np.einsum("pabx,pxy,paby->pab", diff, self.icov, diff)


def build_templates(c, bank, y, st, bg, rng, shrink=0.2):
    P = len(bank)
    mu = np.zeros((P, K, NB)); icov = np.zeros((P, NB, NB)); logdet = np.zeros(P)
    firstfeat = []
    for j, p in enumerate(bank):
        X, _ = ping(c, {k: v.copy() for k, v in st.items()}, p, bg, rng)
        firstfeat.append(X)
        for k in range(K):
            mu[j, k] = X[y == k].mean(0)
        R = X - mu[j, y]
        S = np.cov(R.T)
        S = (1 - shrink) * S + shrink * np.eye(NB) * np.trace(S) / NB
        icov[j] = np.linalg.inv(S)
        logdet[j] = np.linalg.slogdet(S)[1]
    return Templates(mu, icov, logdet), np.array(firstfeat)


def run_policy(c, bank, T, y, st, bg, rng, choose, kmax=KMAX):
    """choose(step, logpost) -> array of bank indices per trial. Returns accuracy
    after each ping and the ping indices used."""
    st = {k: v.copy() for k, v in st.items()}
    n = len(y)
    logpost = np.zeros((n, K))
    acc, used, tag = [], [], []
    site = np.zeros((n, c.M), bool)
    for i, k in enumerate(y):
        site[i, c.loc[k] - 2:c.loc[k] + 2] = True
    s0_site = (st["s"] * site).sum(1)
    for step in range(kmax):
        js = choose(step, logpost)
        p = bank[js]
        X, st = ping(c, st, p, bg, rng)
        for j in np.unique(js):
            m = js == j
            logpost[m] += T.loglik(j, X[m])
        acc.append(float(np.mean(logpost.argmax(1) == y)))
        used.append(js.copy())
        tot = st["s"].sum(1) + 1e-12
        tag.append({"site_fraction": float(np.mean((st["s"] * site).sum(1) / tot)),
                    "site_mass_vs_start": float(np.mean((st["s"] * site).sum(1)) / (np.mean(s0_site) + 1e-12))})
    run_policy.last_tag = tag
    return acc, used


def softmax(L):
    L = L - L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="adaptive_results.json")
    a = ap.parse_args()
    ntr, nte = (200, 200) if a.quick else (400, 400)
    seeds = [101] if a.quick else [101, 202, 303]
    bank, names = make_bank()
    c = Cable(seed=2)
    R = {"bank": names, "energy_l2": float(E_TARGET), "conditions": {}}
    t0 = time.time()
    for bg, tjit in ((1.0, 0.6), (1.0, 0.0), (4.0, 0.6)):
        rng = np.random.default_rng(int(bg * 100) + 7)
        ytr, sttr = history(c, ntr, bg, 0.0, rng)
        T, F1 = build_templates(c, bank, ytr, sttr, bg, rng)
        D = T.disc()
        # one-ping training accuracy per bank ping (template decoder, in-sample)
        tr1 = np.array([np.mean(T.loglik(j, F1[j]).argmax(1) == ytr) for j in range(len(bank))])
        j_best = int(np.argmax(tr1))
        # greedy fixed schedule on TRAINING trials, sequential (state carries over)
        sched = []
        for step in range(KMAX):
            best = (-1, None)
            for j in range(len(bank)):
                seq = sched + [j]
                acc, _ = run_policy(c, bank, T, ytr, sttr, bg, np.random.default_rng(5 + step),
                                    lambda s, L, seq=seq: np.full(len(ytr), seq[s]), kmax=len(seq))
                if acc[-1] > best[0]:
                    best = (acc[step], j)
            sched.append(best[1])
            print(f" bg {bg} schedule step {step + 1}: {names[best[1]]} train acc {best[0]:.3f}", flush=True)

        def pol_uniform(step, L): return np.zeros(L.shape[0], int)
        def pol_fixed(step, L): return np.full(L.shape[0], j_best)
        def pol_sched(step, L): return np.full(L.shape[0], sched[step])

        def pol_adapt(step, L):
            pi = softmax(L)
            score = np.einsum("na,nb,pab->np", pi, pi, D)       # (n, P)
            return score.argmax(1)

        cond = {"fixed_best": names[j_best], "fixed_best_train_acc_1ping": float(tr1[j_best]),
                "uniform_train_acc_1ping": float(tr1[0]), "schedule": [names[j] for j in sched],
                "policies": {}}
        for pname in ("uniform", "random", "fixed_best", "schedule", "adaptive"):
            accs, usage, tags = [], {}, []
            for sd in seeds:
                rng = np.random.default_rng(sd + int(bg * 1000))
                yte, stte = history(c, nte, bg, tjit, rng)
                prng = np.random.default_rng(sd + 7)
                pol = {"uniform": pol_uniform, "fixed_best": pol_fixed, "schedule": pol_sched,
                       "adaptive": pol_adapt,
                       "random": lambda s, L: prng.integers(0, len(bank), L.shape[0])}[pname]
                acc, used = run_policy(c, bank, T, yte, stte, bg, rng, pol)
                accs.append(acc)
                tags.append(run_policy.last_tag)
                for u in used:
                    for j in u:
                        usage[names[j]] = usage.get(names[j], 0) + 1
            accs = np.array(accs)
            cond["policies"][pname] = {"acc_mean": accs.mean(0).round(4).tolist(),
                                       "acc_min": accs.min(0).round(4).tolist(),
                                       "acc_max": accs.max(0).round(4).tolist(),
                                       "top_pings": sorted(usage.items(), key=lambda kv: -kv[1])[:5],
                                       "tag_site_mass_vs_start": [round(float(np.mean([t[k]["site_mass_vs_start"] for t in tags])), 3) for k in range(KMAX)],
                                       "tag_site_fraction": [round(float(np.mean([t[k]["site_fraction"] for t in tags])), 3) for k in range(KMAX)]}
            print(f" bg {bg} jit {tjit} {pname:<10} " + " ".join(f"{x:.3f}" for x in accs.mean(0))
                  + "  tag " + " ".join(f"{x:.2f}" for x in cond["policies"][pname]["tag_site_mass_vs_start"]), flush=True)
        R["conditions"][f"bg{bg:g}_testjitter{tjit:g}"] = cond
    R["runtime_s"] = round(time.time() - t0, 1)
    json.dump(R, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
