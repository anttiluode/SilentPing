"""
SilentPing — can a neutral ping read an item out of state that is invisible at rest?

In silico version of Wolff et al. (2017) "dynamic hidden states" / Stokes (2015)
activity-silent working memory, built on two substrates:

  S1  STF   : Tsodyks-Markram short-term facilitation synapses (the mainstream mechanism,
              Mongillo, Barak & Tsodyks 2008). Known-answer calibration.
  S2  CABLE : one passive cable neuron with a slow, synapse-specific resident gain tag
              (NotSoSimpleNeuron lineage: resident state changes the operator the next
              event meets). Asks whether a SINGLE neuron's ping echo says WHERE the
              silent state sits.

Conditions, all with identical stimulus/delay history:
  delay   decode from output during the late delay            -> should be chance (silent)
  rest    decode from the readout window, no ping             -> should be ~chance
  ping    same window, neutral ping (identical on every trial) -> should be high
  reset   resident state reset to baseline just before ping   -> chance (null: ping alone)
  swap    resident state transplanted from a trial with a different label;
          the ping-trained decoder should report the DONOR's label (causal receipt)

Sweeps: ping delay (decay of the hidden state) and background drive (silence is a
gain x drive statement, not an absence).

numpy + scikit-learn only.  python silent_ping.py  [--quick]
"""
import argparse, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

K = 4  # items
S1_BG = 0.5   # Hz background afferent rate for the STF headline run
S2_BG = 1.0   # background event density for the cable headline run


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


def cv_acc(X, y, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf(), X, y, cv=cv).mean())


def perm_null(X, y, n=50, seed=0):
    rng = np.random.default_rng(seed)
    return [cv_acc(X, rng.permutation(y), seed) for _ in range(n)]


# --------------------------------------------------------------------------------------
# S1: Tsodyks-Markram facilitating synapses -> output population, Poisson spikes, dt=1ms
# --------------------------------------------------------------------------------------
class STF:
    def __init__(self, n_in=80, n_out=400, U=0.2, tauF=1500., tauD=200., tau_syn=5.,
                 r0=2.0, gain=150.0, seed=0):
        rng = np.random.default_rng(seed)
        self.n_in, self.n_out = n_in, n_out
        self.U, self.tauF, self.tauD, self.tau_syn = U, tauF, tauD, tau_syn
        self.r0, self.gain = r0, gain
        self.items = np.zeros((K, n_in), bool)          # each item = its own afferent set
        perm = rng.permutation(n_in)
        for k in range(K):
            self.items[k, perm[k * (n_in // K):(k + 1) * (n_in // K)]] = True
        # sparse random weights plus item selectivity: output neuron i prefers item i % K
        W = rng.random((n_out, n_in)) * (rng.random((n_out, n_in)) < 0.3)
        pref = self.items[np.arange(n_out) % K]
        self.W = W * np.where(pref, 3.0, 1.0) / 3.0

    def init(self, n):
        return dict(u=np.full((n, self.n_in), self.U), x=np.ones((n, self.n_in)),
                    I=np.zeros((n, self.n_out)))

    def step(self, st, spikes_in, rng):
        u, x = st['u'], st['x']
        u += (self.U - u) / self.tauF
        x += (1 - x) / self.tauD
        u[:] = np.where(spikes_in, u + self.U * (1 - u), u)   # facilitate, then release
        rel = np.where(spikes_in, u * x, 0.0)
        x -= rel
        st['I'] += -st['I'] / self.tau_syn + rel @ self.W.T
        rate = np.maximum(self.r0 + self.gain * st['I'], 0.0)   # Hz
        return rng.random(rate.shape) < rate * 1e-3             # output spikes

    def run_history(self, labels, r_bg, delay_ms, rng, stim_ms=250, stim_rate=40.):
        n = len(labels)
        st = self.init(n)
        mask = self.items[labels]
        delay_counts = np.zeros((n, self.n_out))
        for t in range(stim_ms + delay_ms):
            rate = np.where(mask, stim_rate, r_bg) if t < stim_ms else np.full(mask.shape, r_bg)
            sp = self.step(st, rng.random(mask.shape) < rate * 1e-3, rng)
            if t >= stim_ms + delay_ms - 500:                   # late delay window
                delay_counts += sp
        return st, delay_counts

    def readout(self, st, r_bg, ping, rng, win=150, bins=3):
        st = {k: v.copy() for k, v in st.items()}
        n = st['u'].shape[0]
        out = np.zeros((n, self.n_out, bins))
        for t in range(win):
            spk = rng.random((n, self.n_in)) < r_bg * 1e-3
            if ping and t == 0:
                spk[:] = True                                   # one synchronous volley, identical every trial
            sp = self.step(st, spk, rng)
            out[:, :, t * bins // win] += sp
        return out.reshape(n, -1)

    def reset(self, st):
        st = {k: v.copy() for k, v in st.items()}
        st['u'][:] = self.U; st['x'][:] = 1.0; st['I'][:] = 0.0
        return st


# --------------------------------------------------------------------------------------
# S2: passive cable neuron, soma at compartment 0, slow local resident gain s
#     v' = A v + (1 + g s) * input + noise ;  s grows where v exceeds threshold, decays slowly
#     readout = soma voltage waveform (+ measurement noise)
# --------------------------------------------------------------------------------------
class Cable:
    def __init__(self, M=24, D=0.45, leak=0.006, g=3.0, theta=1.0, k_s=0.02, tau_s=3000.,
                 noise=0.01, meas=0.02, seed=0):
        self.M, self.g, self.theta, self.k_s, self.tau_s = M, g, theta, k_s, tau_s
        self.noise, self.meas = noise, meas
        L = -2 * np.eye(M) + np.eye(M, k=1) + np.eye(M, k=-1)
        L[0, 0] = L[-1, -1] = -1                                  # sealed ends
        self.A = np.eye(M) + D * L - leak * np.eye(M)
        self.loc = np.array([3, 7, 12, 18])                      # item sites, near -> far
        self.drive = 0.08 * np.ones(M)

    def run_history(self, labels, bg, delay, rng, stim=120, jitter=0.0):
        n = len(labels)
        v = np.zeros((n, self.M)); s = np.zeros((n, self.M))
        site = np.zeros((n, self.M))
        for i, k in enumerate(labels):
            site[i, self.loc[k] - 2:self.loc[k] + 2] = 1.0
        amp = 0.30 * (1 + jitter * rng.uniform(-1, 1, (n, 1)))   # write strength jitter
        soma_delay = []
        for t in range(stim + delay):
            inp = site * amp if t < stim else 0.0
            inp = inp + self._bg(bg, n, rng)
            v = v @ self.A.T + (1 + self.g * s) * inp + self.noise * rng.standard_normal(v.shape)
            s += -s / self.tau_s + self._write(inp, v, s)
            if t >= stim + delay - 200:
                soma_delay.append(v[:, 0] + self.meas * rng.standard_normal(n))
        return dict(v=v, s=s), np.array(soma_delay).T

    def readout(self, st, bg, ping, rng, win=160):
        v, s = st['v'].copy(), st['s'].copy()
        n = v.shape[0]; trace = []
        for t in range(win):
            inp = self._bg(bg, n, rng)
            if ping and t == 0:
                inp = inp + 0.5                                   # uniform, identical every trial
            v = v @ self.A.T + (1 + self.g * s) * inp + self.noise * rng.standard_normal(v.shape)
            s += -s / self.tau_s + self._write(inp, v, s)
            trace.append(v[:, 0] + self.meas * rng.standard_normal(n))
        return np.array(trace).T

    def _write(self, inp, v, s):
        # synapse-specific tag (NMDA-like coincidence): resident state grows only where
        # input ARRIVES while the local membrane is depolarised past theta
        return self.k_s * np.maximum(inp, 0) / 0.3 * np.maximum(v - self.theta, 0) * (1 - s)

    def _bg(self, bg, n, rng):
        # balanced background: excitatory and inhibitory events, zero mean, rate ~ bg
        ev = rng.random((n, self.M)) < 0.02 * bg
        return np.where(ev, 0.1 * rng.choice([-1.0, 1.0], (n, self.M)), 0.0)

    def reset(self, st):
        return dict(v=st['v'].copy(), s=np.zeros_like(st['s']))


# --------------------------------------------------------------------------------------
def bin_power(X, nb=8):
    # second-order attacker: a silent state that only modulates VARIANCE would be invisible
    # to a linear decoder on the raw trace but visible here
    Xc = X - X.mean(0, keepdims=True)
    return np.stack([(b ** 2).mean(1) for b in np.array_split(Xc, nb, axis=1)], 1)


def battery(model, n_trials, bg, delay, rng, readout_kw=None, name=""):
    readout_kw = readout_kw or {}
    y = np.repeat(np.arange(K), n_trials // K)
    rng.shuffle(y)
    st, delay_feat = model.run_history(y, bg, delay, rng, **readout_kw.get('hist', {}))
    Xr = model.readout(st, bg, False, rng)
    Xp = model.readout(st, bg, True, rng)
    Xn = model.readout(model.reset(st), bg, True, rng)
    # swap: transplant the resident state of a donor trial with label (y+1)%K
    donor = np.array([rng.choice(np.where(y == (k + 1) % K)[0]) for k in y])
    st_sw = {k: v[donor].copy() for k, v in st.items()}
    Xs = model.readout(st_sw, bg, True, rng)
    half = len(y) // 2                                            # train ping-decoder on half, test swap on the other
    c = clf().fit(Xp[:half], y[:half])
    pred = c.predict(Xs[half:])
    res = dict(
        delay=cv_acc(delay_feat, y), rest=cv_acc(Xr, y),
        rest_power=cv_acc(bin_power(Xr), y), delay_power=cv_acc(bin_power(delay_feat), y),
        ping=cv_acc(Xp, y),
        reset=cv_acc(Xn, y),
        swap_donor=float(np.mean(pred == y[donor][half:])),
        swap_own=float(np.mean(pred == y[half:])),
    )
    return res, dict(y=y, Xp=Xp, Xr=Xr, Xn=Xn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='results.json')
    a = ap.parse_args()
    nT = 200 if a.quick else 600
    nperm = 10 if a.quick else 40
    R = {'chance': 1 / K}
    t0 = time.time()

    # ---------------- S1 STF ----------------
    m = STF(seed=1)
    rng = np.random.default_rng(10)
    main1, d1 = battery(m, nT, bg=S1_BG, delay=1000, rng=rng)
    null = perm_null(d1['Xp'], d1['y'], nperm)
    main1['perm_null_p95'] = float(np.percentile(null, 95))
    R['S1_STF_main'] = main1
    print('S1 main', main1, f'{time.time()-t0:.0f}s')

    R['S1_delay_sweep'] = {}
    for dl in ([500, 2000, 4000] if a.quick else [250, 500, 1000, 2000, 3000, 4500, 6000]):
        r, _ = battery(m, nT, bg=S1_BG, delay=dl, rng=np.random.default_rng(dl))
        R['S1_delay_sweep'][dl] = {k: r[k] for k in ('rest', 'ping', 'reset')}
        print(' S1 delay', dl, R['S1_delay_sweep'][dl])

    R['S1_background_sweep'] = {}
    for bg in ([0.5, 2, 10] if a.quick else [0.0, 0.25, 0.5, 1, 2, 5, 10]):
        r, _ = battery(m, nT, bg=bg, delay=1000, rng=np.random.default_rng(int(bg * 10) + 7))
        R['S1_background_sweep'][bg] = {k: r[k] for k in ('delay', 'rest', 'ping', 'reset')}
        print(' S1 bg', bg, R['S1_background_sweep'][bg])

    # ---------------- S2 Cable ----------------
    c = Cable(seed=2)
    rng = np.random.default_rng(20)
    main2, d2 = battery(c, nT, bg=S2_BG, delay=1500, rng=rng)
    null = perm_null(d2['Xp'], d2['y'], nperm)
    main2['perm_null_p95'] = float(np.percentile(null, 95))
    R['S2_CABLE_main'] = main2
    print('S2 main', main2, f'{time.time()-t0:.0f}s')

    # where is carried by when: jitter write strength, compare amplitude-only vs waveform
    R['S2_where_from_when'] = {}
    for jit in [0.0, 0.6]:
        y = np.repeat(np.arange(K), nT // K); rng = np.random.default_rng(30 + int(jit * 10))
        rng.shuffle(y)
        st, _ = c.run_history(y, S2_BG, 1500, rng, jitter=jit)
        Xp = c.readout(st, S2_BG, True, rng)
        Xbase = c.readout(c.reset(st), S2_BG, True, rng).mean(0)   # mean ping response with no resident state
        ev = Xp - Xbase
        amp = ev.sum(1, keepdims=True)
        shape = ev / (np.linalg.norm(ev, axis=1, keepdims=True) + 1e-9)
        peak_t = np.argmax(ev, axis=1)[:, None].astype(float)
        R['S2_where_from_when'][jit] = dict(
            waveform=cv_acc(ev, y), amplitude_only=cv_acc(amp, y),
            shape_only_normalized=cv_acc(shape, y), peak_latency_only=cv_acc(peak_t, y),
            mean_peak_latency_by_item=[float(peak_t[y == k].mean()) for k in range(K)])
        print(' S2 where/when jitter', jit, R['S2_where_from_when'][jit])

    R['S2_delay_sweep'] = {}
    for dl in ([500, 3000, 8000] if a.quick else [250, 1000, 2000, 4000, 6000, 9000]):
        r, _ = battery(c, nT, bg=S2_BG, delay=dl, rng=np.random.default_rng(dl + 1))
        R['S2_delay_sweep'][dl] = {k: r[k] for k in ('rest', 'ping', 'reset')}
        print(' S2 delay', dl, R['S2_delay_sweep'][dl])

    R['S2_background_sweep'] = {}
    for bg in ([0.0, 1.0, 4.0] if a.quick else [0.0, 0.5, 1.0, 2.0, 4.0, 10.0, 30.0]):
        r, _ = battery(c, nT, bg=bg, delay=1500, rng=np.random.default_rng(int(bg * 10) + 99))
        R['S2_background_sweep'][bg] = {k: r[k] for k in ('delay', 'rest', 'ping', 'reset')}
        print(' S2 bg', bg, R['S2_background_sweep'][bg])

    R['runtime_s'] = time.time() - t0
    json.dump(R, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
