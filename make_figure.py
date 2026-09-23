import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import silent_ping as sp

R = json.load(open('results.json')); ch = R['chance']
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
cols = dict(delay='#999', rest='#bbb', rest_power='#d9a', delay_power='#c8a', ping='#2a6', reset='#c44', swap_donor='#36c')

def bars(a, d, keys, title):
    a.bar(range(len(keys)), [d[k] for k in keys], color=[cols[k] for k in keys])
    a.axhline(ch, ls=':', c='k', lw=1); a.axhline(d['perm_null_p95'], ls='--', c='k', lw=1)
    a.set_xticks(range(len(keys))); a.set_xticklabels(keys, rotation=30, ha='right')
    a.set_ylim(0, 1); a.set_ylabel('decoding accuracy (5-fold CV)'); a.set_title(title)

bars(ax[0, 0], R['S1_STF_main'], ['delay', 'rest', 'rest_power', 'ping', 'reset', 'swap_donor'],
     f"S1 STF synapses (bg {sp.S1_BG} Hz, delay 1 s)")
bars(ax[0, 1], R['S2_CABLE_main'], ['delay', 'rest', 'rest_power', 'delay_power', 'ping', 'reset', 'swap_donor'],
     f"S2 cable neuron, resident tag (bg {sp.S2_BG}, delay 1500)")

# evoked soma echo per item (S2)
c = sp.Cable(); rng = np.random.default_rng(5)
y = np.repeat(np.arange(sp.K), 60); st, _ = c.run_history(y, 0.0, 1500, rng)
ev = c.readout(st, 0.0, True, rng) - c.readout(c.reset(st), 0.0, True, rng)
a = ax[0, 2]
for k in range(sp.K):
    a.plot(ev[y == k].mean(0), label=f'tag at compartment {c.loc[k]}')
a.set_title('S2: ping echo minus no-state echo, at the soma'); a.set_xlabel('steps after ping'); a.legend(fontsize=8)

a = ax[1, 0]
for key, lab, unit in [('S1_delay_sweep', 'S1 STF', 1.0), ('S2_delay_sweep', 'S2 cable', 1.0)]:
    xs = sorted(R[key], key=float); a.plot([float(x) for x in xs], [R[key][x]['ping'] for x in xs], 'o-', label=lab + ' ping')
    a.plot([float(x) for x in xs], [R[key][x]['rest'] for x in xs], 'x:', label=lab + ' rest')
a.axhline(ch, ls=':', c='k'); a.set_xlabel('delay (ms for S1, steps for S2)'); a.set_title('hidden state decays; ping tracks it'); a.legend(fontsize=8); a.set_ylim(0, 1)

a = ax[1, 1]
for key, lab in [('S1_background_sweep', 'S1'), ('S2_background_sweep', 'S2')]:
    xs = sorted(R[key], key=float); xv = [float(x) + 0.1 for x in xs]
    a.semilogx(xv, [R[key][x]['ping'] for x in xs], 'o-', label=lab + ' ping')
    a.semilogx(xv, [R[key][x]['delay'] for x in xs], 'x:', label=lab + ' delay (listening)')
a.axhline(ch, ls=':', c='k'); a.set_xlabel('background drive (+0.1, log)'); a.set_title('background erodes the ping advantage'); a.legend(fontsize=8); a.set_ylim(0, 1)

a = ax[1, 2]; W = R['S2_where_from_when']; ks = ['waveform', 'shape_only_normalized', 'amplitude_only', 'peak_latency_only']
w = 0.38
for i, j in enumerate(sorted(W, key=float)):
    a.bar(np.arange(len(ks)) + (i - 0.5) * w, [W[j][k] for k in ks], w, label=f'write-strength jitter ±{float(j):.0%}')
a.axhline(ch, ls=':', c='k'); a.set_xticks(range(len(ks))); a.set_xticklabels(ks, rotation=20, ha='right'); a.set_ylim(0, 1)
a.set_title('S2: WHERE the silent tag is, from one soma trace'); a.legend(fontsize=8)
fig.suptitle('SilentPing — neutral ping reads activity-silent state (dotted = chance, dashed = permutation p95)')
fig.tight_layout(); fig.savefig('silent_ping.png', dpi=110)
print('ok')
