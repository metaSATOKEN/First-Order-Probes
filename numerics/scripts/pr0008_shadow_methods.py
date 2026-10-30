# Copyright 2026 Kentaro Sato
# SPDX-License-Identifier: Apache-2.0
"""PR-0008 fixed batch gates. No simulation, truth labels or adaptive selection."""
import math
import numpy as np

GATES = ('G-old', 'G-M', 'G-MC')
POLICIES = ('F*', 'U') + GATES
GROUPS = ('B_M', 'B_L', 'K2_M', 'K200_L', 'V05_L', 'V15_L')
WINDOWS = np.array([0, 1, 0, 1, 1, 1])
CANDIDATES = (('clean', 0), ('GM05', 160), ('GM05', 800),
              ('GP05', 160), ('GP05', 800), ('P02', 160), ('P02', 800))
CARD_SHA = '806e31452451ff02f0c7f0a0b2a9394d69bed3ce4af81270416a4450ee5f9370'


def batch_scores(values, old_alarm, old_valid):
    a = np.asarray(values)
    if a.ndim != 2 or a.shape[1] != 17 or a.dtype != np.float64 or len(a) < 3:
        raise ValueError('Expected float64 Nx17, N>=3')
    scores = np.full(3, np.nan)
    if not np.isfinite(a).all():
        return scores, dict(reason='nonfinite candidate')
    q = a[:, [11, 12]]
    mean, sd = q.mean(0), q.std(0, ddof=1)
    if np.asarray(old_valid).all(): scores[0] = np.mean(old_alarm)
    t = np.full(2, np.nan)
    ok = np.isfinite(sd) & (sd > 0)
    t[ok] = math.sqrt(len(a))*mean[ok]/sd[ok]
    if np.isfinite(t[0]): scores[1] = abs(t[0])
    if np.isfinite(t).all(): scores[2] = max(abs(t[0]), t[1])
    return scores, dict(mean=mean.tolist(), sd=sd.tolist(), signed_T=t.tolist())


def fold_thresholds(clean):
    clean = np.asarray(clean)
    if clean.shape != (25, 6, 3): raise ValueError('clean scores must be 25x6x3')
    out = np.full((5, 2, 3), np.nan)
    for fold in range(5):
        dev = np.arange(25)//5 != fold
        for window in range(2):
            for gate in range(3):
                x = clean[dev][:, WINDOWS == window, gate]
                if np.isfinite(x).all(): out[fold, window, gate] = np.max(x)
    return out


def gate_states(scores, thresholds):
    scores = np.asarray(scores)
    if scores.shape != (25, 6, 7, 3): raise ValueError('scores must be 25x6x7x3')
    # 0 = NOT-VETOED, 1 = HOLD-INCONSISTENT, 2 = HOLD-UNKNOWN.
    out = np.full(scores.shape, 2, np.uint8)
    for r in range(25):
        for g in range(6):
            c = thresholds[r//5, WINDOWS[g]]
            ok = np.isfinite(scores[r, g]) & np.isfinite(c)
            out[r, g][ok] = (scores[r, g] > c)[ok].astype(np.uint8)
    return out


def decode(packed):
    a = np.asarray(packed)
    if a.dtype != np.uint32 or a.ndim != 2 or a.shape[1] != 2 or np.any(a >> 24):
        raise ValueError('Invalid packed decision shape/type/high bits')
    n = (a[:, :, None] >> (4*np.arange(6, dtype=np.uint32))) & 15
    if np.any(((n & 8) == 0) & ((n & 7) != 0)):
        raise ValueError('Alarm without availability')
    return (n & 7) > 0, (n & 8) > 0


def paired(a, b):
    aa, av = decode(a); bb, bv = decode(b); ok = av & bv
    return np.stack([(ok & x).sum(0) for x in
                     (aa & bb, aa & ~bb, ~aa & bb, ~aa & ~bb)], axis=-1)


def select_counts(old, updated, states):
    """old: R,G,E,A,M; updated: R,G,C,E,A,M; returns ... ,policy."""
    o = np.broadcast_to(old[:, :, None], updated.shape)
    return np.stack([o, updated] + [np.where(states[..., i, None, None, None] > 0,
                      o, updated) for i in range(3)], axis=-1)


def independent_decide(values, state, ordered, meta, cut):
    """Scalar-coordinate projection and ranks, independent of frozen matrix multiply."""
    m = state['model']; q0 = values[:, 11]-m['mean'][0]; q1 = values[:, 12]-m['mean'][1]
    s = np.column_stack([q0*w[0]+q1*w[1] for w in m['weights']])
    valid = np.isfinite(values).all(1) & np.isfinite(s).all(1) & bool(state['valid'] and meta['valid'])
    ranks = np.column_stack((np.searchsorted(ordered[:, 0], abs(s[:, 0]), side='left'),
                             np.searchsorted(ordered[:, 1], s[:, 1], side='left')))
    ranks[:, m['disabled']] = 0
    out = np.zeros((len(values), 2), np.uint32)
    for ai in range(2):
        for mi in range(6):
            if mi < 5:
                bs = [s[:, 0] > cut[ai, mi, 0], -s[:, 0] > cut[ai, mi, 1], s[:, 1] > cut[ai, mi, 2]]
            else:
                hit = ranks[:, 0] > cut[ai, mi, 0]
                bs = [hit & (s[:, 0] >= 0), hit & (s[:, 0] < 0), ranks[:, 1] > cut[ai, mi, 0]]
            v = sum((np.asarray(b, np.uint32) << j) for j, b in enumerate(bs))
            out[:, ai] |= np.where(valid, v | 8, 0).astype(np.uint32) << (4*mi)
    return out
