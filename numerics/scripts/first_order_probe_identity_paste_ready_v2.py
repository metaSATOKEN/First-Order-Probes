# Colab one-cell paste-ready numerical audit — v2
# Copy this entire file into one Google Colab code cell and run it.
# The output is a numerical audit/simulation, not a physical experiment.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from pathlib import Path

SEED = 20260731
rng = np.random.default_rng(SEED)
plt.close('all')
plt.rcParams.update({'figure.dpi': 130, 'axes.grid': True, 'grid.alpha': 0.25})
print('NumPy', np.__version__, '| seed', SEED)

def spectral_envelope(kind, omega, *, kappa=10.0, center=50.0, width=2.0):
    """Nonnegative one-sided spectral envelope on an rFFT grid."""
    if kind == 'white':
        S = np.ones_like(omega)
    elif kind == 'ou':
        S = 2.0 * kappa / (kappa**2 + omega**2)
    elif kind == 'narrowband':
        S = np.exp(-0.5 * ((omega - center) / width)**2)
    elif kind == 'multiband':
        S = (
            1.0 * np.exp(-0.5 * ((omega - 0.22 * center) / (0.07 * center))**2)
            + 0.7 * np.exp(-0.5 * ((omega - 3.5 * center) / (0.35 * center))**2)
            + 0.03
        )
    else:
        raise ValueError(f'unknown spectrum: {kind}')
    return np.maximum(S, 0.0)


def stationary_periodic_source(kind, N, dt, rng, **kwargs):
    omega = 2.0 * np.pi * np.fft.rfftfreq(N, d=dt)
    S = spectral_envelope(kind, omega, **kwargs)
    Z = rng.normal(size=omega.size) + 1j * rng.normal(size=omega.size)
    X = np.sqrt(S) * Z
    X[0] = 0.0  # zero mean
    if N % 2 == 0:
        X[-1] = 0.0  # avoid an unpaired Nyquist component
    x = np.fft.irfft(X, n=N)
    x -= x.mean()
    x /= x.std()
    return x, omega, S


def first_order_output(x, tau, dt):
    omega = 2.0 * np.pi * np.fft.rfftfreq(x.size, d=dt)
    H = 1.0 / (1.0 - 1j * omega * tau)
    return np.fft.irfft(H * np.fft.rfft(x), n=x.size)


def record_moments(y_i, y_j, tau_i, tau_j):
    a = y_i - np.mean(y_i)
    b = y_j - np.mean(y_j)
    V_i = np.mean(a * a)
    V_j = np.mean(b * b)
    C_ij = np.mean(a * b)
    residual = (tau_i + tau_j) * C_ij - tau_i * V_i - tau_j * V_j
    scale = max(abs(tau_i * V_i + tau_j * V_j), np.finfo(float).tiny)
    gamma = abs(C_ij) / np.sqrt(V_i * V_j)
    w = tau_i * V_i / (tau_j * V_j)
    gamma_min = 2.0 * np.sqrt(tau_i * tau_j) / (tau_i + tau_j)
    gamma_pred = gamma_min * np.cosh(0.5 * np.log(w))
    return {
        'V_i': V_i, 'V_j': V_j, 'C_ij': C_ij,
        'variance_ratio_v': V_i / V_j,
        'balance_ratio_w': w,
        'gamma_measured': gamma,
        'gamma_predicted': gamma_pred,
        'residual': residual,
        'relative_residual': residual / scale,
    }


# Probe and sampling parameters
tau_i = 0.040
tau_j = 0.010
r = tau_i / tau_j
dt = 0.002
N = 2**17
omega_star = 1.0 / np.sqrt(tau_i * tau_j)

source_specs = [
    ('slow narrowband', 'narrowband', {'center': 0.10 * omega_star, 'width': 0.015 * omega_star}),
    ('OU, slow', 'ou', {'kappa': 1.0}),
    ('OU, intermediate', 'ou', {'kappa': 20.0}),
    ('near saturation band', 'narrowband', {'center': omega_star, 'width': 0.025 * omega_star}),
    ('white on grid', 'white', {}),
    ('multiband', 'multiband', {'center': omega_star}),
    ('high narrowband', 'narrowband', {'center': 10.0 * omega_star, 'width': 0.20 * omega_star}),
]

rows = []
records = {}
for label, kind, kwargs in source_specs:
    x, omega, S = stationary_periodic_source(kind, N, dt, rng, **kwargs)
    y_i = first_order_output(x, tau_i, dt)
    y_j = first_order_output(x, tau_j, dt)
    m = record_moments(y_i, y_j, tau_i, tau_j)
    rows.append({'source': label, **m})
    records[label] = (x, y_i, y_j)

results = pd.DataFrame(rows)
display(results[['source', 'variance_ratio_v', 'gamma_measured', 'gamma_predicted', 'relative_residual']])
print('maximum |relative residual| =', results.relative_residual.abs().max())
print('maximum |gamma - prediction| =', np.max(np.abs(results.gamma_measured - results.gamma_predicted)))

# Independent benchmark for the finite-band white row.
omega_grid = 2.0 * np.pi * np.fft.rfftfreq(N, d=dt)
keep = np.ones(omega_grid.size, dtype=bool)
keep[0] = False
if N % 2 == 0:
    keep[-1] = False
omega_paired = omega_grid[keep]
power_i = 1.0 / (1.0 + (omega_paired * tau_i)**2)
power_j = 1.0 / (1.0 + (omega_paired * tau_j)**2)

w_white_continuum = (
    np.arctan((np.pi / dt) * tau_i)
    / np.arctan((np.pi / dt) * tau_j)
)
w_white_discrete = tau_i * power_i.sum() / (tau_j * power_j.sum())

# Each retained complex Gaussian Fourier power has unit relative variance.
var_log_w = np.sum(
    (power_i / power_i.sum() - power_j / power_j.sum())**2
)
sd_white_delta = w_white_discrete * np.sqrt(var_log_w)
w_white_observed = float(
    results.loc[results.source == 'white on grid', 'balance_ratio_w'].iloc[0]
)
z_white = (w_white_observed - w_white_discrete) / sd_white_delta

white_benchmark = pd.DataFrame(
    [{
        'w_continuum_truncation': w_white_continuum,
        'w_discrete_grid_mean_power_ratio': w_white_discrete,
        'w_observed_realization': w_white_observed,
        'delta_method_sd': sd_white_delta,
        'standardized_deviation_z': z_white,
    }]
)
print('\nFinite-band white-source benchmark')
display(white_benchmark)
white_benchmark.to_csv('white_bandlimit_benchmark.csv', index=False)


def gamma_from_v(v, r):
    return (r * v + 1.0) / ((r + 1.0) * np.sqrt(v))

fig, ax = plt.subplots(figsize=(8.2, 5.2))
colors = {2: '#78909c', 4: '#1565c0', 10: '#d84315'}
for rr in [2, 4, 10]:
    z = np.linspace(-2.0, 0.0, 500)
    v = rr**z
    ax.plot(z, gamma_from_v(v, rr), lw=2.5, color=colors[rr], label=f'r = {rr}')

# Simulated points for r = 4
z_sim = np.log(results.variance_ratio_v.to_numpy()) / np.log(r)
ax.scatter(z_sim, results.gamma_measured, s=42, facecolor='white', edgecolor=colors[4], zorder=5, label='simulated spectra (r = 4)')
for x0, y0, label in zip(z_sim, results.gamma_measured, results.source):
    ax.annotate(label, (x0, y0), xytext=(4, 5), textcoords='offset points', fontsize=7)

# OU path for r = 4: kappa from slow to fast
kappas = np.logspace(-3, 5, 500)
v_ou = (1.0 + kappas * tau_j) / (1.0 + kappas * tau_i)
z_ou = np.log(v_ou) / np.log(r)
g_ou = gamma_from_v(v_ou, r)
ax.plot(z_ou, g_ou, '--', color='#0d47a1', lw=2.0, label='OU trajectory (r = 4)')
ax.annotate('increasing $\\kappa$', xy=(z_ou[-40], g_ou[-40]), xytext=(-0.48, 0.93),
            arrowprops={'arrowstyle': '->', 'color': '#0d47a1'}, color='#0d47a1')

ax.set(xlim=(-2.03, 0.03), ylim=(0.5, 1.015),
       xlabel=r'$\ln(V_i/V_j)/\ln r$', ylabel=r'compatibility $\gamma$',
       title='Universal variance-ratio compatibility law')
ax.set_xticks([-2, -1, 0], ['high-frequency\nendpoint', 'variance-balance\nminimum', 'low-frequency\nendpoint'])
ax.legend(fontsize=8, loc='lower right')
fig.tight_layout()
fig.savefig('first_order_compatibility_u_curve.png', bbox_inches='tight')
plt.show()

def block_bootstrap_residual(y_i, y_j, tau_i, tau_j, dt, *, block_duration=None, B=500, seed=1234):
    if block_duration is None:
        block_duration = 10.0 * max(tau_i, tau_j)
    L = max(4, int(round(block_duration / dt)))
    n = min(len(y_i), len(y_j))
    a = np.asarray(y_i[:n])
    b = np.asarray(y_j[:n])
    n_blocks = int(np.ceil(n / L))
    offsets = np.arange(L)
    local_rng = np.random.default_rng(seed)
    values = np.empty(B)
    for q in range(B):
        starts = local_rng.integers(0, n - L + 1, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]).reshape(-1)[:n]
        aa = a[idx]
        bb = b[idx]
        values[q] = record_moments(aa, bb, tau_i, tau_j)['residual']
    return values, L, n_blocks

_, y_i_boot, y_j_boot = records['OU, intermediate']
boot, block_len, n_blocks = block_bootstrap_residual(y_i_boot, y_j_boot, tau_i, tau_j, dt)
point = record_moments(y_i_boot, y_j_boot, tau_i, tau_j)['residual']
ci = np.quantile(boot, [0.025, 0.5, 0.975])
print({'point_residual': point, 'bootstrap_2.5%': ci[0], 'bootstrap_median': ci[1],
       'bootstrap_97.5%': ci[2], 'block_samples': block_len, 'blocks': n_blocks})

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.hist(boot, bins=40, color='#90caf9', edgecolor='white')
ax.axvline(0.0, color='black', lw=1.5, label='theory: 0')
ax.axvline(point, color='#d84315', lw=1.5, label='full-record estimate')
ax.set(xlabel=r'$\widehat{\mathcal{R}}_{ij}$', ylabel='bootstrap count', title='Paired moving-block bootstrap')
ax.legend()
fig.tight_layout()
plt.show()

def exact_ou_probe_transition(kappa, tau_i, tau_j, dt):
    """Exact stationary discrete transition for (x, y_i, y_j)."""
    decay_x = np.exp(-kappa * dt)

    def source_to_probe(tau):
        decay_y = np.exp(-dt / tau)
        if abs(1.0 - kappa * tau) < 1e-10:
            return (dt / tau) * decay_y
        return (decay_x - decay_y) / (1.0 - kappa * tau)

    transition = np.array(
        [
            [decay_x, 0.0, 0.0],
            [source_to_probe(tau_i), np.exp(-dt / tau_i), 0.0],
            [source_to_probe(tau_j), 0.0, np.exp(-dt / tau_j)],
        ]
    )

    v_xi = 1.0 / (1.0 + kappa * tau_i)
    v_xj = 1.0 / (1.0 + kappa * tau_j)
    c_ij = (
        tau_i / (1.0 + kappa * tau_i)
        + tau_j / (1.0 + kappa * tau_j)
    ) / (tau_i + tau_j)
    stationary_covariance = np.array(
        [
            [1.0, v_xi, v_xj],
            [v_xi, v_xi, c_ij],
            [v_xj, c_ij, v_xj],
        ]
    )

    innovation_covariance = (
        stationary_covariance
        - transition @ stationary_covariance @ transition.T
    )
    innovation_covariance = 0.5 * (
        innovation_covariance + innovation_covariance.T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(innovation_covariance)
    innovation_factor = eigenvectors @ np.diag(
        np.sqrt(np.maximum(eigenvalues, 0.0))
    )
    return transition, stationary_covariance, innovation_factor


def discrete_isserlis_residual_sd(
    transition, stationary_covariance, tau_i, tau_j, sample_counts
):
    """Exact discrete Isserlis SD for raw, known-zero-mean moments."""
    moment_pairs = [(1, 2), (1, 1), (2, 2)]
    coefficients = np.array([tau_i + tau_j, -tau_i, -tau_j])
    maximum_count = int(np.max(sample_counts))
    lag_covariance = stationary_covariance.copy()
    residual_autocovariance = np.empty(maximum_count)

    for lag in range(maximum_count):
        value = 0.0
        for u, (a, b) in enumerate(moment_pairs):
            for v, (c, d) in enumerate(moment_pairs):
                value += coefficients[u] * coefficients[v] * (
                    lag_covariance[a, c] * lag_covariance[b, d]
                    + lag_covariance[a, d] * lag_covariance[b, c]
                )
        residual_autocovariance[lag] = value
        lag_covariance = transition @ lag_covariance

    predicted_sd = []
    for count in sample_counts:
        count = int(count)
        weighted_lags = np.dot(
            np.arange(count - 1, 0, -1),
            residual_autocovariance[1:count],
        )
        variance = (
            count * residual_autocovariance[0] + 2.0 * weighted_lags
        ) / count**2
        predicted_sd.append(np.sqrt(max(variance, 0.0)))
    return np.asarray(predicted_sd)


kappa_validation = 20.0
window_samples = np.array([1024, 2048, 4096, 8192])
ensemble_size = 3000
transition, stationary_covariance, innovation_factor = (
    exact_ou_probe_transition(
        kappa_validation, tau_i, tau_j, dt
    )
)
isserlis_sd = discrete_isserlis_residual_sd(
    transition,
    stationary_covariance,
    tau_i,
    tau_j,
    window_samples,
)

ensemble_rng = np.random.default_rng(20260802)
states = ensemble_rng.multivariate_normal(
    np.zeros(3), stationary_covariance, size=ensemble_size
)
cumulative_residual = np.zeros(ensemble_size)
residual_ensembles = {}
checkpoints = set(window_samples.tolist())

for sample_index in range(1, int(window_samples[-1]) + 1):
    output_i = states[:, 1]
    output_j = states[:, 2]
    cumulative_residual += (
        (tau_i + tau_j) * output_i * output_j
        - tau_i * output_i**2
        - tau_j * output_j**2
    )
    if sample_index in checkpoints:
        residual_ensembles[sample_index] = (
            cumulative_residual / sample_index
        ).copy()
    innovations = (
        ensemble_rng.normal(size=(ensemble_size, 3))
        @ innovation_factor.T
    )
    states = states @ transition.T + innovations

empirical_mean = np.array(
    [residual_ensembles[int(n)].mean() for n in window_samples]
)
empirical_sd = np.array(
    [residual_ensembles[int(n)].std(ddof=1) for n in window_samples]
)
window_duration = window_samples * dt

# Continuous-time Gaussian boundary prediction.
boundary_vector = np.array([0.0, tau_i, -tau_j])
boundary_variance = (
    boundary_vector @ stationary_covariance @ boundary_vector
)
boundary_sd = []
for count, duration in zip(window_samples, window_duration):
    endpoint_covariance = (
        boundary_vector
        @ np.linalg.matrix_power(transition, int(count))
        @ stationary_covariance
        @ boundary_vector
    )
    boundary_sd.append(
        np.sqrt(
            max(boundary_variance**2 - endpoint_covariance**2, 0.0)
        ) / duration
    )
boundary_sd = np.asarray(boundary_sd)

finite_window_results = pd.DataFrame(
    {
        'samples': window_samples,
        'duration_T': window_duration,
        'ensemble_mean_residual': empirical_mean,
        'empirical_sd': empirical_sd,
        'isserlis_predicted_sd': isserlis_sd,
        'continuous_boundary_sd': boundary_sd,
        'empirical_to_isserlis_ratio': empirical_sd / isserlis_sd,
    }
)
empirical_slope = np.polyfit(
    np.log(window_duration), np.log(empirical_sd), 1
)[0]
isserlis_slope = np.polyfit(
    np.log(window_duration), np.log(isserlis_sd), 1
)[0]

print('\nIndependent nonperiodic OU finite-window audit')
display(finite_window_results)
print(
    {
        'ensemble_size': ensemble_size,
        'empirical_log_log_slope': empirical_slope,
        'isserlis_log_log_slope': isserlis_slope,
        'maximum_relative_sd_discrepancy': np.max(
            np.abs(empirical_sd / isserlis_sd - 1.0)
        ),
    }
)

finite_window_results.to_csv(
    'ou_finite_window_scaling.csv', index=False
)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.loglog(
    window_duration,
    empirical_sd,
    'o-',
    color='#1565c0',
    lw=2.0,
    label='independent OU ensemble',
)
ax.loglog(
    window_duration,
    isserlis_sd,
    's--',
    color='#d84315',
    lw=1.8,
    label='discrete Isserlis prediction',
)
ax.loglog(
    window_duration,
    isserlis_sd[0] * (window_duration / window_duration[0])**(-1.0),
    ':',
    color='#455a64',
    lw=1.8,
    label=r'$T^{-1}$ reference',
)
ax.set(
    xlabel='record duration T',
    ylabel=r'$\mathrm{SD}(\widehat{\mathcal{R}}_{ij})$',
    title='Nonperiodic finite-window residual scaling',
)
ax.legend()
fig.tight_layout()
fig.savefig('ou_finite_window_scaling.png', bbox_inches='tight')
plt.show()


row = results.loc[results.source == 'near saturation band'].iloc[0]
gain_sensitivity = tau_j * row.V_j - tau_i * row.V_i
tau_i_sensitivity = row.C_ij - row.V_i
print({'source': row.source,
       'gain_sensitivity_dR_deta': gain_sensitivity,
       'tau_i_sensitivity_dR_dtau_i': tau_i_sensitivity,
       'balance_ratio_w': row.balance_ratio_w})

# Independent private components obey the sign rule in ensemble expectation.
local_rng = np.random.default_rng(44)
amp_i, amp_j = 0.25, 0.18
N_private = 2**14
private_rows = []
for _ in range(200):
    base_x, _, _ = stationary_periodic_source('ou', N_private, dt, local_rng, kappa=20.0)
    base_i = first_order_output(base_x, tau_i, dt)
    base_j = first_order_output(base_x, tau_j, dt)
    z_i = local_rng.normal(size=N_private)
    z_j = local_rng.normal(size=N_private)
    p_i = amp_i * first_order_output(z_i, tau_i, dt)
    p_j = amp_j * first_order_output(z_j, tau_j, dt)
    measured = record_moments(base_i + p_i, base_j + p_j, tau_i, tau_j)['residual']
    predicted = -tau_i * np.var(p_i) - tau_j * np.var(p_j)
    private_rows.append((measured, predicted))
private_rows = np.asarray(private_rows)
print({'ensemble_mean_measured_residual': private_rows[:, 0].mean(),
       'ensemble_mean_private_power_prediction': private_rows[:, 1].mean(),
       'mean_difference': (private_rows[:, 0] - private_rows[:, 1]).mean(),
       'finite_record_difference_std': (private_rows[:, 0] - private_rows[:, 1]).std(ddof=1)})

def analyze_recorded_csv(path, tau_i, tau_j, *, col_i='y_i', col_j='y_j', N_i=0.0, N_j=0.0, N_ij=0.0):
    data = pd.read_csv(path)
    a = data[col_i].to_numpy(dtype=float)
    b = data[col_j].to_numpy(dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    V_i_obs = np.mean(a * a)
    V_j_obs = np.mean(b * b)
    C_obs = np.mean(a * b)
    V_i = V_i_obs - N_i
    V_j = V_j_obs - N_j
    C = C_obs - N_ij
    R = (tau_i + tau_j) * C - tau_i * V_i - tau_j * V_j
    gamma = abs(C) / np.sqrt(V_i * V_j)
    return pd.Series({'V_i_signal': V_i, 'V_j_signal': V_j, 'C_signal': C,
                      'gamma_signal': gamma, 'residual': R})

print("Usage: analyze_recorded_csv('record.csv', tau_i, tau_j, N_i=..., N_j=..., N_ij=...)")

# Export the numerical audit table.
results.to_csv('first_order_probe_identity_results.csv', index=False)
print('wrote result tables and figures, including white-band and nonperiodic OU audits')
