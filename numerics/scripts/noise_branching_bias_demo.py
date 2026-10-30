# Colab one-cell demonstration of common-noise misclassification bias.
# Copy this entire file into one Colab code cell and run it.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

plt.close('all')

# Symmetric normalization: V_i^(s) = V_j^(s) = 1 and true gamma = 0.8.
gamma_true = 0.8
omitted_common_covariance = np.array([0.00, 0.05, 0.10, 0.20, 0.25])
gamma_inferred = np.abs(gamma_true + omitted_common_covariance)

bias_table = pd.DataFrame(
    {
        'true_gamma_signal': gamma_true,
        'omitted_common_covariance': omitted_common_covariance,
        'inferred_gamma_signal': gamma_inferred,
        'inconsistent_above_one': gamma_inferred > 1.0,
    }
)
display(bias_table)
bias_table.to_csv('noise_branching_bias_demo.csv', index=False)

beta = np.linspace(0.0, 0.30, 301)
fig, ax = plt.subplots(figsize=(6.8, 4.2))
for true_gamma, color in [(0.4, '#78909c'), (0.7, '#1565c0'), (0.9, '#d84315')]:
    ax.plot(
        beta,
        true_gamma + beta,
        lw=2.3,
        color=color,
        label=f'true gamma = {true_gamma}',
    )
ax.axhline(1.0, color='black', ls='--', lw=1.4, label='physical upper bound')
ax.set(
    xlabel='omitted normalized common covariance',
    ylabel='inferred intrinsic compatibility',
    title='Upward bias from common noise treated as independent',
    xlim=(0.0, 0.30),
    ylim=(0.35, 1.22),
)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig('noise_branching_bias_demo.png', bbox_inches='tight')
plt.show()

# Multi-recorder diagnostic for one identical copied fluctuation.
number_of_recorders = 5
common_variance = 0.10
contamination = common_variance * (
    np.ones((number_of_recorders, number_of_recorders))
    - np.eye(number_of_recorders)
)
print(
    'eigenvalues of diagonal-subtracted/common-cross-omitted contamination:',
    np.linalg.eigvalsh(contamination),
)
print('wrote noise_branching_bias_demo.csv and noise_branching_bias_demo.png')
